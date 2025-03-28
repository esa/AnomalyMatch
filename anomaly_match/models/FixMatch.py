#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import sys
import os
from tqdm.auto import tqdm
from pathlib import Path

from loguru import logger

from anomaly_match.utils.consistency_loss import consistency_loss
from anomaly_match.utils.cross_entropy_loss import cross_entropy_loss
from anomaly_match.utils.accuracy import accuracy
from anomaly_match.utils.save_cfg import save_cfg
from anomaly_match.datasets.data_utils import get_data_loader


class FixMatch:
    def __init__(
        self,
        net_builder,
        num_classes,
        in_channels,
        ema_m,
        T,
        p_cutoff,
        lambda_u,
        hard_label=True,
        logger=None,
    ):
        """FixMatch implementation for semi-supervised learning.

        This class implements the FixMatch algorithm for semi-supervised learning,
        which combines consistency regularization with pseudo-labeling.

        Args:
            net_builder: Function that builds the backbone network
            num_classes: Number of classification classes
            in_channels: Number of input image channels
            ema_m: Momentum for exponential moving average of evaluation model
            T: Temperature parameter for sharpening predictions
            p_cutoff: Confidence threshold for pseudo-labeling
            lambda_u: Weight for unsupervised loss component
            hard_label: If True, uses hard pseudo-labels, otherwise soft labels
            logger: Logger instance for outputting information
        """
        super(FixMatch, self).__init__()

        # Store parameters
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # Create two versions of the model: one for training and one for evaluation with EMA
        self.train_model = net_builder(num_classes=num_classes, in_channels=in_channels)
        self.eval_model = net_builder(num_classes=num_classes, in_channels=in_channels)
        self.T = T
        self.p_cutoff = p_cutoff
        self.lambda_u = lambda_u
        self.use_hard_label = hard_label

        self.optimizer = None
        self.scheduler = None

        # Iteration tracking
        self.it = 0
        self.total_it = 0
        self.best_eval_acc = 0.0
        self.best_it = 0

        # Logging
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        # Initialize evaluation model as a copy of training model
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net

        self.eval_model.eval()

    @torch.no_grad()
    def _eval_model_update(self):
        """Update the evaluation model using exponential moving average of the training model weights."""
        train_model_params = (
            self.train_model.module.parameters()
            if hasattr(self.train_model, "module")
            else self.train_model.parameters()
        )
        for param_train, param_eval in zip(train_model_params, self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1 - self.ema_m))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, cfg, lb_dset, ulb_dset, eval_dset=None):
        """Set up data loaders for training and evaluation.

        Args:
            cfg: Configuration object with dataloader parameters
            lb_dset: Labeled dataset
            ulb_dset: Unlabeled dataset
            eval_dset: Evaluation dataset (optional)
        """
        logger.debug("Setting up data loaders")

        loader_dict = {}
        dset_dict = {"train_lb": lb_dset, "train_ulb": ulb_dset, "eval": eval_dset}

        loader_dict["train_lb"] = get_data_loader(
            dset_dict["train_lb"],
            cfg.batch_size,
            data_sampler="RandomSampler",
            num_iters=cfg.num_train_iter,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            use_weighted_sampler=cfg.oversample,
        )

        loader_dict["train_ulb"] = get_data_loader(
            dset_dict["train_ulb"],
            cfg.batch_size * cfg.uratio,
            data_sampler="RandomSampler",
            num_iters=cfg.num_train_iter,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        # Only create eval loader if eval_dset is provided
        if eval_dset is not None:
            actual_eval_batch_size = min(cfg.eval_batch_size, len(dset_dict["eval"]))
            loader_dict["eval"] = get_data_loader(
                dset_dict["eval"], actual_eval_batch_size, num_workers=0
            )

        self.loader_dict = loader_dict

    def set_optimizer(self, optimizer, scheduler=None):
        """Set the optimizer and optional scheduler.

        Args:
            optimizer: PyTorch optimizer for training
            scheduler: Optional learning rate scheduler
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, cfg, progressbar=None, progress_callback=None):
        """Train the model using the FixMatch algorithm.

        Args:
            cfg: Configuration object with training parameters
            progressbar: Optional tqdm progress bar
            progress_callback: Optional callback function for reporting progress

        Returns:
            dict: Evaluation metrics after training
        """
        ngpus_per_node = torch.cuda.device_count()

        logger.info(
            f"Starting FixMatch training for {cfg.num_train_iter} iterations on {ngpus_per_node} GPUs"
        )

        self.it = 0

        # Set model to training mode
        self.train_model.train()

        # Set up progress tracking
        progressbar = tqdm(
            desc="Training...",
            total=cfg.num_train_iter + self.total_it,
            file=sys.stdout,
            disable=(cfg.log_level not in ["DEBUG", "TRACE"]),
        )
        progressbar.update(self.total_it)
        progressbar.refresh()

        # Main training loop
        for (x_lb, y_lb, _), (x_ulb_w, x_ulb_s, _) in zip(
            self.loader_dict["train_lb"], self.loader_dict["train_ulb"]
        ):
            # Check if we've reached target iterations
            if self.it >= cfg.num_train_iter:
                logger.debug(f"Training finished after {self.total_it} iterations")
                break

            # Get batch sizes
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]

            # Log training details at TRACE level
            logger.trace(
                f"Iteration {self.it}: labeled batch shape: {x_lb.shape}, "
                f"label distribution: {y_lb.unique(return_counts=True)}"
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                x_lb, x_ulb_w, x_ulb_s = (
                    x_lb.cuda(cfg.gpu),
                    x_ulb_w.cuda(cfg.gpu),
                    x_ulb_s.cuda(cfg.gpu),
                )
                y_lb = y_lb.cuda(cfg.gpu)

            # Combine inputs for efficient processing
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

            # Forward pass and loss calculation
            logits = self.train_model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            del logits  # Free memory

            # Calculate supervised loss
            sup_loss = cross_entropy_loss(logits_x_lb, y_lb, reduction="mean")

            # Calculate unsupervised loss with consistency regularization
            unsup_loss, mask = consistency_loss(
                logits_x_ulb_w,
                logits_x_ulb_s,
                "ce",
                self.T,
                self.p_cutoff,
                use_hard_labels=cfg.hard_label,
            )

            # Combine losses
            total_loss = sup_loss + self.lambda_u * unsup_loss

            # Backpropagation and parameter updates
            total_loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            self.train_model.zero_grad()

            # Update evaluation model with EMA
            with torch.no_grad():
                self._eval_model_update()
                train_accuracy = accuracy(logits_x_lb, y_lb)[0]

            # Store metrics for logging
            metrics = {
                "sup_loss": sup_loss.detach().item(),
                "unsup_loss": unsup_loss.detach().item(),
                "total_loss": total_loss.detach().item(),
                "mask_ratio": (1.0 - mask.detach()).item(),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "accuracy": train_accuracy.item(),
            }

            # Update progress information
            progressbar.set_postfix_str(f"Loss={metrics['total_loss']:.3e}")
            progressbar.update(1)
            if progress_callback:
                progress_callback(self.it, cfg.num_train_iter)
            progressbar.refresh()

            # Update widget progress bar if provided
            if cfg.progress_bar:
                cfg.progress_bar.value = (self.it + 1.0) / cfg.num_train_iter

            # Periodic evaluation
            if cfg.num_eval_iter > 0 and self.it % cfg.num_eval_iter == 0 and self.it > 0:
                progressbar.close()
                logger.debug(f"Performing evaluation at {self.total_it} iterations")

                # Run evaluation
                eval_dict = self.evaluate(cfg=cfg)

                # Update best accuracy if needed
                if eval_dict["eval/top-1-acc"] > self.best_eval_acc:
                    self.best_eval_acc = eval_dict["eval/top-1-acc"]
                    self.best_it = self.total_it

                # Log evaluation results
                self.print_fn(
                    f"Iteration {self.total_it}, EMA model used: {hasattr(self, 'eval_model')}, "
                    f"Accuracy: {eval_dict['eval/top-1-acc']:.4f}, Loss: {eval_dict['eval/loss']:.4e}, "
                    f"AUROC: {eval_dict['eval/auroc']:.4f}, AUPRC: {eval_dict['eval/auprc']:.4f}, "
                    f"Best accuracy: {self.best_eval_acc:.4f} at iteration {self.best_it}"
                )

                # Create new progress bar
                progressbar = tqdm(
                    desc="Training...",
                    total=cfg.num_train_iter + self.total_it - self.it,
                    file=sys.stdout,
                )
                progressbar.update(self.total_it)
                progressbar.refresh()

            # Increment iteration counters
            self.it += 1
            self.total_it += 1

        progressbar.close()

        # Final evaluation
        eval_dict = self.evaluate(cfg=cfg)
        eval_dict.update({"eval/best_acc": self.best_eval_acc, "eval/best_it": self.best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, cfg, eval_loader=None, progress_callback=None):
        """Evaluate the model on a dataset.

        Args:
            cfg: Configuration object
            eval_loader: Optional custom evaluation dataloader
            progress_callback: Optional callback function for reporting progress

        Returns:
            dict: Dictionary of evaluation metrics
        """
        if cfg.test_ratio <= 0:
            logger.info("Test ratio is 0, skipping evaluation")
            return {}

        # Clear GPU memory before evaluation
        torch.cuda.empty_cache()

        # Determine which model to use for evaluation
        use_ema = hasattr(self, "eval_model")
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()

        # Use provided loader or default
        if eval_loader is None:
            eval_loader = self.loader_dict["eval"]

        # Initialize metrics
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        total_len = len(eval_loader)
        confusion_matrix = torch.zeros(2, 2)
        predictions_and_labels = {}
        all_probs = []
        all_labels = []

        # Evaluate all batches
        logger.debug(f"Starting evaluation over {total_len} batches")
        for batch_idx, (x, y, filename) in enumerate(eval_loader):
            # Move to GPU if available
            if torch.cuda.is_available():
                x, y = x.cuda(cfg.gpu), y.cuda(cfg.gpu)

            # Process batch
            num_batch = x.shape[0]
            total_num += num_batch

            # Forward pass
            logits = eval_model(x)
            loss = F.cross_entropy(logits, y.long(), reduction="mean")
            pred = torch.max(logits, dim=-1)[1]
            probs = F.softmax(logits, dim=-1)[:, 1]  # Probability of anomaly class

            # Calculate accuracy
            acc = torch.sum(pred == y)

            # Update confusion matrix
            for i in range(y.shape[0]):
                confusion_matrix[y[i], pred[i]] += 1

            # Accumulate metrics
            total_loss += loss.detach() * num_batch
            total_acc += acc.detach()

            # Store data for ROC and PR curves
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Store predictions and labels
            predictions_and_labels.update(
                {filename[i]: (probs[i].detach(), y[i].detach()) for i in range(num_batch)}
            )

            # Update progress
            if progress_callback:
                progress_callback(batch_idx + 1, total_len)

        # Restore train mode if necessary
        if not use_ema:
            eval_model.train()

        # Calculate final metrics
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            # Compute AUROC and AUPRC
            auroc = roc_auc_score(all_labels, all_probs)
            precision, recall, _ = precision_recall_curve(all_labels, all_probs)
            auprc = auc(recall, precision)

            logger.debug(f"Evaluated {total_num} samples")
            logger.debug(
                f"Loss: {total_loss / total_num:.4f}, Top-1 Acc: {total_acc / total_num:.4f}"
            )
            logger.info(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
            logger.info(f"Confusion Matrix: {confusion_matrix}")

            # Sort predictions by confidence difference (most uncertain first)
            predictions_and_labels = {
                k: v
                for k, v in sorted(
                    predictions_and_labels.items(),
                    key=lambda item: abs(item[1][0].item() - item[1][1].item()),
                    reverse=True,
                )
            }
        else:
            logger.warning("Not enough distinct class labels for ROC/PRC calculation")
            auroc = 0.0
            auprc = 0.0
            precision = []
            recall = []

        # Return metrics dictionary
        return {
            "eval/loss": total_loss / total_num if total_num > 0 else float("inf"),
            "eval/top-1-acc": total_acc / total_num if total_num > 0 else 0.0,
            "eval/auroc": auroc,
            "eval/auprc": auprc,
            "eval/confusion_matrix": confusion_matrix,
            "eval/predictions_and_labels": predictions_and_labels,
            "eval/roc_data": (all_labels, all_probs),
            "eval/precision_recall": (precision, recall),
        }

    def get_scored_binary_unlabeled_samples(
        self, data_loader, target_class, cfg, N_to_load=None, progress_callback=None, data_iter=None
    ):
        """Evaluate and score unlabeled samples for the given target class.

        Args:
            data_loader: DataLoader for unlabeled data
            target_class: Class index to score against (typically 1 for anomaly)
            cfg: Configuration object
            N_to_load: Maximum number of samples to evaluate
            progress_callback: Optional callback for progress reporting
            data_iter: Optional iterator to continue from a previous run

        Returns:
            tuple: (scores, images, filenames, data_iterator)
                - scores: Tensor of anomaly scores
                - images: Tensor of corresponding images
                - filenames: List of filenames for the images
                - data_iterator: Iterator for continuing evaluation
        """
        logger.debug("Getting scores for unlabeled samples")

        # Determine which model to use
        use_ema = hasattr(self, "eval_model")
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()

        # Initialize result containers
        scores = []
        imgs = []
        filenames = []
        images_evaluated = 0
        total_images = N_to_load if N_to_load is not None else len(data_loader.dataset)

        # Create iterator if not provided
        if data_iter is None:
            data_iter = iter(data_loader)

        # Default N_to_load if not specified
        if N_to_load is None:
            N_to_load = len(data_loader.dataset)

        # Process batches until we reach N_to_load
        while images_evaluated < N_to_load:
            try:
                # Get next batch
                x, _, filename = next(data_iter)
            except StopIteration:
                # End of data
                logger.debug("Reached end of dataloader")
                break

            # Move to GPU if available
            if torch.cuda.is_available():
                x = x.cuda()

            # Get predictions and scores
            with torch.no_grad():
                logits = eval_model(x)
                batch_scores = F.softmax(logits, dim=-1)[:, target_class].detach()
                scores.append(batch_scores)
                imgs.append(x.cpu())
                filenames.extend(filename)

            # Update counters and progress
            images_evaluated += x.size(0)
            if progress_callback:
                progress_callback(images_evaluated, total_images)

        # Handle empty result case
        if not scores:
            logger.warning("No samples evaluated")
            return torch.tensor([]), torch.tensor([]), [], data_iter

        # Combine results
        scores = torch.cat(scores)
        imgs = torch.cat(imgs)

        # Log statistics
        logger.debug(f"Evaluated {len(scores)} samples")
        logger.debug(
            f"Score stats - min: {scores.min():.4f}, max: {scores.max():.4f}, "
            f"mean: {scores.mean():.4f}, std: {scores.std():.4f}"
        )

        # Sort by scores (highest anomaly score first)
        scores, indices = scores.sort(descending=True)
        imgs = imgs[indices.cpu()]
        filenames = [filenames[i] for i in indices.cpu()]

        return scores, imgs, filenames, data_iter

    def save_run(self, save_name, save_path, cfg=None):
        """Save a training run's model weights and configuration.

        Args:
            save_name: Filename for the saved model
            save_path: Directory path for saving
            cfg: Optional configuration to save alongside the model
        """
        save_filename = os.path.join(save_path, save_name)

        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Handle distributed training case
        train_model = (
            self.train_model.module if hasattr(self.train_model, "module") else self.train_model
        )
        eval_model = (
            self.eval_model.module if hasattr(self.eval_model, "module") else self.eval_model
        )

        # Save model state
        torch.save(
            {
                "train_model": train_model.state_dict(),
                "eval_model": eval_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "it": self.it,
            },
            save_filename,
        )

        # Save configuration if provided
        if cfg is not None:
            save_cfg(cfg)

        logger.info(f"Model saved to: {save_filename}")

    def save_model(self, cfg):
        """Save the current model to the path specified in the config.

        Args:
            cfg: Configuration containing model_path for saving
        """
        logger.info(f"Saving model to {cfg.model_path}")

        # Create directory if needed
        dir_path = Path(cfg.model_path).parent
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Handle distributed training case
        train_model = (
            self.train_model.module if hasattr(self.train_model, "module") else self.train_model
        )
        eval_model = (
            self.eval_model.module if hasattr(self.eval_model, "module") else self.eval_model
        )

        # Save model state
        torch.save(
            {
                "train_model": train_model.state_dict(),
                "eval_model": eval_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "it": self.it,
            },
            cfg.model_path,
        )

    def load_model(self, cfg):
        """Load a saved model from the path specified in the config.

        Args:
            cfg: Configuration containing model_path to load from
        """
        logger.info(f"Loading model from {cfg.model_path}")

        # Verify path exists
        if not os.path.exists(cfg.model_path):
            logger.error(f"Model path {cfg.model_path} does not exist")
            return

        # Load checkpoint
        checkpoint = torch.load(cfg.model_path, weights_only=False)

        # Handle distributed training case
        train_model = (
            self.train_model.module if hasattr(self.train_model, "module") else self.train_model
        )
        eval_model = (
            self.eval_model.module if hasattr(self.eval_model, "module") else self.eval_model
        )

        # Restore model components from checkpoint
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if "train_model" in key:
                    train_model.load_state_dict(checkpoint[key])
                elif "eval_model" in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == "it":
                    self.it = checkpoint[key]
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                logger.debug(f"Checkpoint component loaded: {key}")
            else:
                logger.debug(f"Checkpoint component skipped: {key}")
