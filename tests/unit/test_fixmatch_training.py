#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Tests for FixMatch train() and evaluate() methods.

Uses a lightweight test-cnn model with tiny synthetic datasets
to exercise the actual training and evaluation code paths on CPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from anomaly_match.models.FixMatch import FixMatch
from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.get_net_builder import get_net_builder
from anomaly_match.utils.get_optimizer import get_optimizer

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


class LabeledDataset(Dataset):
    """Minimal labeled dataset returning (image, target, filename) tuples."""

    def __init__(self, n_samples: int = 8, num_classes: int = 2, image_size: int = 32):
        self.data = torch.randn(n_samples, 3, image_size, image_size)
        # Ensure both classes are present
        self.targets = torch.tensor([i % num_classes for i in range(n_samples)])
        self.filenames = [f"img_{i}.png" for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.filenames[idx]


class UnlabeledDataset(Dataset):
    """Minimal unlabeled dataset returning (weak_aug, strong_aug, dummy_target) tuples."""

    def __init__(self, n_samples: int = 16, image_size: int = 32):
        self.data_w = torch.randn(n_samples, 3, image_size, image_size)
        self.data_s = torch.randn(n_samples, 3, image_size, image_size)
        # Dummy targets (FixMatch ignores unlabeled targets, but the loader yields them)
        self.targets = torch.full((n_samples,), -1)

    def __len__(self):
        return len(self.data_w)

    def __getitem__(self, idx):
        return self.data_w[idx], self.data_s[idx], self.targets[idx]


class EvalDataset(Dataset):
    """Minimal evaluation dataset returning (image, target, filename) tuples.

    Both classes are guaranteed present so AUROC/AUPRC can be computed.
    """

    def __init__(self, n_samples: int = 8, num_classes: int = 2, image_size: int = 32):
        self.data = torch.randn(n_samples, 3, image_size, image_size)
        self.targets = torch.tensor([i % num_classes for i in range(n_samples)])
        self.filenames = [f"eval_{i}.png" for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.filenames[idx]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IMG_SIZE = 32
NUM_TRAIN_ITER = 2
BATCH_SIZE = 4


@pytest.fixture()
def cfg():
    """Minimal config for CPU-only training with 2 iterations."""
    c = get_default_cfg()
    c.batch_size = BATCH_SIZE
    c.uratio = 1
    c.num_train_iter = NUM_TRAIN_ITER
    c.num_workers = 0
    c.pin_memory = False
    c.eval_batch_size = BATCH_SIZE
    c.oversample = False
    c.gpu = 0
    c.hard_label = True
    c.test_ratio = 0.5
    c.num_eval_iter = -1  # no mid-training eval by default
    c.log_level = "INFO"
    return c


def _build_fixmatch(session_tracker=None):
    """Construct a FixMatch instance backed by test-cnn.

    Mirrors training_process.py: moves models to CUDA when available,
    since FixMatch.train()/evaluate() move data to CUDA unconditionally.
    """
    net_builder = get_net_builder("test-cnn", pretrained=False, in_channels=3)
    model = FixMatch(
        net_builder=net_builder,
        num_classes=2,
        in_channels=3,
        ema_m=0.999,
        T=0.5,
        p_cutoff=0.95,
        lambda_u=1.0,
        session_tracker=session_tracker,
    )
    if torch.cuda.is_available():
        model.train_model.cuda(0)
        model.eval_model.cuda(0)
    optimizer = get_optimizer(model.train_model, name="SGD", lr=0.01)
    model.set_optimizer(optimizer)
    return model


def _attach_loaders(model, cfg, eval_dset=True):
    """Create tiny DataLoaders and attach them to the model via loader_dict."""
    lb_loader = DataLoader(
        LabeledDataset(n_samples=8, image_size=IMG_SIZE),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    ulb_loader = DataLoader(
        UnlabeledDataset(n_samples=16, image_size=IMG_SIZE),
        batch_size=cfg.batch_size * cfg.uratio,
        shuffle=True,
        drop_last=True,
    )
    loader_dict = {"train_lb": lb_loader, "train_ulb": ulb_loader}

    if eval_dset:
        eval_loader = DataLoader(
            EvalDataset(n_samples=8, image_size=IMG_SIZE),
            batch_size=cfg.eval_batch_size,
        )
        loader_dict["eval"] = eval_loader

    model.loader_dict = loader_dict


@pytest.fixture()
def model_with_loaders(cfg):
    """FixMatch model with loaders attached, ready for train()/evaluate()."""
    model = _build_fixmatch()
    _attach_loaders(model, cfg)
    return model


# ---------------------------------------------------------------------------
# Tests: train()
# ---------------------------------------------------------------------------


class TestFixMatchTrain:
    """Tests for FixMatch.train()."""

    def test_train_completes_and_advances_iteration(self, cfg, model_with_loaders):
        """Train for 2 iterations; verify it and total_it advance."""
        eval_dict = model_with_loaders.train(cfg)

        assert model_with_loaders.it == cfg.num_train_iter
        assert model_with_loaders.total_it == cfg.num_train_iter
        # Final evaluate() is called, so we should get a non-empty dict
        assert isinstance(eval_dict, dict)
        assert "eval/top-1-acc" in eval_dict

    def test_train_updates_ema_model(self, cfg, model_with_loaders):
        """After training, eval_model params should differ from initial snapshot."""
        # Snapshot eval params before training
        initial_params = [p.clone() for p in model_with_loaders.eval_model.parameters()]

        model_with_loaders.train(cfg)

        # At least some parameters should have changed via EMA
        any_changed = any(
            not torch.equal(p_init, p_now)
            for p_init, p_now in zip(initial_params, model_with_loaders.eval_model.parameters())
        )
        assert any_changed, "EMA model parameters should have been updated during training"

    def test_train_with_progress_callback(self, cfg, model_with_loaders):
        """progress_callback should be invoked once per iteration."""
        callback = MagicMock()
        model_with_loaders.train(cfg, progress_callback=callback)

        assert callback.call_count == cfg.num_train_iter
        # Each call receives (current_it, total_iters)
        for call_args in callback.call_args_list:
            it_arg, total_arg = call_args[0]
            assert total_arg == cfg.num_train_iter
            assert 0 <= it_arg < cfg.num_train_iter

    def test_train_with_hard_label_false(self, cfg, model_with_loaders):
        """Train with soft pseudo-labels (hard_label=False)."""
        cfg.hard_label = False
        eval_dict = model_with_loaders.train(cfg)

        assert model_with_loaders.it == cfg.num_train_iter
        assert isinstance(eval_dict, dict)
        assert "eval/loss" in eval_dict

    def test_train_with_scheduler(self, cfg):
        """Train with a learning-rate scheduler attached."""
        model = _build_fixmatch()
        _attach_loaders(model, cfg)

        scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1, gamma=0.9)
        model.set_optimizer(model.optimizer, scheduler)

        eval_dict = model.train(cfg)
        assert model.it == cfg.num_train_iter
        assert isinstance(eval_dict, dict)

    def test_train_with_session_tracker(self, cfg):
        """SessionTracker.update_model_iteration is called each iteration."""
        tracker = SessionTracker(session_name="test")
        tracker.start_new_session_iteration()

        model = _build_fixmatch(session_tracker=tracker)
        _attach_loaders(model, cfg)

        model.train(cfg)

        # The tracker should have recorded iterations
        assert tracker.total_model_iterations == cfg.num_train_iter

    def test_train_cumulative_total_it(self, cfg, model_with_loaders):
        """total_it accumulates across multiple train() calls."""
        model_with_loaders.train(cfg)
        first_total = model_with_loaders.total_it

        # Reset loaders for a second round (loaders are consumed by zip)
        _attach_loaders(model_with_loaders, cfg)
        model_with_loaders.train(cfg)

        assert model_with_loaders.total_it == first_total + cfg.num_train_iter

    def test_train_with_mid_training_eval(self, cfg):
        """Set num_eval_iter so evaluation triggers during training."""
        cfg.num_train_iter = 3
        cfg.num_eval_iter = 1  # evaluate every iteration (except the first)

        model = _build_fixmatch()

        # Need enough batches for 3 iterations with drop_last=True
        lb_loader = DataLoader(
            LabeledDataset(n_samples=32, image_size=IMG_SIZE),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        ulb_loader = DataLoader(
            UnlabeledDataset(n_samples=32, image_size=IMG_SIZE),
            batch_size=cfg.batch_size * cfg.uratio,
            shuffle=True,
            drop_last=True,
        )
        eval_loader = DataLoader(
            EvalDataset(n_samples=8, image_size=IMG_SIZE),
            batch_size=cfg.eval_batch_size,
        )
        model.loader_dict = {
            "train_lb": lb_loader,
            "train_ulb": ulb_loader,
            "eval": eval_loader,
        }

        eval_dict = model.train(cfg)

        assert model.it == cfg.num_train_iter
        # best_eval_acc should be updated since mid-training eval ran
        assert model.best_eval_acc >= 0.0
        assert isinstance(eval_dict, dict)

    def test_train_with_zero_test_ratio(self, cfg, model_with_loaders):
        """When test_ratio <= 0, evaluate() returns empty dict."""
        cfg.test_ratio = 0.0
        eval_dict = model_with_loaders.train(cfg)
        assert eval_dict == {}


# ---------------------------------------------------------------------------
# Tests: evaluate()
# ---------------------------------------------------------------------------


class TestFixMatchEvaluate:
    """Tests for FixMatch.evaluate()."""

    def test_evaluate_returns_metrics(self, cfg, model_with_loaders):
        """evaluate() produces a dict with all expected keys."""
        eval_dict = model_with_loaders.evaluate(cfg=cfg)

        assert isinstance(eval_dict, dict)
        assert "eval/top-1-acc" in eval_dict
        assert "eval/loss" in eval_dict
        assert "eval/auroc" in eval_dict
        assert "eval/auprc" in eval_dict
        assert "eval/confusion_matrix" in eval_dict
        assert "eval/predictions_and_labels" in eval_dict
        assert "eval/roc_data" in eval_dict
        assert "eval/precision_recall" in eval_dict

    def test_evaluate_with_custom_loader(self, cfg, model_with_loaders):
        """evaluate() accepts an external eval_loader."""
        custom_loader = DataLoader(
            EvalDataset(n_samples=6, image_size=IMG_SIZE),
            batch_size=3,
        )
        eval_dict = model_with_loaders.evaluate(cfg=cfg, eval_loader=custom_loader)

        assert isinstance(eval_dict, dict)
        assert "eval/top-1-acc" in eval_dict

    def test_evaluate_with_progress_callback(self, cfg, model_with_loaders):
        """progress_callback receives (batch_idx+1, total_batches)."""
        callback = MagicMock()
        model_with_loaders.evaluate(cfg=cfg, progress_callback=callback)

        assert callback.call_count > 0
        # Last call should have batch_idx+1 == total_len
        last_call = callback.call_args_list[-1]
        batch_idx_plus_one, total_len = last_call[0]
        assert batch_idx_plus_one == total_len

    def test_evaluate_skipped_when_test_ratio_zero(self, cfg, model_with_loaders):
        """If test_ratio <= 0, evaluate() returns an empty dict."""
        cfg.test_ratio = 0.0
        eval_dict = model_with_loaders.evaluate(cfg=cfg)
        assert eval_dict == {}

    def test_evaluate_roc_data_has_labels_and_probs(self, cfg, model_with_loaders):
        """roc_data tuple should contain lists of labels and probabilities."""
        eval_dict = model_with_loaders.evaluate(cfg=cfg)
        labels, probs = eval_dict["eval/roc_data"]
        assert len(labels) == len(probs)
        assert len(labels) > 0

    def test_evaluate_single_class_labels(self, cfg):
        """When only one class in eval set, auroc/auprc should be 0."""
        model = _build_fixmatch()

        # Create eval dataset with only one class
        single_class_dset = EvalDataset(n_samples=4, image_size=IMG_SIZE)
        single_class_dset.targets = torch.zeros(4, dtype=torch.long)

        eval_loader = DataLoader(single_class_dset, batch_size=4)
        model.loader_dict = {"eval": eval_loader}

        eval_dict = model.evaluate(cfg=cfg, eval_loader=eval_loader)

        assert eval_dict["eval/auroc"] == 0.0
        assert eval_dict["eval/auprc"] == 0.0

    def test_evaluate_after_training(self, cfg, model_with_loaders):
        """Evaluate after training to ensure trained model evaluates correctly."""
        model_with_loaders.train(cfg)

        # Rebuild loaders for a fresh evaluation
        _attach_loaders(model_with_loaders, cfg)

        eval_dict = model_with_loaders.evaluate(cfg=cfg)
        assert isinstance(eval_dict, dict)
        assert "eval/top-1-acc" in eval_dict
        assert eval_dict["eval/loss"] < float("inf")


# ---------------------------------------------------------------------------
# Tests: set_data_loader via the actual FixMatch method
# ---------------------------------------------------------------------------


class TestFixMatchSetDataLoader:
    """Tests for FixMatch.set_data_loader() with BasicDataset-compatible mocks."""

    def test_set_data_loader_creates_all_loaders(self, cfg):
        """set_data_loader populates loader_dict with all three keys."""
        model = _build_fixmatch()
        lb_dset = LabeledDataset(n_samples=8, image_size=IMG_SIZE)
        ulb_dset = LabeledDataset(n_samples=16, image_size=IMG_SIZE)
        eval_dset = LabeledDataset(n_samples=8, image_size=IMG_SIZE)

        model.set_data_loader(cfg, lb_dset, ulb_dset, eval_dset)

        assert "train_lb" in model.loader_dict
        assert "train_ulb" in model.loader_dict
        assert "eval" in model.loader_dict

    def test_set_data_loader_without_eval(self, cfg):
        """When eval_dset is None, no eval loader is created."""
        model = _build_fixmatch()
        lb_dset = LabeledDataset(n_samples=8, image_size=IMG_SIZE)
        ulb_dset = LabeledDataset(n_samples=16, image_size=IMG_SIZE)

        model.set_data_loader(cfg, lb_dset, ulb_dset, eval_dset=None)

        assert "train_lb" in model.loader_dict
        assert "train_ulb" in model.loader_dict
        assert "eval" not in model.loader_dict
