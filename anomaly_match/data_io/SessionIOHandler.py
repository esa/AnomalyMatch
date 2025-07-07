#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import json
import pickle
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from loguru import logger
import torch

from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.pipeline.SessionTracker import IterationInfo
from anomaly_match.data_io.save_config import save_config_toml


class SessionIOHandler:
    """
    Handles saving and loading of session data from SessionTracker.

    This class manages the persistence of session information, including:
    - Session metadata and iteration information
    - Labeled data CSV files
    - Model checkpoints
    - Configuration files
    """

    def __init__(self, base_save_path: str = "anomaly_match_results/sessions"):
        """
        Initialize the SessionIOHandler.

        Args:
            base_save_path: Base directory for saving session data.
        """
        self.base_save_path = Path(base_save_path)
        self.base_save_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized SessionIOHandler with base path: {self.base_save_path}")

    def get_session_save_path(self, session_tracker: SessionTracker) -> Path:
        """
        Get the save path for a session.

        Args:
            session_tracker: SessionTracker instance.

        Returns:
            Path where the session should be saved.
        """
        session_folder = f"{session_tracker.session_name}_{session_tracker.session_start_time.strftime('%Y%m%d_%H%M%S')}"
        return self.base_save_path / session_folder

    def save_session(
        self, session_tracker: SessionTracker, save_path: Optional[Path] = None, cfg=None
    ) -> Path:
        """
        Save complete session data to disk.

        Args:
            session_tracker: SessionTracker instance to save.
            save_path: Optional custom save path. If None, uses default naming.
            cfg: Optional configuration to save alongside the session.

        Returns:
            Path where the session was saved.
        """
        if save_path is None:
            save_path = self.get_session_save_path(session_tracker)

        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving session to: {save_path}")

        # Save session metadata
        self._save_session_metadata(session_tracker, save_path)

        # Save labeled data CSV
        self._save_labeled_data(session_tracker, save_path)

        # Save configuration if available
        self._save_config(session_tracker, save_path, cfg)

        logger.debug(f"Session saved successfully to: {save_path}")
        return save_path

    def _save_session_metadata(self, session_tracker: SessionTracker, save_path: Path) -> None:
        """Save session metadata as JSON."""
        metadata = {
            "session_info": session_tracker.get_session_info(),
            "all_iterations": session_tracker.get_all_iterations_info(),
        }

        metadata_path = save_path / "session_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.debug(f"Saved session metadata to: {metadata_path}")

    def _save_labeled_data(self, session_tracker: SessionTracker, save_path: Path) -> None:
        """Save labeled data as CSV."""
        labeled_data_path = save_path / "labeled_data.csv"

        # Ensure the iteration column exists and is properly saved
        labeled_df = session_tracker.get_labeled_data_df()
        if "iteration" not in labeled_df.columns:
            labeled_df["iteration"] = -1  # Default for legacy data

        labeled_df.to_csv(labeled_data_path, index=False)
        logger.debug(f"Saved labeled data to: {labeled_data_path}")

    def _save_config(self, session_tracker: SessionTracker, save_path: Path, cfg=None) -> None:
        """Save configuration if available."""
        try:
            # Try to use the provided config first, then fall back to session_tracker.final_cfg
            config_to_save = cfg

            if config_to_save is not None:
                config_path = save_path / "config.toml"
                save_config_toml(config_to_save, config_path)
                logger.debug(f"Saved configuration to: {config_path}")
            else:
                logger.debug("No configuration available to save")
        except Exception as e:
            logger.warning(f"Failed to save configuration: {e}")

    def save_model_checkpoint(
        self,
        model_state: Dict[str, Any],
        session_tracker: SessionTracker,
        checkpoint_name: str = None,
    ) -> str:
        """
        Save a model checkpoint within the session directory.

        Args:
            model_state: Model state dictionary to save.
            session_tracker: Associated session tracker.
            checkpoint_name: Optional custom checkpoint name.

        Returns:
            Path to saved checkpoint.
        """
        save_path = self.get_session_save_path(session_tracker)
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoints_dir = save_path / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        if checkpoint_name is None:
            checkpoint_name = f"model_iter_{session_tracker.total_model_iterations}.pkl"

        checkpoint_path = checkpoints_dir / checkpoint_name

        with open(checkpoint_path, "wb") as f:
            pickle.dump(model_state, f)

        # Update the session tracker with the checkpoint path
        session_tracker.update_model_state_path(str(checkpoint_path))

        logger.debug(f"Saved model checkpoint to: {checkpoint_path}")
        return str(checkpoint_path)

    def save_model(self, model, cfg, session_tracker: SessionTracker = None) -> str:
        """
        Save the model to the session directory if session_tracker is available,
        otherwise use the path specified in the config.

        Args:
            model: FixMatch model instance to save
            cfg: Configuration containing model_path or for extracting normalisation
            session_tracker: Optional session tracker for saving in session directory

        Returns:
            Path where the model was saved
        """
        if session_tracker is not None:
            # Always save as part of session when session_tracker is available
            save_path = self.get_session_save_path(session_tracker)
            save_path.mkdir(parents=True, exist_ok=True)

            # Include iteration number in model filename
            iteration_num = (
                len(session_tracker.session_iterations) - 1
                if session_tracker.session_iterations
                else 0
            )
            model_filename = f"model_iteration_{iteration_num}.pth"
            model_path = save_path / model_filename
        else:
            if cfg.model_path is None:
                logger.error("No model path specified in config or session tracker")
                raise ValueError("Model path must be specified in config or session tracker")
            else:
                model_path = Path(cfg.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.trace(f"Saving model to {model_path}")

        # Handle distributed training case
        train_model = (
            model.train_model.module if hasattr(model.train_model, "module") else model.train_model
        )
        eval_model = (
            model.eval_model.module if hasattr(model.eval_model, "module") else model.eval_model
        )

        # Create save state
        save_state = {
            "train_model": train_model.state_dict(),
            "eval_model": eval_model.state_dict(),
            "optimizer": model.optimizer.state_dict() if model.optimizer else None,
            "scheduler": model.scheduler.state_dict() if model.scheduler else None,
            "it": model.it,
            "total_it": model.total_it,
            "best_eval_acc": model.best_eval_acc,
            "best_it": model.best_it,
            "last_normalisation_method": getattr(model, "last_normalisation_method", None),
            "normalisation_method": cfg.normalisation_method,
        }

        # Save model
        torch.save(save_state, model_path)

        if session_tracker is not None:
            # Ensure there's an active session iteration
            if not session_tracker.session_iterations:
                session_tracker.start_new_session_iteration()
            session_tracker.update_model_state_path(str(model_path))
            # Update config to point to the actual saved model path
            cfg.model_path = str(model_path)

        logger.debug(f"Model saved successfully to: {model_path}")
        return str(model_path)

    def load_model(self, model, cfg, model_path: str = None) -> bool:
        """
        Load a saved model from the specified path or config model_path.

        Args:
            model: FixMatch model instance to load into
            cfg: Configuration object that will be updated with normalisation method
            model_path: Optional specific path to load from. If None, uses cfg.model_path

        Returns:
            bool: True if loading was successful, False otherwise
        """
        # Determine model path
        if model_path is None:
            if not hasattr(cfg, "model_path") or cfg.model_path is None:
                logger.error("No model path specified in config or as parameter")
                return False
            load_path = cfg.model_path
        else:
            load_path = model_path

        logger.info(f"Loading model from {load_path}")

        # Verify path exists
        if not os.path.exists(load_path):
            logger.error(f"Model path {load_path} does not exist")
            return False

        try:
            # Load checkpoint
            checkpoint = torch.load(load_path, weights_only=False)

            # Handle distributed training case
            train_model = (
                model.train_model.module
                if hasattr(model.train_model, "module")
                else model.train_model
            )
            eval_model = (
                model.eval_model.module if hasattr(model.eval_model, "module") else model.eval_model
            )

            # Load model states
            if "train_model" in checkpoint:
                train_model.load_state_dict(checkpoint["train_model"])
                logger.debug("Loaded train model state")

            if "eval_model" in checkpoint:
                eval_model.load_state_dict(checkpoint["eval_model"])
                logger.debug("Loaded eval model state")

            # Load optimizer state if available
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                if model.optimizer is not None:
                    model.optimizer.load_state_dict(checkpoint["optimizer"])
                    logger.debug("Loaded optimizer state")

            # Load scheduler state if available
            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                if model.scheduler is not None:
                    model.scheduler.load_state_dict(checkpoint["scheduler"])
                    logger.debug("Loaded scheduler state")

            # Load iteration counters
            if "it" in checkpoint:
                model.it = checkpoint["it"]
            if "total_it" in checkpoint:
                model.total_it = checkpoint["total_it"]
            if "best_eval_acc" in checkpoint:
                model.best_eval_acc = checkpoint["best_eval_acc"]
            if "best_it" in checkpoint:
                model.best_it = checkpoint["best_it"]

            # Handle normalisation method updates
            normalisation_updated = False
            if "last_normalisation_method" in checkpoint:
                cfg.normalisation_method = checkpoint["normalisation_method"]
                model.last_normalisation_method = checkpoint["normalisation_method"]
                normalisation_updated = True
                logger.debug(f"Updated normalisation method to: {cfg.normalisation_method}")
            elif "normalisation_method" in checkpoint:
                cfg.normalisation_method = checkpoint["last_normalisation_method"]
                model.last_normalisation_method = checkpoint["last_normalisation_method"]
                normalisation_updated = True
                logger.debug(
                    f"Updated normalisation method from legacy field: {cfg.normalisation_method}"
                )

            if normalisation_updated:
                logger.info(f"Model loaded with normalisation method: {cfg.normalisation_method}")

            # Update config to point to the successfully loaded model path
            cfg.model_path = load_path
            logger.debug(f"Updated config model_path to: {load_path}")

            logger.info(f"Model loaded successfully from: {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            return False

    def load_model_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a model checkpoint from the specified path.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary containing the checkpoint data, or None if loading failed
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
                return None

            # Try loading as pickle first (new format), then as torch (legacy)
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)
                logger.debug(f"Loaded checkpoint from pickle format: {checkpoint_path}")
            except (pickle.UnpicklingError, EOFError):
                # Fall back to torch format
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
                logger.debug(f"Loaded checkpoint from torch format: {checkpoint_path}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None

    def load_session(self, session_path: Path) -> SessionTracker:
        """
        Load a session from disk.

        Args:
            session_path: Path to the session directory.

        Returns:
            Loaded SessionTracker instance.
        """
        session_path = Path(session_path)
        if not session_path.exists():
            raise FileNotFoundError(f"Session path does not exist: {session_path}")

        logger.info(f"Loading session from: {session_path}")

        # Load session metadata
        metadata_path = session_path / "session_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Session metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create new SessionTracker and populate it
        session_info = metadata["session_info"]
        session_tracker = SessionTracker(session_info["session_name"])

        # Restore session data
        session_tracker.total_model_iterations = session_info["total_model_iterations"]

        # Load labeled data if available
        labeled_data_path = session_path / "labeled_data.csv"
        if labeled_data_path.exists():
            session_tracker.labeled_data_df = pd.read_csv(labeled_data_path)

        for iter_data in metadata.get("all_iterations", []):
            iteration_info = IterationInfo(
                iteration_number=iter_data["iteration_number"],
                timestamp=iter_data["timestamp"],
                model_loss=iter_data.get("model_loss"),
                test_performance=iter_data.get("test_performance"),
                model_state_path=iter_data.get("model_state_path"),
                num_newly_labeled_anomalous=iter_data.get("num_newly_labeled_anomalous", 0),
                num_newly_labeled_nominal=iter_data.get("num_newly_labeled_nominal", 0),
            )
            session_tracker.session_iterations.append(iteration_info)

        session_tracker.current_session_iteration = len(session_tracker.session_iterations)

        logger.info(f"Session loaded successfully from: {session_path}")
        return session_tracker

    def list_sessions(self) -> List[Path]:
        """
        List all available sessions in the base save path.

        Returns:
            List of paths to session directories.
        """
        if not self.base_save_path.exists():
            return []

        sessions = []
        for item in self.base_save_path.iterdir():
            if item.is_dir() and (item / "session_metadata.json").exists():
                sessions.append(item)

        return sorted(sessions)

    def get_session_summary(self, session_path: Path) -> Dict[str, Any]:
        """
        Get a summary of a session without fully loading it.

        Args:
            session_path: Path to the session directory.

        Returns:
            Dictionary with session summary information.
        """
        metadata_path = session_path / "session_metadata.json"
        if not metadata_path.exists():
            return {"error": "Session metadata not found"}

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata["session_info"]
        except Exception as e:
            return {"error": f"Failed to load session summary: {str(e)}"}

    def save_run(
        self,
        model,
        save_name: str,
        save_path: str,
        cfg=None,
        session_tracker: SessionTracker = None,
    ) -> str:
        """
        Save a training run's model weights and configuration.

        This replaces the FixMatch.save_run method and integrates with SessionTracker.

        Args:
            model: FixMatch model instance to save
            save_name: Filename for the saved model
            save_path: Directory path for saving
            cfg: Optional configuration to save alongside the model
            session_tracker: Optional session tracker to update with model path

        Returns:
            str: Path where the model was saved
        """
        save_filename = os.path.join(save_path, save_name)

        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Handle distributed training case
        train_model = (
            model.train_model.module
            if hasattr(model.train_model, "module") and model.train_model.module is not None
            else model.train_model
        )
        eval_model = (
            model.eval_model.module
            if hasattr(model.eval_model, "module") and model.eval_model.module is not None
            else model.eval_model
        )

        # Save model state
        save_state = {
            "train_model": train_model.state_dict(),
            "eval_model": eval_model.state_dict(),
            "optimizer": model.optimizer.state_dict() if model.optimizer else None,
            "scheduler": model.scheduler.state_dict() if model.scheduler else None,
            "it": model.it,
            "total_it": getattr(model, "total_it", model.it),
            "best_eval_acc": getattr(model, "best_eval_acc", None),
            "best_it": getattr(model, "best_it", None),
            "last_normalisation_method": getattr(model, "last_normalisation_method", None),
        }

        torch.save(save_state, save_filename)

        # Update session tracker if provided
        if session_tracker is not None:
            if not session_tracker.session_iterations:
                session_tracker.start_new_session_iteration()
            session_tracker.update_model_state_path(save_filename)

        # Save configuration if provided and NOT using session_tracker (to avoid duplicates)
        if cfg is not None and session_tracker is None:
            # Use the new TOML config saving
            try:
                from anomaly_match.data_io.save_config import save_config_toml

                config_path = os.path.join(
                    save_path, f"{os.path.splitext(save_name)[0]}_config.toml"
                )
                save_config_toml(cfg, config_path)
                logger.info(f"Configuration saved to: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to save config as TOML: {e}")
                # Fallback to legacy save_cfg if available
                try:
                    from anomaly_match.utils.validate_config import save_cfg

                    save_cfg(cfg)
                except Exception as fallback_e:
                    logger.error(f"Failed to save config with fallback method: {fallback_e}")

        logger.info(f"Training run saved to: {save_filename}")
        return save_filename

    def save_labels_to_output_dir(
        self, labeled_data_df: pd.DataFrame, output_dir: str, session_tracker: SessionTracker = None
    ) -> str:
        """
        Save labeled data to the session directory if session_tracker is available,
        otherwise use output_dir for backward compatibility.

        This replaces the session.save_labels functionality.

        Args:
            labeled_data_df: DataFrame containing labeled data
            output_dir: Output directory to save to (used only if no session_tracker)
            session_tracker: Session tracker to save to session directory

        Returns:
            str: Path where labels were saved
        """
        if session_tracker is not None:
            # Save to session directory - centralized approach
            session_path = self.get_session_save_path(session_tracker)
            session_path.mkdir(parents=True, exist_ok=True)
            filepath = session_path / "labeled_data.csv"

            # Update session tracker - it will handle adding iteration column internally
            session_tracker.update_labeled_data(labeled_data_df)

            # Save only the original columns to CSV (exclude iteration column for backward compatibility)
            csv_df = labeled_data_df.copy()
            csv_df.to_csv(filepath, index=False)

            logger.info(f"Labels saved to session directory: {filepath}")
        else:
            # Backward compatibility: save to specified output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            filepath = Path(output_dir) / "labeled_data.csv"

            # Save original data without iteration column for backward compatibility
            labeled_data_df.to_csv(filepath, index=False)

            logger.info(f"Labels saved to output directory: {filepath}")

        return str(filepath)

    def update_config_paths_for_session(self, cfg, session_tracker: SessionTracker) -> None:
        """
        Update configuration paths to use session directories.

        This ensures all saves go to the centralized session folder.

        Args:
            cfg: Configuration object to update
            session_tracker: Active session tracker
        """
        session_path = self.get_session_save_path(session_tracker)

        # Update model path to session directory only if not already set by user
        if cfg.model_path is None:
            cfg.model_path = str(session_path / "model.pth")

        # Update output directory to session directory
        cfg.output_dir = str(session_path)

        # Update save directory to session directory
        cfg.save_dir = str(session_path)

        logger.debug(f"Updated config paths to use session directory: {session_path}")


def print_session(filepath: str) -> None:
    """
    Print session information in a formatted way.

    Args:
        filepath: Path to the session directory.
    """
    try:
        io_handler = SessionIOHandler()
        session_path = Path(filepath)

        if not session_path.exists():
            print(f"Error: Session path does not exist: {filepath}")
            return

        # Get session summary first
        summary = io_handler.get_session_summary(session_path)
        if "error" in summary:
            print(f"Error loading session: {summary['error']}")
            return

        # Print session information
        print("=" * 60)
        print("ANOMALY MATCH SESSION REPORT")
        print("=" * 60)
        print(f"Session Name: {summary['session_name']}")
        print(f"Start Time: {summary['session_start_time']}")
        print(f"Duration: {summary['session_duration_minutes']:.1f} minutes")
        print()

        print("TRAINING SUMMARY:")
        print("-" * 30)
        print(f"Total Session Iterations: {summary['total_session_iterations']}")
        print(f"Total Model Iterations: {summary['total_model_iterations']}")
        print(f"Final Model Loss: {summary.get('final_model_loss', 'N/A')}")
        print(f"Average Model Loss: {summary.get('average_model_loss', 'N/A')}")
        print()

        print("LABELING SUMMARY:")
        print("-" * 30)
        print(f"Total Labeled Samples: {summary['total_labeled_samples']}")
        print(f"Anomalous Samples: {summary['total_anomalous_samples']}")
        print(f"Nominal Samples: {summary['total_nominal_samples']}")
        print()

        # Load full session for detailed iteration info
        try:
            session_tracker = io_handler.load_session(session_path)
            iterations_info = session_tracker.get_all_iterations_info()

            if iterations_info:
                print("ITERATION DETAILS:")
                print("-" * 30)
                for iter_info in iterations_info:
                    print(f"Iteration {iter_info['iteration_number']}:")
                    print(f"  Timestamp: {iter_info['timestamp']}")
                    print(f"  Anomalous samples: {iter_info['num_anomalous_samples']}")
                    print(f"  Nominal samples: {iter_info['num_nominal_samples']}")
                    if iter_info["model_loss"]:
                        print(f"  Model loss: {iter_info['model_loss']:.4f}")
                    if iter_info["test_performance"]:
                        print(f"  Test performance: {iter_info['test_performance']}")
                    print()
        except Exception as e:
            print(f"Warning: Could not load detailed iteration info: {str(e)}")

        # Check for additional files
        print("FILES:")
        print("-" * 30)
        labeled_data_path = session_path / "labeled_data.csv"
        if labeled_data_path.exists():
            print(f"✓ labeled_data.csv ({labeled_data_path.stat().st_size} bytes)")

        # Check for config files (prefer TOML, but also show legacy pickle)
        config_toml_path = session_path / "config.toml"
        if config_toml_path.exists():
            print("✓ config.toml")

        checkpoints_dir = session_path / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.pkl"))
            print(f"✓ {len(checkpoints)} model checkpoint(s)")

        print("=" * 60)

    except Exception as e:
        print(f"Error loading session: {str(e)}")
        logger.error(f"Error in print_session: {str(e)}")
