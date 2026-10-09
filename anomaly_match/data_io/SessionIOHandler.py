#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Session I/O handler for saving and loading models, labels, and predictions."""

from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from anomaly_match.data_io.checkpoint_io import load_checkpoint, save_checkpoint
from anomaly_match.data_io.save_config import save_config_toml
from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL, LABEL_REMOVED
from anomaly_match.pipeline.SessionTracker import IterationInfo, SessionTracker

if TYPE_CHECKING:
    from dotmap import DotMap

    from anomaly_match.models.FixMatch import FixMatch


class SessionIOHandler:
    """Handles saving and loading of session data from SessionTracker.

    This class manages the persistence of session information, including:
    - Session metadata and iteration information
    - Labeled data CSV files
    - Model checkpoints
    - Configuration files

    Args:
        base_save_path: Base directory for saving session data.
    """

    def __init__(self, base_save_path: str = "anomaly_match_results/sessions"):
        self.base_save_path = Path(base_save_path)
        self.base_save_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized SessionIOHandler with base path: {self.base_save_path}")

    def get_session_save_path(self, session_tracker: SessionTracker) -> Path:
        """Get the save path for a session.

        Args:
            session_tracker: SessionTracker instance.

        Returns:
            Path where the session should be saved.
        """
        session_folder = f"{session_tracker.session_name}_{session_tracker.session_start_time.strftime('%Y%m%d_%H%M%S')}"
        return self.base_save_path / session_folder

    def save_session(
        self,
        session_tracker: SessionTracker,
        save_path: Path | None = None,
        cfg: DotMap | None = None,
    ) -> Path:
        """Save complete session data to disk.

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

    def _save_config(
        self, session_tracker: SessionTracker, save_path: Path, cfg: DotMap | None = None
    ) -> None:
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

    def save_model(
        self, model: FixMatch, cfg: DotMap, session_tracker: SessionTracker | None = None
    ) -> str:
        """Save the model to the session directory or config-specified path.

        If session_tracker is available, saves to the session directory,
        otherwise uses the path specified in the config.

        Args:
            model: FixMatch model instance to save
            cfg: Configuration containing model_path or for extracting normalisation
            session_tracker: Optional session tracker for saving in session directory

        Returns:
            Path where the model was saved

        Raises:
            ValueError: If no session_tracker is provided and model_path is not
                specified in config.
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
            model_filename = f"model_iteration_{iteration_num}.safetensors"
            model_path = save_path / model_filename
        else:
            if cfg.model_path is None:
                logger.error("No model path specified in config or session tracker")
                raise ValueError("Model path must be specified in config or session tracker")
            else:
                model_path = Path(cfg.model_path).with_suffix(".safetensors")
            model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.trace(f"Saving model to {model_path}")

        # Handle distributed training case
        train_model = (
            model.train_model.module if hasattr(model.train_model, "module") else model.train_model
        )
        eval_model = (
            model.eval_model.module if hasattr(model.eval_model, "module") else model.eval_model
        )

        # Get fitsbolt config if present (DotMap pickles directly)
        fitsbolt_cfg = getattr(cfg, "fitsbolt_cfg", None)
        if fitsbolt_cfg is not None:
            logger.debug("Including fitsbolt config in model checkpoint")

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
            "normalisation_method": cfg.normalisation.normalisation_method,
            "num_channels": cfg.num_channels,
            "net": cfg.net,
            "fitsbolt_cfg": fitsbolt_cfg,
            # Persist the band-mixing matrix as the authoritative standalone field.
            # fitsbolt only embeds a copy in fitsbolt_cfg when fits_extension is a
            # band list (not for catalogue/Cutana sources where it's None), and
            # prediction applies it on the GPU — so without this field a
            # >input-channel model (e.g. 4-band -> 3 ch) can't score.
            "channel_combination": cfg.normalisation.channel_combination,
        }

        # Embed labeled data as CSV string so the checkpoint is self-contained
        if session_tracker is not None:
            labeled_df = session_tracker.get_labeled_data_df()
            if labeled_df.empty:
                logger.warning("Saving model checkpoint without labeled data")
            else:
                save_state["labeled_data_csv"] = labeled_df.to_csv(index=False)

        # Save model
        save_checkpoint(save_state, model_path)

        if session_tracker is not None:
            # Ensure there's an active session iteration
            if not session_tracker.session_iterations:
                session_tracker.start_new_session_iteration()
            session_tracker.update_model_state_path(str(model_path))
            # Update config to point to the actual saved model path
            cfg.model_path = str(model_path)

        logger.debug(f"Model saved successfully to: {model_path}")
        return str(model_path)

    def load_model(self, model: FixMatch, cfg: DotMap, model_path: str | None = None) -> bool:
        """Load a saved model from the specified path or config model_path.

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
            checkpoint = load_checkpoint(load_path)

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
            # Checkpoints store two fields (both should have the same value in normal operation):
            # - "last_normalisation_method": from model.last_normalisation_method (what model was trained with)
            # - "normalisation_method": from cfg.normalisation.normalisation_method (config at save time)
            # We prefer "last_normalisation_method" (actual training method) over config value
            # Note: last_normalisation_method can be None if model was saved before first training
            normalisation_updated = False
            if checkpoint.get("last_normalisation_method") is not None:
                cfg.normalisation.normalisation_method = checkpoint["last_normalisation_method"]
                model.last_normalisation_method = checkpoint["last_normalisation_method"]
                normalisation_updated = True
                logger.debug(
                    f"Updated normalisation method to: {cfg.normalisation.normalisation_method.name}"
                )
            elif checkpoint.get("normalisation_method") is not None:
                cfg.normalisation.normalisation_method = checkpoint["normalisation_method"]
                model.last_normalisation_method = checkpoint["normalisation_method"]
                normalisation_updated = True
                logger.debug(
                    f"Updated normalisation method from config field: {cfg.normalisation.normalisation_method.name}"
                )

            if normalisation_updated:
                logger.info(
                    f"Model loaded with normalisation method: {cfg.normalisation.normalisation_method.name}"
                )

            # Warn if model was trained with a different number of channels
            saved_channels = checkpoint.get("num_channels")
            if saved_channels is not None and saved_channels != cfg.num_channels:
                logger.warning(
                    f"Channel mismatch: model was trained with {saved_channels} channels "
                    f"but current dataset has {cfg.num_channels} channels. "
                    f"This will likely cause errors."
                )

            # Load fitsbolt config if present in checkpoint (DotMap pickles directly)
            if "fitsbolt_cfg" in checkpoint and checkpoint["fitsbolt_cfg"] is not None:
                cfg.fitsbolt_cfg = checkpoint["fitsbolt_cfg"]
                logger.debug("Loaded fitsbolt config from model checkpoint")

            # Update config to point to the successfully loaded model path
            cfg.model_path = load_path
            logger.debug(f"Updated config model_path to: {load_path}")

            logger.info(f"Model loaded successfully from: {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            return False

    @staticmethod
    def get_labeled_data_from_checkpoint(checkpoint: dict[str, Any]) -> pd.DataFrame:
        """Extract labeled data DataFrame from a model checkpoint.

        Args:
            checkpoint: Loaded checkpoint dict (from load_checkpoint)

        Returns:
            DataFrame with columns [filename, label, iteration].

        Raises:
            ValueError: If the checkpoint does not contain labeled data.
        """
        csv_str = checkpoint.get("labeled_data_csv")
        if csv_str is None:
            raise ValueError(
                "Checkpoint does not contain labeled data. "
                "Only checkpoints saved with a session tracker include labeled data."
            )
        return pd.read_csv(StringIO(csv_str))

    def load_session(self, session_path: Path) -> SessionTracker:
        """Load a session from disk.

        Args:
            session_path: Path to the session directory.

        Returns:
            Loaded SessionTracker instance.

        Raises:
            FileNotFoundError: If the session path or session metadata file
                does not exist.
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
                unlabelled_scores_file=iter_data.get("unlabelled_scores_file"),
                test_scores_file=iter_data.get("test_scores_file"),
            )
            session_tracker.session_iterations.append(iteration_info)

        session_tracker.current_session_iteration = len(session_tracker.session_iterations)

        logger.info(f"Session loaded successfully from: {session_path}")
        return session_tracker

    def list_sessions(self) -> list[Path]:
        """List all available sessions in the base save path.

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

    def get_session_summary(self, session_path: Path) -> dict[str, Any]:
        """Get a summary of a session without fully loading it.

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

    def save_labels_to_output_dir(
        self,
        labeled_data_df: pd.DataFrame,
        output_dir: str,
        session_tracker: SessionTracker | None = None,
    ) -> str:
        """Save labeled data to the session directory or output_dir.

        If session_tracker is available, saves to the session directory,
        otherwise uses output_dir for backward compatibility.

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

    def merge_gallery_labels(
        self,
        existing_label_file: str | None,
        new_labels: dict[str, str],
        output_path: str,
    ) -> str:
        """Merge new gallery labels with an existing label CSV and write the result.

        The CSV uses ``id, label`` columns.  ``id`` is the data-source
        identifier (filename for image folders, source-id for Cutana, index
        for Zarr) and is read and written as text throughout, so numeric
        catalogue ids survive the round trip exactly as the catalogue wrote
        them.

        Args:
            existing_label_file: Path to the current labels CSV (may be ``None``).
            new_labels: Dict mapping ``id → csv_label_string``.  Values are
                ``LABEL_ANOMALY``, ``LABEL_NORMAL`` or ``LABEL_REMOVED``;
                a ``removed`` value overrides any existing label for that id
                so an un-labelled gallery cell drops out of the labeled set.
            output_path: Where to write the merged CSV.

        Returns:
            Absolute path to the written CSV file.

        Raises:
            ValueError: If the existing CSV cannot be read, or carries no
                ``id`` column.  Either way the merge key is unavailable, so
                proceeding would write out only the new gallery labels and
                silently discard everything the user had labelled before.
        """
        existing_df = pd.DataFrame(columns=["id", "label"])
        if existing_label_file and os.path.isfile(existing_label_file):
            try:
                # ``dtype={"id": str}`` rather than a post-hoc ``astype(str)``:
                # the gallery hands ids back as str, and pandas treats 51 and
                # "51" as distinct keys, so an unnormalised concat makes
                # ``drop_duplicates`` below append a second row for a relabelled
                # source instead of overwriting the first (#556).  Normalising at
                # the read is what makes that exact, since type inference is
                # lossy: a single blank id infers float64 and turns 51 into
                # "51.0" (which never matches), and a zero-padded "007" into "7".
                existing_df = pd.read_csv(existing_label_file, dtype={"id": str})
            except Exception as exc:
                # Not recoverable: continuing would write a CSV holding only the
                # handful of new gallery labels, silently discarding every label
                # the user has already made and retraining on the remainder.
                raise ValueError(
                    f"Could not read existing labels CSV {existing_label_file}: {exc}"
                ) from exc

        if "id" not in existing_df.columns:
            raise ValueError(
                f"Labels CSV {existing_label_file} has no 'id' column — "
                f"found {list(existing_df.columns)}"
            )

        existing_ids = set(existing_df["id"])
        valid_new: dict[str, str] = {}
        for raw_id, label in new_labels.items():
            source_id = str(raw_id)
            if label in (LABEL_ANOMALY, LABEL_NORMAL):
                valid_new[source_id] = label
            elif label == LABEL_REMOVED and source_id in existing_ids:
                # A 'removed' override only matters for an id that already
                # carried a label — marking a never-labelled id removed would
                # just add a meaningless row that the dataset build skips anyway.
                valid_new[source_id] = label

        if valid_new:
            new_df = pd.DataFrame(
                [{"id": source_id, "label": label} for source_id, label in valid_new.items()]
            )
            merged = pd.concat([existing_df, new_df]).drop_duplicates(subset="id", keep="last")
        else:
            merged = existing_df

        # Strict id,label output — drop any extra columns from existing CSVs
        merged = merged[["id", "label"]]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        merged.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def filter_labels_csv_to_ids(csv_path: str, valid_ids: set[str]) -> tuple[int, int]:
        """Rewrite *csv_path* keeping only rows whose ``id`` is in *valid_ids*.

        Used just before launching the training subprocess to drop
        labels for IDs the data source didn't surface (catalogue rows
        that Cutana silently rejected, files the user moved since
        labelling, ...).  Without this the training subprocess logs a
        ``Filename X not found in dataset`` warning per orphaned label
        — one line per row, which on a 1000+ label CSV floods the UI
        output and masks real warnings.

        Args:
            csv_path: Path to the labels CSV to filter in place.
            valid_ids: IDs to keep.  Rows whose ``id`` is not in this
                set are dropped.

        Returns:
            Tuple ``(kept_count, dropped_count)``.
        """
        # Same read-time normalisation as ``merge_gallery_labels``, and for the
        # same reason: a post-hoc ``astype(str)`` inherits pandas' inference, so
        # one blank id row makes the column float64 and turns every id into
        # "51.0".  Nothing then matches *valid_ids*, and this function — which
        # runs immediately before training and rewrites the CSV in place —
        # would silently drop the user's entire label set.  Reading as str also
        # keeps zero-padded ids ("007") from being rewritten as "7".
        df = pd.read_csv(csv_path, dtype={"id": str})
        total = len(df)
        mask = df["id"].isin(valid_ids)
        kept = df[mask]
        kept.to_csv(csv_path, index=False)
        return len(kept), total - len(kept)

    def update_config_paths_for_session(self, cfg: DotMap, session_tracker: SessionTracker) -> None:
        """Update configuration paths to use session directories.

        This ensures all saves go to the centralized session folder.

        Args:
            cfg: Configuration object to update
            session_tracker: Active session tracker
        """
        session_path = self.get_session_save_path(session_tracker)

        # Create session directory immediately so logs and outputs have a home
        session_path.mkdir(parents=True, exist_ok=True)

        # Update model path to session directory only if not already set by user
        if cfg.model_path is None:
            cfg.model_path = str(session_path / "model.safetensors")

        # Update output directory to session directory
        cfg.output_dir = str(session_path)

        # Update save directory to session directory
        cfg.save_dir = str(session_path)

        # Add session-specific log file
        logger.add(
            str(session_path / "session.log"),
            rotation="10 MB",
            format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
            level="DEBUG",
        )

        logger.debug(f"Updated config paths to use session directory: {session_path}")


def print_session(filepath: str) -> None:
    """Print session information in a formatted way.

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
        print(f"Normal Samples: {summary['total_nominal_samples']}")
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
                    print(f"  Normal samples: {iter_info['num_nominal_samples']}")
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
            checkpoints = list(checkpoints_dir.glob("*.safetensors"))
            print(f"✓ {len(checkpoints)} model checkpoint(s)")

        print("=" * 60)

    except Exception as e:
        print(f"Error loading session: {str(e)}")
        logger.error(f"Error in print_session: {str(e)}")
