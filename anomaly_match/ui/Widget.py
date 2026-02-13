#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import os
import numpy as np
from ipywidgets import HBox
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from loguru import logger
from sklearn.metrics import roc_curve
import time
import datetime

from skimage.util import img_as_ubyte

from anomaly_match.data_io.load_images import load_and_process_single_wrapper
from anomaly_match.ui.preview_widget import PreviewWidget

# Import the newly created UI elements
from anomaly_match.ui.ui_elements import (
    create_ui_elements,
    attach_click_listeners,
    HTML_setup,
)


def shorten_filename(filename: str, max_length: int = 25) -> str:
    """
    Shorten a filename to fit within the specified maximum length.

    Preserves the extension and shows beginning/end of the basename.

    Args:
        filename: The filename to shorten.
        max_length: Maximum total length of the result.

    Returns:
        Shortened filename if needed, original otherwise.
    """
    if len(filename) <= max_length:
        return filename

    # Split into base and extension, handling multiple dots
    if "." in filename:
        # Find the last dot for extension
        last_dot_idx = filename.rfind(".")
        basename = filename[:last_dot_idx]
        extension = filename[last_dot_idx:]  # includes the dot
    else:
        basename = filename
        extension = ""

    # Calculate available space for basename
    # Format: "start...end" + extension
    ellipsis = "..."
    available = max_length - len(extension) - len(ellipsis)

    if available <= 6:
        # Very short max_length, just truncate
        return filename[: max_length - 3] + "..."

    # Split available space: more at start, less at end
    start_len = (available * 2) // 3
    end_len = available - start_len

    shortened = basename[:start_len] + ellipsis + basename[-end_len:] + extension
    return shortened


class Widget:
    """A widget-based user interface for interacting with the anomaly detection session."""

    def __init__(self, session):
        """
        Initializes the UI widget with the given session.

        Args:
            session (Session): The session to interact with.
        """
        # Store session
        self.session = session
        self.cfg = session.cfg

        # Create the preview widget (handles image display and transforms)
        self.preview = PreviewWidget(session)

        # Create all UI elements (moved from the big monolithic class)
        self.ui = create_ui_elements()

        # Replace the ui_elements image/text widgets with preview widget's versions
        self.ui["image_widget"] = self.preview.image_widget
        self.ui["filename_text"] = self.preview.filename_text

        # Rebuild center_row with preview widget's components
        from ipywidgets import VBox
        import ipywidgets as widgets

        self.ui["center_row"] = VBox(
            [self.preview.filename_text, self.preview.image_widget],
            layout=widgets.Layout(background_color="black"),
        )

        # Update main_layout to use the new center_row with preview widget's components.
        # The main_layout VBox has this structure:
        #   [0] model_controls, [1] top_row, [2] center_row, [3:] transform_controls + bottom rows
        # We replace center_row (index 2) with our new one containing the preview widget.
        main_layout = self.ui["main_layout"]
        model_controls = main_layout.children[0]
        top_row = main_layout.children[1]
        remaining_rows = main_layout.children[3:]  # transform_controls, bottom_row1-5

        main_layout.children = (model_controls, top_row, self.ui["center_row"], *remaining_rows)

        # Set the full resolution button reference
        self.preview.set_full_res_button(self.ui["transform_buttons"]["full_res"])

        # Attach the output widget so the session logs go there
        session.set_terminal_out(self.ui["out"])
        # Also attach it to the memory monitor
        self.ui["memory_monitor"].set_output_widget(self.ui["out"])

        # Sync initial batch slider value with session config
        self.ui["batch_size_slider"].value = self.cfg.N_to_load

        # Set initial normalization dropdown value from session cached value
        self.ui["normalisation_dropdown"].value = session.cached_image_normalisation_enum

        # Attach all click listeners / slider observers
        attach_click_listeners(self)
        self.ui["remember_button"].on_click(self.remember_current_file)

        # Attach RGB channel checkbox observers
        self.ui["red_channel_checkbox"].observe(self.toggle_red_channel, names="value")
        self.ui["green_channel_checkbox"].observe(self.toggle_green_channel, names="value")
        self.ui["blue_channel_checkbox"].observe(self.toggle_blue_channel, names="value")

        logger.debug("Initializing Widget UI")

        with self.ui["out"]:
            # Update UI components
            self.update()

            # Only attempt sorting if we have scores
            if self.session.scores is not None:
                self.sort_by_anomalous()

        # Start memory monitoring
        self.ui["memory_monitor"].start()

    def __del__(self):
        """Clean up resources when widget is destroyed."""
        if "memory_monitor" in self.ui:
            self.ui["memory_monitor"].stop()

    def start(self):
        """Starts the UI by displaying the components."""
        display(HTML_setup)
        display(self._build_main_layout())
        display(
            self.ui["out"],  # Might want it at the bottom, but you can rearrange
        )

    def _build_main_layout(self):
        """
        Builds the main VBox layout to be displayed in the start() method.
        """
        main_layout = self.ui["main_layout"]
        right_pane = self.ui["side_display"]

        # Compose everything
        return self._pack_layout(main_layout, right_pane)

    def _pack_layout(self, main_layout, side_display):
        """
        Helps compose final layout for display.
        """

        # Reduce font size on buttons using widget styling
        for _, widget in self.ui.items():
            if hasattr(widget, "style") and hasattr(widget, "description"):
                widget.style.font_size = "90%"  # Slightly smaller font for buttons

        # Adjust the image widget size
        self.ui["image_widget"].layout.width = "auto"
        self.ui["image_widget"].layout.height = "auto"

        return HBox([main_layout, side_display])

    def search_all_files(self):
        """Searches all files and displays the top N with their scores."""
        with self.ui["out"]:
            self.session.cfg.N_to_load = self.ui["batch_size_slider"].value
            logger.debug(
                f"Searching all files for anomalies with batch size: {self.session.cfg.N_to_load}"
            )

            # Set progress bar color to cyan for 'search_all_files' task
            self.ui["progress_bar"].style = {"bar_color": "cyan"}
            self.ui["train_label"].value = "Searching all files..."

            # Define progress callback to update UI based on information from session.py
            def update_progress(
                batch=None,
                num_batches=None,
                batch_update=False,
                eta_str=None,
                progress_percent=None,
                images_per_second=None,
                completed=False,
                total_time_str=None,
                final_speed=None,
            ):
                # Always update progress bar if batch info provided
                if batch is not None and num_batches is not None:
                    self.ui["progress_bar"].value = batch / num_batches

                    # Handle completion message
                    if completed and total_time_str:
                        self.ui[
                            "train_label"
                        ].value = f"Search complete in {total_time_str} ({final_speed:.1f} img/sec)"
                        return

                    # Handle batch update with ETA information
                    if batch_update:
                        if eta_str:
                            # Use the ETA info provided by session.py
                            message = f"Evaluating files... Batch: {batch}/{num_batches}"
                            if progress_percent is not None:
                                message += f" ({progress_percent:.1f}%)"
                            if images_per_second is not None:
                                message += f" | {images_per_second:.1f} img/sec"
                            message += f" | ETA: {eta_str}"
                            self.ui["train_label"].value = message
                        else:
                            # Early in the process when ETA isn't available yet
                            self.ui[
                                "train_label"
                            ].value = f"Evaluating files... Batch: {batch}/{num_batches}"
                    else:
                        # Regular evaluation updates (not used in this function but keeping for completeness)
                        if eta_str:
                            self.ui[
                                "train_label"
                            ].value = f"Evaluating... {batch}/{num_batches} | ETA: {eta_str}"
                        else:
                            self.ui["train_label"].value = f"Evaluating... {batch}/{num_batches}"

            self.session.evaluate_all_images(
                top_N=self.cfg.top_N, progress_callback=update_progress
            )
            # update models last_normalisation_method only after successful eval
            if self.session.model.last_normalisation_method is None:
                self.session.model.last_normalisation_method = (
                    self.session.cfg.normalisation.normalisation_method
                )
            elif (
                self.session.model.last_normalisation_method
                != self.session.cfg.normalisation.normalisation_method
            ):
                logger.warning(
                    f"Evaluated with a new normalisation {self.session.cfg.normalisation.normalisation_method.name} method "
                    + f"not previously used with the model: {self.session.model.last_normalisation_method.name}"
                )
                self.session.model.last_normalisation_method = (
                    self.session.cfg.normalisation.normalisation_method
                )

            # Display will be updated by the callback when completed
            self.display_top_files_scores()
            self.ui["progress_bar"].style = {"bar_color": "green"}

    def display_top_files_scores(self):
        """Displays the top files and their scores."""
        self.preview.set_index(0)
        self.preview.update_display()
        self.ui["progress_bar"].style = {"bar_color": "green"}
        self.display_gallery()

    def update_image_display(self):
        """Updates the display of the current image."""
        self.preview.update_display()

    def update_image_UI_label(self, filename=None, score=None):
        """Updates the UI label with the current image's filename, score, and label."""
        self.preview.update_label_only()

    # ======== Sorting Methods ========
    def sort_by_anomalous(self):
        """Sorts the images by their anomalous scores and updates the display."""
        self.session.sort_by_anomalous()
        self.preview.set_index(0)
        self.preview.update_display()

    def sort_by_nominal(self):
        """Sorts the images by their nominal scores and updates the display."""
        self.session.sort_by_nominal()
        self.preview.set_index(0)
        self.preview.update_display()

    def sort_by_mean(self):
        """Sorts the images by distance to mean score and updates the display."""
        self.session.sort_by_mean()
        self.preview.set_index(0)
        self.preview.update_display()

    def sort_by_median(self):
        """Sorts the images by distance to median score and updates the display."""
        self.session.sort_by_median()
        self.preview.set_index(0)
        self.preview.update_display()

    # ======== Navigation ========
    def next_image(self):
        """Displays the next image in the catalog."""
        new_index = min(len(self.session.img_catalog) - 1, self.preview.current_index + 1)
        self.preview.reset_full_resolution_mode()
        self.preview.set_index(new_index)
        self.preview.update_display()

    def previous_image(self):
        """Displays the previous image in the catalog."""
        new_index = max(0, self.preview.current_index - 1)
        self.preview.reset_full_resolution_mode()
        self.preview.set_index(new_index)
        self.preview.update_display()

    # ======== Image Transformations ========
    def restore_image(self):
        """Restores the current image to its original state."""
        self.ui["brightness_slider"].value = 1.0
        self.ui["contrast_slider"].value = 1.0
        self.ui["red_channel_checkbox"].value = True
        self.ui["green_channel_checkbox"].value = True
        self.ui["blue_channel_checkbox"].value = True
        self.preview.restore()

    def toggle_invert_image(self):
        """Toggles the inversion of the current image."""
        self.preview.toggle_invert()

    def toggle_unsharp_mask(self):
        """Toggles the application of an unsharp mask."""
        self.preview.toggle_unsharp_mask()

    def show_full_resolution(self):
        """Toggles full resolution mode and updates the display."""
        self.preview.toggle_full_resolution()

    def adjust_brightness_contrast(self, _):
        """Adjusts brightness and contrast of the current image."""
        self.preview.set_brightness(self.ui["brightness_slider"].value)
        self.preview.set_contrast(self.ui["contrast_slider"].value)

    def display_gallery(self):
        """Displays a small gallery of either mispredicted or top anomalous/nominal images."""
        with self.ui["gallery"]:
            self.ui["gallery"].clear_output(wait=True)
            try:
                if self.cfg.test_ratio > 0:
                    # Show mispredicted images
                    mispredicted_images = []
                    image_text = []

                    # First collect all filenames we want to display
                    display_files = []
                    for filename, (pred, label) in self.session.eval_performance[
                        "eval/predictions_and_labels"
                    ].items():
                        pred, label = pred.item(), label.item()
                        if pred != label:
                            display_files.append((filename, pred, label))

                    # Limit to top 10
                    display_files = display_files[:10]

                    # Load images one at a time using context manager
                    for filename, pred, label in display_files:
                        path = os.path.join(self.cfg.data_dir, filename)
                        if os.path.exists(path):
                            try:
                                img_array = load_and_process_single_wrapper(
                                    path,
                                    self.session.cfg,
                                    desc="widget loading image",
                                    show_progress=False,
                                )

                                img = Image.fromarray(img_array)
                                mispredicted_images.append(img_array)
                                display_name = shorten_filename(filename)

                                image_text.append(
                                    f"{display_name}\nPred: {pred:.2f} | Label: {label:.2f}"
                                )

                            except Exception as e:
                                print(f"Error loading {filename}: {e}")
                                continue

                    num_images = len(mispredicted_images)
                    if num_images > 0:
                        plt.figure(figsize=(12, 4), facecolor="black")
                        eval_perf = self.session.eval_performance
                        plt.suptitle(
                            f"Top {num_images} Mispredicted Test Images | "
                            f"Acc: {eval_perf['eval/top-1-acc'] * 100:.1f}% | "
                            f"AUROC: {eval_perf['eval/auroc']:.3f} | "
                            f"AUPRC: {eval_perf['eval/auprc']:.3f}",
                            fontsize=12,
                            color="white",
                        )

                        for i, img in enumerate(mispredicted_images):
                            ax = plt.subplot(2, 5, i + 1)
                            plt.imshow(img)
                            plt.title(image_text[i], fontsize=8, color="white")
                            plt.axis("off")
                            ax.set_facecolor("black")

                        plt.tight_layout(pad=1.0)
                        plt.show()
                        plt.close()

                        # Create separate figure for ROC and PRC curves
                        plt.figure(figsize=(10, 4), facecolor="black")

                        # Plot ROC curve
                        ax1 = plt.subplot(1, 2, 1)
                        labels, probs = eval_perf["eval/roc_data"]
                        fpr, tpr, _ = roc_curve(labels, probs)
                        ax1.plot(fpr, tpr, "b-", label=f"ROC (AUC={eval_perf['eval/auroc']:.3f})")
                        ax1.plot([0, 1], [0, 1], "r--")
                        ax1.set_title("ROC Curve", color="white")
                        ax1.grid(True, alpha=0.3)
                        ax1.set_xlabel("False Positive Rate", color="white")
                        ax1.set_ylabel("True Positive Rate", color="white")
                        ax1.tick_params(colors="white")
                        ax1.legend(loc="lower right", facecolor="black", labelcolor="white")
                        ax1.set_facecolor("black")

                        # Plot PRC curve
                        ax2 = plt.subplot(1, 2, 2)
                        precision, recall = eval_perf["eval/precision_recall"]
                        ax2.plot(
                            recall,
                            precision,
                            "g-",
                            label=f"PRC (AUC={eval_perf['eval/auprc']:.3f})",
                        )
                        ax2.set_title("Precision-Recall Curve", color="white")
                        ax2.grid(True, alpha=0.3)
                        ax2.set_xlabel("Recall", color="white")
                        ax2.set_ylabel("Precision", color="white")
                        ax2.tick_params(colors="white")
                        ax2.legend(loc="lower left", facecolor="black", labelcolor="white")
                        ax2.set_facecolor("black")

                        plt.tight_layout(pad=2.0)
                        plt.show()
                        plt.close()

                else:
                    # Show top 5 anomalous & top 5 nominal
                    scores = self.session.scores
                    indices = np.argsort(scores)
                    num_images_to_display = min(5, len(scores) // 2)

                    top_nominal_indices = indices[:num_images_to_display]
                    top_anomalous_indices = indices[-num_images_to_display:][::-1]

                    images = []
                    image_text = []

                    for idx in top_anomalous_indices:
                        img_arr = self.session.img_catalog[idx]
                        img_arr = img_arr - np.min(img_arr)
                        img_arr = img_arr / np.max(img_arr)
                        img_arr = img_as_ubyte(img_arr)
                        if img_arr.shape[-1] == 1:
                            img_arr = np.repeat(img_arr, 3, axis=-1)
                        pil_img = Image.fromarray(img_arr)
                        images.append(pil_img)
                        filename = self.session.filenames[idx]
                        display_name = shorten_filename(filename)
                        score = scores[idx]
                        image_text.append(f"{display_name}\nScore: {score:.4f}")

                    for idx in top_nominal_indices:
                        img_arr = self.session.img_catalog[idx]
                        img_arr = img_arr - np.min(img_arr)
                        img_arr = img_arr / np.max(img_arr)
                        img_arr = img_as_ubyte(img_arr)
                        if img_arr.shape[-1] == 1:
                            img_arr = np.repeat(img_arr, 3, axis=-1)
                        pil_img = Image.fromarray(img_arr)
                        images.append(pil_img)
                        filename = self.session.filenames[idx]
                        display_name = shorten_filename(filename)
                        score = scores[idx]
                        image_text.append(f"{display_name}\nScore: {score:.4f}")

                    num_images = len(images)
                    plt.figure(figsize=(12, 6), facecolor="black")
                    plt.suptitle(
                        f"Top {len(top_anomalous_indices)} Anomalous and "
                        f"Top {len(top_nominal_indices)} Nominal Images",
                        fontsize=12,
                        color="white",
                    )

                    for i, im in enumerate(images):
                        ax = plt.subplot(2, 5, i + 1)
                        plt.imshow(im)
                        plt.title(image_text[i], fontsize=8, color="white")
                        plt.axis("off")
                        ax.set_facecolor("black")

                    plt.tight_layout(pad=1.0)
                    plt.show()
            except Exception as e:
                logger.error(f"Error displaying gallery: {e}")

    def save_labels(self):
        self.session.save_labels()

    def remember_current_file(self, _):
        """Remembers the currently displayed file."""
        self.session.remember_current_file(self.session.filenames[self.preview.current_index])

    def save_model(self):
        """Saves the model using the session."""
        self.session.save_model()

    def load_model(self):
        """Loads the model using the session."""
        with self.ui["out"]:
            logger.debug(
                f"Loading model, cfg norm: {self.session.cfg.normalisation.normalisation_method}, "
            )
        self.session.load_model()

        # Update the normalization dropdown to match the session's method
        self.ui[
            "normalisation_dropdown"
        ].value = self.session.cfg.normalisation.normalisation_method

        with self.ui["out"]:
            logger.debug(
                f"Loaded model, cfg norm: {self.session.cfg.normalisation.normalisation_method},"
                + f" model norm: {self.session.model.last_normalisation_method}"
            )
        self.update()

    def train(self):
        """Starts the training process."""

        with self.ui["out"]:
            logger.debug(
                f"Session norm: {self.session.cached_image_normalisation_enum}, "
                f"selected norm: {self.session.cfg.normalisation.normalisation_method}"
            )

        with self.ui["out"]:
            logger.debug("Starting training...")
            self.ui["progress_bar"].style = {"bar_color": "blue"}
            self.ui["progress_bar"].value = 0.0

            self.cfg.num_train_iter = self.ui["train_iteration_slider"].value
            logger.debug(
                f"Training for {self.cfg.num_train_iter} iterations, "
                f"evaluating every {self.cfg.num_eval_iter}..."
            )

            self.ui["train_label"].value = "Training started..."

            # Track start time and previous iterations for ETA calculation
            start_time = time.time()
            last_update_time = start_time
            last_iteration = 0

            def update_training_progress(iteration, total_iterations):
                nonlocal last_update_time, last_iteration

                # Update progress bar
                self.ui["progress_bar"].value = iteration / total_iterations

                # Calculate time statistics and ETA
                current_time = time.time()
                elapsed_time = current_time - start_time

                # Only update ETA every 0.5 seconds or when iteration changes significantly
                if (current_time - last_update_time >= 0.5) or (
                    iteration - last_iteration >= max(1, total_iterations / 100)
                ):
                    if iteration > 0:
                        # Calculate time per iteration and ETA
                        time_per_iteration = elapsed_time / iteration
                        remaining_iterations = total_iterations - iteration
                        eta_seconds = time_per_iteration * remaining_iterations
                        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

                        # Update display with iteration count and ETA
                        self.ui["train_label"].value = (
                            f"Training... Iteration {iteration}/{total_iterations} | ETA: {eta_str}"
                        )
                    else:
                        # Early iterations - no reliable ETA yet
                        self.ui[
                            "train_label"
                        ].value = f"Training... Iteration {iteration}/{total_iterations}"

                    last_update_time = current_time
                    last_iteration = iteration

            self.session.train(self.cfg, progress_callback=update_training_progress)

            # update models last_normalisation_method after successful training
            if self.session.model.last_normalisation_method is None:
                self.session.model.last_normalisation_method = (
                    self.session.cfg.normalisation.normalisation_method
                )
            elif (
                self.session.model.last_normalisation_method
                != self.session.cfg.normalisation.normalisation_method
            ):
                logger.warning(
                    f"Trained with a new normalisation {self.session.cfg.normalisation.normalisation_method.name} method "
                    + f"not previously used with the model: {self.session.model.last_normalisation_method.name}"
                )
                self.session.model.last_normalisation_method = (
                    self.session.cfg.normalisation.normalisation_method
                )

            # Calculate total time taken
            total_time = time.time() - start_time
            time_str = str(datetime.timedelta(seconds=int(total_time)))

            self.ui["progress_bar"].style = {"bar_color": "green"}
            self.ui["train_label"].value = f"Training complete in {time_str}."
            self.update()
            self.sort_by_anomalous()

    def update(self):
        """Updates the UI components and performs evaluation."""
        self.ui["progress_bar"].style = {"bar_color": "cyan"}
        self.session.update_predictions()
        self.preview.set_index(0)

        if self.cfg.test_ratio > 0:
            if self.session.eval_performance is not None:
                self.ui[
                    "train_label"
                ].value = f"Training Complete. Eval Acc: {self.session.eval_performance['eval/top-1-acc'] * 100:.2f}%"
            else:
                self.ui["train_label"].value = "Training Complete. No evaluation performed yet."
        else:
            self.ui["train_label"].value = "Training Complete. No eval since test_ratio is 0."

        self.ui["progress_bar"].style = {"bar_color": "green"}
        with self.ui["out"]:
            logger.debug("Updating gallery...")
            self.display_gallery()
            logger.debug("Gallery updated.")

    def update_batch_size(self, change):
        """Updates the batch size in the session config."""
        self.session.cfg.N_to_load = change["new"]

    def next_batch(self):
        """Loads the next batch and updates predictions."""
        with self.ui["out"]:
            logger.debug("Loading next batch of data")
            self.ui["progress_bar"].style = {"bar_color": "orange"}
            self.ui["train_label"].value = "Predicting next batch..."

            self.session.load_next_batch()

            self.ui["progress_bar"].style = {"bar_color": "green"}
            self.ui["train_label"].value = "Batch loading complete."
            self.update()
            self.sort_by_anomalous()

    def reset_model(self):
        """Resets the model in the session."""
        with self.ui["out"]:
            self.session.reset_model()
            self.update()
            self.sort_by_anomalous()

    def load_top_files(self):
        """Loads the top files and updates the display."""
        with self.ui["out"]:
            self.session.load_top_files(self.cfg.top_N)
            self.display_top_files_scores()

    # Add channel toggle methods
    def toggle_red_channel(self, change):
        """Toggles the red channel on/off."""
        self.preview.set_rgb_channels(r=change["new"])

    def toggle_green_channel(self, change):
        """Toggles the green channel on/off."""
        self.preview.set_rgb_channels(g=change["new"])

    def toggle_blue_channel(self, change):
        """Toggles the blue channel on/off."""
        self.preview.set_rgb_channels(b=change["new"])

    def select_normalisation(self, change):
        """Updates the normalization method when dropdown selection changes."""
        new_value = change["new"]
        if new_value != self.cfg.normalisation.normalisation_method:
            self.session.set_normalisation_method(new_value)
            self.preview.update_display()

    def unlabel_current_image(self):
        """Removes the label from the currently displayed image."""
        self.session.unlabel_image(self.preview.current_index)
        self.preview.update_label_only()
