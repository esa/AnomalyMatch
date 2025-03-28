#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
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

# Import the newly created UI elements
from anomaly_match.ui.ui_elements import create_ui_elements, attach_click_listeners, HTML_setup
from anomaly_match.ui.utility_functions import apply_transforms
from anomaly_match.utils.numpy_to_byte_stream import numpy_array_to_byte_stream


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

        # Create all UI elements (moved from the big monolithic class)
        self.ui = create_ui_elements()

        # This is to keep track of the current image index
        self.current_index = 0

        # Initialize transformation states
        self.invert = False
        self.brightness = 1.0
        self.contrast = 1.0
        self.unsharp_mask_applied = False
        self.original_image = None
        self.modified_image = None

        # Initialize RGB channel states
        self.show_r = True
        self.show_g = True
        self.show_b = True

        # Attach the output widget so the session logs go there
        session.set_terminal_out(self.ui["out"])
        # Also attach it to the memory monitor
        self.ui["memory_monitor"].set_output_widget(self.ui["out"])

        # Sync initial batch slider value with session config
        self.ui["batch_size_slider"].value = self.cfg.N_to_load

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
        return HBox([main_layout, side_display])

    def search_all_files(self):
        """Searches all files and displays the top 1000 with their scores."""
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
                        self.ui["train_label"].value = (
                            f"Search complete in {total_time_str} ({final_speed:.1f} img/sec)"
                        )
                        return

                    # Handle batch update with ETA information
                    if batch_update:
                        if eta_str:
                            # Use the ETA info provided by session.py
                            message = f"Searching files... Batch: {batch}/{num_batches}"
                            if progress_percent is not None:
                                message += f" ({progress_percent:.1f}%)"
                            if images_per_second is not None:
                                message += f" | {images_per_second:.1f} img/sec"
                            message += f" | ETA: {eta_str}"
                            self.ui["train_label"].value = message
                        else:
                            # Early in the process when ETA isn't available yet
                            self.ui["train_label"].value = (
                                f"Searching files... Batch: {batch}/{num_batches}"
                            )
                    else:
                        # Regular evaluation updates (not used in this function but keeping for completeness)
                        if eta_str:
                            self.ui["train_label"].value = (
                                f"Evaluating... {batch}/{num_batches} | ETA: {eta_str}"
                            )
                        else:
                            self.ui["train_label"].value = f"Evaluating... {batch}/{num_batches}"

            self.session.evaluate_all_images(top_N=5000, progress_callback=update_progress)

            # Display will be updated by the callback when completed
            self.display_top_files_scores()
            self.ui["progress_bar"].style = {"bar_color": "green"}

    def display_top_files_scores(self):
        """Displays the top files and their scores."""
        self.current_index = 0
        self.update_image_display()
        self.ui["progress_bar"].style = {"bar_color": "green"}
        self.display_gallery()

    def update_image_display(self):
        """Updates the display of the current image."""
        img = self.session.img_catalog[self.current_index]
        filename = self.session.filenames[self.current_index]
        score = self.session.scores[self.current_index]

        # Normalize the image array to 0-1 range, then to 255
        img = img - np.min(img)
        img = img / np.max(img)
        img = (img * 255).astype(np.uint8)
        if img.shape[-1] == 1:  # Convert grayscale to RGB if necessary
            img = np.repeat(img, 3, axis=-1)

        self.original_image = Image.fromarray(img)
        self.modified_image = apply_transforms(
            self.original_image,
            invert=self.invert,
            brightness=self.brightness,
            contrast=self.contrast,
            unsharp_mask_applied=self.unsharp_mask_applied,
            show_r=self.show_r,
            show_g=self.show_g,
            show_b=self.show_b,
        )
        self.display_image(self.modified_image, filename, score)

    def display_image(self, img, filename=None, score=None):
        """Displays the given PIL image in the widget."""
        image_byte_stream = numpy_array_to_byte_stream(np.array(img))
        self.ui["image_widget"].value = image_byte_stream
        self.update_image_UI_label(filename, score)

    def update_image_UI_label(self, filename=None, score=None):
        """Updates the UI label with the current image's filename, score, and label."""
        label_color = "white"
        label_text = "None"
        label = self.session.get_label(self.current_index)
        if label == "anomaly":
            label_color = "red"
            label_text = "Anomalous"
        elif label == "normal":
            label_color = "green"
            label_text = "Nominal"

        fname = self.session.filenames[self.current_index]
        sc = self.session.scores[self.current_index]
        total_len = len(self.session.img_catalog) - 1
        self.ui["filename_text"].value = (
            f'<span style="color:white">'
            f"Filename: {fname} | Score: {sc:.4f} | Index: {self.current_index} / {total_len}"
            f'</span> | <span style="color:{label_color}">Label: {label_text}</span>'
        )

    # ======== Sorting Methods ========
    def sort_by_anomalous(self):
        """Sorts the images by their anomalous scores and updates the display."""
        self.session.sort_by_anomalous()
        self.current_index = 0
        self.update_image_display()

    def sort_by_nominal(self):
        """Sorts the images by their nominal scores and updates the display."""
        self.session.sort_by_nominal()
        self.current_index = 0
        self.update_image_display()

    def sort_by_mean(self):
        """Sorts the images by distance to mean score and updates the display."""
        self.session.sort_by_mean()
        self.current_index = 0
        self.update_image_display()

    def sort_by_median(self):
        """Sorts the images by distance to median score and updates the display."""
        self.session.sort_by_median()
        self.current_index = 0
        self.update_image_display()

    # ======== Navigation ========
    def next_image(self):
        """Displays the next image in the catalog."""
        self.current_index = min(len(self.session.img_catalog) - 1, self.current_index + 1)
        self.update_image_display()

    def previous_image(self):
        """Displays the previous image in the catalog."""
        self.current_index = max(0, self.current_index - 1)
        self.update_image_display()

    # ======== Image Transformations ========
    def restore_image(self):
        """Restores the current image to its original state."""
        self.invert = False
        self.ui["brightness_slider"].value = 1.0
        self.ui["contrast_slider"].value = 1.0
        self.unsharp_mask_applied = False

        # Reset RGB channels
        self.show_r = True
        self.show_g = True
        self.show_b = True
        self.ui["red_channel_checkbox"].value = True
        self.ui["green_channel_checkbox"].value = True
        self.ui["blue_channel_checkbox"].value = True

        self.modified_image = self.original_image
        self.display_image(self.modified_image)

    def toggle_invert_image(self):
        """Toggles the inversion of the current image."""
        self.invert = not self.invert
        self.modified_image = apply_transforms(
            self.original_image,
            invert=self.invert,
            brightness=self.brightness,
            contrast=self.contrast,
            unsharp_mask_applied=self.unsharp_mask_applied,
        )
        self.display_image(self.modified_image)

    def toggle_unsharp_mask(self):
        """Toggles the application of an unsharp mask."""
        self.unsharp_mask_applied = not self.unsharp_mask_applied
        self.modified_image = apply_transforms(
            self.original_image,
            invert=self.invert,
            brightness=self.brightness,
            contrast=self.contrast,
            unsharp_mask_applied=self.unsharp_mask_applied,
        )
        self.display_image(self.modified_image)

    def adjust_brightness_contrast(self, _):
        """Adjusts brightness and contrast of the current image."""
        self.brightness = self.ui["brightness_slider"].value
        self.contrast = self.ui["contrast_slider"].value
        self.modified_image = apply_transforms(
            self.original_image,
            invert=self.invert,
            brightness=self.brightness,
            contrast=self.contrast,
            unsharp_mask_applied=self.unsharp_mask_applied,
        )
        self.display_image(self.modified_image)

    def display_gallery(self):
        """Displays a small gallery of either mispredicted or top anomalous/nominal images."""
        with self.ui["gallery"]:
            self.ui["gallery"].clear_output(wait=True)

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
                            with Image.open(path) as img:
                                # Convert to RGB and make a copy in memory
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                img_array = np.array(img)
                                mispredicted_images.append(img_array)
                                image_text.append(
                                    f"{filename}\nPred: {pred:.2f} | Label: {label:.2f}"
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
                    ax1.plot(fpr, tpr, "b-", label=f'ROC (AUC={eval_perf["eval/auroc"]:.3f})')
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
                        recall, precision, "g-", label=f'PRC (AUC={eval_perf["eval/auprc"]:.3f})'
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
                    img_arr = (img_arr * 255).astype(np.uint8)
                    if img_arr.shape[-1] == 1:
                        img_arr = np.repeat(img_arr, 3, axis=-1)
                    pil_img = Image.fromarray(img_arr)
                    images.append(pil_img)
                    filename = self.session.filenames[idx]
                    score = scores[idx]
                    image_text.append(f"{filename}\nScore: {score:.4f}")

                for idx in top_nominal_indices:
                    img_arr = self.session.img_catalog[idx]
                    img_arr = img_arr - np.min(img_arr)
                    img_arr = img_arr / np.max(img_arr)
                    img_arr = (img_arr * 255).astype(np.uint8)
                    if img_arr.shape[-1] == 1:
                        img_arr = np.repeat(img_arr, 3, axis=-1)
                    pil_img = Image.fromarray(img_arr)
                    images.append(pil_img)
                    filename = self.session.filenames[idx]
                    score = scores[idx]
                    image_text.append(f"{filename}\nScore: {score:.4f}")

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

    def save_labels(self):
        self.session.save_labels()

    def remember_current_file(self, _):
        """Remembers the currently displayed file."""
        self.session.remember_current_file(self.session.filenames[self.current_index])

    def save_model(self):
        """Saves the model using the session."""
        self.session.save_model()

    def load_model(self):
        """Loads the model using the session."""
        self.session.load_model()
        self.update()

    def train(self):
        """Starts the training process."""
        with self.ui["out"]:
            logger.debug("Starting training...")
            self.ui["progress_bar"].style = {"bar_color": "blue"}
            self.ui["progress_bar"].value = 0.0
            self.cfg.progress_bar = self.ui["progress_bar"]

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
                            f"Training... Iteration {iteration}/{total_iterations} | "
                            f"ETA: {eta_str}"
                        )
                    else:
                        # Early iterations - no reliable ETA yet
                        self.ui["train_label"].value = (
                            f"Training... Iteration {iteration}/{total_iterations}"
                        )

                    last_update_time = current_time
                    last_iteration = iteration

            self.session.train(self.cfg, progress_callback=update_training_progress)

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
        self.current_index = 0

        if self.cfg.test_ratio > 0:
            if self.session.eval_performance is not None:
                self.ui["train_label"].value = (
                    f"Training Complete. Eval Acc: {self.session.eval_performance['eval/top-1-acc'] * 100:.2f}%"
                )
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
            self.session.load_top_files()
            self.display_top_files_scores()

    # Add channel toggle methods
    def toggle_red_channel(self, change):
        """Toggles the red channel on/off."""
        self.show_r = change["new"]
        self.update_image_display()

    def toggle_green_channel(self, change):
        """Toggles the green channel on/off."""
        self.show_g = change["new"]
        self.update_image_display()

    def toggle_blue_channel(self, change):
        """Toggles the blue channel on/off."""
        self.show_b = change["new"]
        self.update_image_display()
