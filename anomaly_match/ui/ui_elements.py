#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox
from IPython.display import HTML
from anomaly_match import __version__
from anomaly_match.ui.memory_monitor import MemoryMonitor

HTML_setup = HTML(
    """
                <style>
                    .widget-label, .widget-slider {
                        background-color: black !important;
                        color: white !important;
                    }
                    .widget-output {
                        background-color: black !important;
                        color: white !important;
                        border: 1px solid white !important;
                    }
                    .widget-box {
                        background-color: black !important;
                    }
                    .widget-hbox, .widget-vbox {
                        background-color: black !important;
                    }
                    .widget-slider .slider {
                        background-color: black !important;
                    }
                    .widget-slider .widget-readout {
                        color: white !important;
                    }
                    .widget-text {
                        background-color: black !important;
                        color: white !important;
                    }
                    .widget-button[button_style="primary"] {
                        background-color: #007bff !important;
                        color: black !important;
                    }
                    .widget-button[button_style="info"] {
                        background-color: #17a2b8 !important;
                        color: black !important;
                    }
                    .widget-button[button_style="warning"] {
                        background-color: #ffc107 !important;
                        color: black !important;
                    }
                    .widget-button[button_style="success"] {
                        background-color: #28a745 !important;
                        color: black !important;
                    }
                    .widget-output pre {
                        color: #d5d5d5 !important;
                        background-color: black !important;
                    }
                </style>
                """
)

# Define the button names
dataset_button_names = [
    "Most Anomalous",
    "Most Nominal",
    "Closest to Mean",
    "Closest to Median",
]
skip_button_names = ["Previous", "Next"]
decision_button_names = ["Anomalous", "Nominal"]
additional_button_names = [
    "Save Model",
    "Load Model",
    "Save Labels",
    "Load Top Files",
]
transform_button_names = ["Invert Image", "Restore", "Apply Unsharp Mask"]


def create_ui_elements():
    """
    Creates and returns the necessary UI widgets as a dictionary.
    """

    # HTML widget
    filename_text = widgets.HTML(
        value="",
        layout=widgets.Layout(background_color="black"),
        style={"color": "white"},
    )

    # Image widget
    image_widget = widgets.Image(
        value=b"",  # Just an empty placeholder
        width=600,
        height=600,
        layout=widgets.Layout(background_color="black"),
    )

    # Dataset, skip, decision, additional, transform buttons
    dset_buttons = [
        Button(
            description=w,
            button_style="primary",
            layout=widgets.Layout(background_color="black"),
        )
        for w in dataset_button_names
    ]

    skip_buttons = [
        Button(
            description=w,
            button_style="primary",
            layout=widgets.Layout(background_color="black"),
        )
        for w in skip_button_names
    ]

    # Update button colors for consistency
    GREEN_COLOR = "#28a745"  # Bootstrap success green
    RED_COLOR = "#dc3545"  # Bootstrap danger red

    decision_buttons = [
        Button(
            description=w,
            button_style="info",
            layout=widgets.Layout(background_color="black"),
            style={"button_color": RED_COLOR if "Anomalous" in w else GREEN_COLOR},
        )
        for w in decision_button_names
    ]

    # Define colors
    ORANGE_COLOR = "#fd7e14"  # Bootstrap orange
    WHITE_COLOR = "#ffffff"  # White

    # Create Remember button with orange color
    remember_button = Button(
        description="Remember",
        button_style="warning",
        layout=widgets.Layout(background_color="black"),
        style={"button_color": ORANGE_COLOR},
    )

    # Update transform buttons with white background but black text
    transform_buttons = [
        Button(
            description=w,
            button_style="success",
            layout=widgets.Layout(background_color="black"),
            style={"button_color": WHITE_COLOR, "text_color": "black"},
        )
        for w in transform_button_names
    ]

    # Sliders with adjusted widths for side-by-side display
    brightness_slider = widgets.FloatSlider(
        value=1.0,
        min=0.5,
        max=2.0,
        step=0.01,
        description="Brightness",
        continuous_update=True,
        layout=widgets.Layout(background_color="black", width="50%"),
        style={"description_width": "initial", "handle_color": "white"},
    )

    contrast_slider = widgets.FloatSlider(
        value=1.0,
        min=0.5,
        max=2.0,
        step=0.01,
        description="Contrast",
        continuous_update=True,
        layout=widgets.Layout(background_color="black", width="50%"),
        style={"description_width": "initial", "handle_color": "white"},
    )

    # RGB channel toggle checkboxes - more compact layout
    channel_label = widgets.HTML(
        value='<div style="color:white; margin-right:5px;">RGB:</div>',
        layout=widgets.Layout(background_color="black", width="35px"),
    )

    red_channel_checkbox = widgets.Checkbox(
        value=True,
        description="R",
        indent=False,
        layout=widgets.Layout(background_color="black", width="35px"),
        style={"description_width": "15px"},
    )

    green_channel_checkbox = widgets.Checkbox(
        value=True,
        description="G",
        indent=False,
        layout=widgets.Layout(background_color="black", width="35px"),
        style={"description_width": "15px"},
    )

    blue_channel_checkbox = widgets.Checkbox(
        value=True,
        description="B",
        indent=False,
        layout=widgets.Layout(background_color="black", width="35px"),
        style={"description_width": "15px"},
    )

    # Group RGB checkboxes in a compact horizontal box
    channel_controls = HBox(
        [channel_label, red_channel_checkbox, green_channel_checkbox, blue_channel_checkbox],
        layout=widgets.Layout(
            background_color="black", width="180px", justify_content="flex-start"
        ),
    )

    # Create a slider container with both sliders and the RGB controls
    slider_row = HBox(
        [brightness_slider, contrast_slider, channel_controls],
        layout=widgets.Layout(background_color="black"),
    )

    # Create transform controls with new slider arrangement
    transform_controls = VBox(
        [
            HBox(
                transform_buttons + [remember_button]
            ),  # Add remember button to transform controls
            slider_row,  # Single row with both sliders and RGB toggles
        ],
        layout=widgets.Layout(background_color="black"),
    )

    batch_size_slider = widgets.IntSlider(
        value=1000,
        min=500,
        max=50000,
        step=500,
        description="Batch Size",
        continuous_update=True,
        layout=widgets.Layout(background_color="black"),
        style={"description_width": "initial", "handle_color": "white"},
    )

    # Buttons
    next_batch_button = Button(
        description="Next Batch",
        button_style="primary",
        layout=widgets.Layout(background_color="black"),
        style={"color": "black"},
    )

    reset_model_button = Button(
        description="Reset Model",
        button_style="danger",
        layout=widgets.Layout(background_color="black"),
        style={"button_color": RED_COLOR},
    )

    train_button = Button(
        description="Train",
        button_style="primary",
        layout=widgets.Layout(background_color="black"),
        style={"color": "black"},
    )

    train_iteration_slider = widgets.IntSlider(
        value=50,
        min=50,
        max=100000,
        step=50,
        description="Train Iterations",
        continuous_update=True,
        style={"description_width": "initial", "handle_color": "white", "text_color": "white"},
        layout=widgets.Layout(background_color="black"),
    )

    train_label = widgets.Label(
        value="", layout=widgets.Layout(background_color="black"), style={"color": "white"}
    )

    progress_bar = widgets.FloatProgress(
        value=0.0,
        min=0.0,
        max=1.0,
        layout=widgets.Layout(background_color="black"),
        style={"bar_color": "green"},
    )

    out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid white",
            height="400px",
            background_color="black",
            overflow="auto",
        ),
        style={"color": "white"},
    )

    distribution_plot = widgets.Output(layout=widgets.Layout(background_color="black"))

    gallery = widgets.Output(layout=widgets.Layout(background_color="black"))

    # Create version and memory monitor display with more explicit styling
    version_text = widgets.HTML(
        value=f'<div style="text-align: left; color: white;">Version: {__version__}</div>',
        layout=widgets.Layout(
            background_color="black",
            padding="3px",
            width="200px",  # Match memory monitor width
            margin="10px",
        ),
    )

    memory_monitor = MemoryMonitor()

    # Create info box for bottom right with explicit styling
    info_box = VBox(
        [version_text, memory_monitor.memory_text],
        layout=widgets.Layout(
            background_color="black",
            # border="1px solid #555",
            padding="3px",
            margin="3px",
            width="auto",
            min_width="100px",  # Ensure minimum width
            align_items="flex-start",  # Align items to the left
        ),
    )

    # Create side display
    side_display = VBox(
        [gallery, info_box],
        layout=widgets.Layout(background_color="black"),
    )

    # Create additional buttons first
    additional_buttons = [
        Button(
            description=w,
            button_style="",
            layout=widgets.Layout(background_color="black"),
            style={"color": "black"},
        )
        for w in additional_button_names
    ]

    # Keep original top row with dataset buttons only
    top_row = HBox(
        dset_buttons,
        layout=widgets.Layout(background_color="black"),
    )

    center_row = VBox(
        [filename_text, image_widget],
        layout=widgets.Layout(background_color="black"),
    )

    # Create model controls row (previously bottom_row2) and move it to the top
    model_controls = HBox(
        additional_buttons,  # All model-related buttons
        layout=widgets.Layout(background_color="black"),
    )

    search_all_files_button = Button(
        description="Evaluate Search Dir",
        button_style="primary",
        layout=widgets.Layout(background_color="black"),
        style={"color": "black"},
    )

    # Update bottom row layout
    bottom_row1 = HBox(
        [
            skip_buttons[0],
            decision_buttons[0],
            decision_buttons[1],
            skip_buttons[1],
        ],
        layout=widgets.Layout(background_color="black"),
    )

    bottom_row3 = HBox(
        [train_iteration_slider, batch_size_slider],
        layout=widgets.Layout(background_color="black"),
    )

    bottom_row4 = HBox(
        [reset_model_button, next_batch_button, train_button, search_all_files_button],
        layout=widgets.Layout(background_color="black"),
    )

    bottom_row5 = HBox(
        [progress_bar, train_label],
        layout=widgets.Layout(background_color="black"),
    )

    # Combine all rows, putting model_controls at the top
    main_layout = VBox(
        [
            model_controls,
            top_row,
            center_row,
            transform_controls,
            bottom_row1,
            bottom_row3,
            bottom_row4,
            bottom_row5,
        ],
        layout=widgets.Layout(background_color="black"),
    )

    return {
        "filename_text": filename_text,
        "image_widget": image_widget,
        "dset_buttons": dset_buttons,
        "skip_buttons": skip_buttons,
        "decision_buttons": decision_buttons,
        "additional_buttons": additional_buttons,
        "transform_buttons": transform_buttons,
        "brightness_slider": brightness_slider,
        "contrast_slider": contrast_slider,
        "batch_size_slider": batch_size_slider,
        "next_batch_button": next_batch_button,
        "reset_model_button": reset_model_button,
        "train_button": train_button,
        "train_iteration_slider": train_iteration_slider,
        "train_label": train_label,
        "progress_bar": progress_bar,
        "out": out,
        "distribution_plot": distribution_plot,
        "gallery": gallery,
        "side_display": side_display,
        "top_row": top_row,
        "center_row": center_row,
        "transform_controls": transform_controls,
        "search_all_files_button": search_all_files_button,
        "bottom_row1": bottom_row1,
        "bottom_row3": bottom_row3,
        "bottom_row4": bottom_row4,
        "bottom_row5": bottom_row5,
        "main_layout": main_layout,
        "remember_button": remember_button,
        "memory_monitor": memory_monitor,
        "red_channel_checkbox": red_channel_checkbox,
        "green_channel_checkbox": green_channel_checkbox,
        "blue_channel_checkbox": blue_channel_checkbox,
    }


def attach_click_listeners(widget):
    """
    Attaches click handlers / observers to the widget's UI elements.
    This function expects the widget to have all needed methods like
    sort_by_anomalous, previous_image, etc.
    """

    # Dataset sort buttons
    widget.ui["dset_buttons"][0].on_click(lambda _: widget.sort_by_anomalous())
    widget.ui["dset_buttons"][1].on_click(lambda _: widget.sort_by_nominal())
    widget.ui["dset_buttons"][2].on_click(lambda _: widget.sort_by_mean())
    widget.ui["dset_buttons"][3].on_click(lambda _: widget.sort_by_median())

    # Skip buttons
    widget.ui["skip_buttons"][0].on_click(lambda _: widget.previous_image())
    widget.ui["skip_buttons"][1].on_click(lambda _: widget.next_image())

    # Decision buttons
    def mark_anomalous(_):
        widget.session.label_image(widget.current_index, "anomaly")
        widget.update_image_UI_label()

    def mark_nominal(_):
        widget.session.label_image(widget.current_index, "normal")
        widget.update_image_UI_label()

    widget.ui["decision_buttons"][0].on_click(mark_anomalous)
    widget.ui["decision_buttons"][1].on_click(mark_nominal)

    # Transform buttons
    widget.ui["transform_buttons"][0].on_click(lambda _: widget.toggle_invert_image())
    widget.ui["transform_buttons"][1].on_click(lambda _: widget.restore_image())
    widget.ui["transform_buttons"][2].on_click(lambda _: widget.toggle_unsharp_mask())

    # Brightness/Contrast observers
    widget.ui["brightness_slider"].observe(widget.adjust_brightness_contrast, names="value")
    widget.ui["contrast_slider"].observe(widget.adjust_brightness_contrast, names="value")

    # Additional buttons
    widget.ui["additional_buttons"][0].on_click(lambda _: widget.save_model())
    widget.ui["additional_buttons"][1].on_click(lambda _: widget.load_model())
    widget.ui["additional_buttons"][2].on_click(lambda _: widget.save_labels())
    widget.ui["additional_buttons"][3].on_click(lambda _: widget.load_top_files())

    # Batch size
    widget.ui["batch_size_slider"].observe(widget.update_batch_size, names="value")

    # Next batch
    widget.ui["next_batch_button"].on_click(lambda _: widget.next_batch())

    # Reset
    widget.ui["reset_model_button"].on_click(lambda _: widget.reset_model())

    # Train
    widget.ui["train_button"].on_click(lambda _: widget.train())

    # Search all files
    widget.ui["search_all_files_button"].on_click(lambda _: widget.search_all_files())
