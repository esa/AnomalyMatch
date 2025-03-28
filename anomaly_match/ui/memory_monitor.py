#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import psutil
import ipywidgets as widgets
import asyncio
import torch
from loguru import logger
from contextlib import nullcontext
from datetime import datetime


class MemoryMonitor:
    """Monitors system and GPU memory usage and displays it in a widget."""

    def __init__(self, update_interval=5.0):
        """Initialize the memory monitor widget.

        Args:
            update_interval (float): How often to update the memory usage display in seconds
        """
        self.update_interval = update_interval
        self.out = None  # Store output widget

        # Create widgets with explicit width to ensure visibility
        self.memory_text = widgets.HTML(
            value="Memory: -- GB",
            layout=widgets.Layout(
                background_color="black",
                padding="5px",
                width="300px",  # Made wider to accommodate timestamp
                margin="0px 0px 0px 10px",
                display="flex",
            ),
            style={"color": "white"},
        )

        self._task = None
        self._running = False

    def set_output_widget(self, out):
        """Set the output widget for logging.

        Args:
            out: Output widget for logging
        """
        self.out = out

    async def update_memory(self):
        """Continuously update memory usage display."""
        with self.out if self.out is not None else nullcontext():
            try:
                while self._running:
                    try:
                        # Get current time
                        current_time = datetime.now().strftime("%H:%M:%S")

                        # Get system memory usage
                        process = psutil.Process()
                        sys_mem = process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

                        # Get GPU memory if available
                        gpu_mem = ""
                        if torch.cuda.is_available():
                            gpu_mem = f" | GPU: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB"

                        # Format with left alignment and timestamp
                        self.memory_text.value = (
                            f'<div style="text-align: left; color: white;">'
                            f"Memory: {sys_mem:.2f} GB{gpu_mem} [{current_time}]"
                            f"</div>"
                        )

                        # Log detailed memory info at DEBUG level
                        logger.debug(f"Memory usage: RAM={sys_mem:.2f}GB{gpu_mem}")

                    except Exception as e:
                        logger.error(f"Error updating memory info: {str(e)}")
                        self.memory_text.value = (
                            '<div style="text-align: left; color: white;">Memory: Error</div>'
                        )

                    await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                logger.debug("Memory monitor task cancelled")
                raise
            except Exception as e:
                logger.error(f"Memory monitor task failed: {str(e)}")
                self.memory_text.value = (
                    '<div style="text-align: left; color: white;">Memory: Task Failed</div>'
                )
                raise

    def start(self):
        """Start monitoring memory usage."""
        with self.out if self.out is not None else nullcontext():
            if not self._running:
                self._running = True
                try:
                    loop = asyncio.get_event_loop()
                    self._task = loop.create_task(self.update_memory())
                    logger.debug("Memory monitor started")
                except Exception as e:
                    logger.error(f"Failed to start memory monitor: {str(e)}")
                    self._running = False

    def stop(self):
        """Stop monitoring memory usage."""
        with self.out if self.out is not None else nullcontext():
            logger.debug("Stopping memory monitor")
            self._running = False
            if self._task:
                self._task.cancel()
                self._task = None

    def __del__(self):
        """Cleanup when the monitor is destroyed."""
        if self._running:
            logger.debug("Memory monitor being destroyed, stopping monitoring")
            self.stop()
