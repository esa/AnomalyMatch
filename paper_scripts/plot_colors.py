#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Colors for AnomalyMatch plots

This module defines a consistent, accessible color scheme for all plots
in the AnomalyMatch paper.
"""

BLUE = "#1f77b4"  # Steel blue for main curves and AUROC
RED = "#d62728"  # Brick red for anomalies/error highlighting and AUPRC
GREEN = "#2ca02c"  # Medium green for success/improvement
ORANGE = "#ff7f0e"  # Dark orange for comparison/reference
PURPLE = "#9467bd"  # Medium purple for tertiary content
BLACK = "#000000"  # Black for text and axes

# Color for perfect detection line (consistent across plots)
PERFECT_LINE_COLOR = RED
PERFECT_LINE_STYLE = "--"
PERFECT_LINE_ALPHA = 0.7

# Reference lines
REFERENCE_LINE_COLOR = "gray"
REFERENCE_LINE_STYLE = ":"
REFERENCE_LINE_ALPHA = 0.7

# Vertical indicator lines
VLINE_COLOR = BLACK
VLINE_STYLE = ":"
VLINE_ALPHA = 0.7

# Horizontal indicator lines
HLINE_COLOR = PURPLE
HLINE_STYLE = "--"
HLINE_ALPHA = 0.7

# Colormap for multiple iterations/classes (viridis is colorblind-friendly)
COLORMAP_NAME = "tab20b"

# Last iteration (should match anomaly detection rate)
LAST_ITER_COLOR = BLUE

# For histograms
NORMAL_COLOR = BLUE
ANOMALY_COLOR = RED
HIST_ALPHA = 0.5
