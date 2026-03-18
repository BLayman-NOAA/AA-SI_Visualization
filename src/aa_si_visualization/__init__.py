# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""
aa_si_visualization - Visualization tools for NOAA Fisheries AA-SI echogram and sonar data.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aa-si-visualization")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without install)
    __version__ = "0.0.0.dev"

from .echogram import (
    plot_sv_echogram,
    plot_cluster_echogram,
    plot_flattend_data_echogram,
    plot_processed_echogram_main,
)
from .assorted import sv_differences_echograms

__all__ = [
    "__version__",
    "plot_sv_echogram",
    "plot_cluster_echogram",
    "plot_flattend_data_echogram",
    "plot_processed_echogram_main",
    "sv_differences_echograms",
]
