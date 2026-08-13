# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Every entry point must accept every documented axis unit.

The plotters each validate x_axis_units separately, for Sv, MVBS and ML data.
When those lists were written out by hand they drifted: 'datetime' was added to
the Sv list but not the ML one, so a pipeline ran for four minutes and then
failed on the ML plot. These tests walk the units rather than trusting the
lists to stay in step.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aa_si_visualization import echogram
from aa_si_visualization._plotting_utils import X_AXIS_UNITS

N_PINGS = 40
N_SAMPLES = 20

# 'meters' needs a GPS track or an explicit speed, and 'bins' is MVBS-only, so
# neither belongs in the plain-Sv sweep.
SV_UNITS = [u for u in X_AXIS_UNITS if u not in ("bins", "meters")]


def _sv_dataset():
    depths = np.linspace(0, 25, N_SAMPLES)
    ping_time = pd.date_range("2024-10-15T11:42", periods=N_PINGS, freq="1s").values
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"),
                   rng.normal(-60, 5, size=(2, N_PINGS, N_SAMPLES))),
            "echo_range": (("channel", "ping_time", "range_sample"),
                           np.broadcast_to(depths, (2, N_PINGS, N_SAMPLES)).copy()),
            "frequency_nominal": (("channel",), np.array([18000.0, 38000.0])),
        },
        coords={
            "channel": ["ch0", "ch1"],
            "ping_time": ping_time,
            "range_sample": np.arange(N_SAMPLES),
        },
    )


def _mvbs_ml_dataset():
    """MVBS-derived ML data: a 1-D echo_range is what marks it as MVBS.

    This is the shape the recipe pipeline produces (reshape_for_ml on MVBS ->
    normalize_ml) and the branch whose unit list went stale.
    """
    ping_time = pd.date_range("2024-10-15T11:42", periods=N_PINGS, freq="1s").values
    echo_range = np.linspace(0, 25, N_SAMPLES)
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "echo_range"),
                   rng.normal(-60, 5, size=(2, N_PINGS, N_SAMPLES))),
            "frequency_nominal": (("channel",), np.array([18000.0, 38000.0])),
            "ml_dataset_normalized": (("ping_time", "echo_range"),
                                      rng.integers(0, 4, size=(N_PINGS, N_SAMPLES)).astype(float)),
        },
        coords={
            "channel": ["ch0", "ch1"],
            "ping_time": ping_time,
            "echo_range": echo_range,
        },
    )


@pytest.mark.parametrize("units", SV_UNITS)
def test_plot_sv_echogram_accepts_every_unit(units):
    echogram.plot_sv_echogram(
        _sv_dataset(), max_depth=25, x_axis_units=units,
        save_image=False, show=False,
    )


# 'bins' is valid here, since this path is MVBS-derived by construction.
ML_UNITS = [u for u in X_AXIS_UNITS if u != "meters"]


@pytest.mark.parametrize("units", ML_UNITS)
def test_plot_ml_echogram_accepts_every_unit(units):
    echogram.plot_flattened_data_echogram(
        _mvbs_ml_dataset(), ml_dataset_name="ml_dataset",
        ml_specific_data_name="normalized",
        ds_Sv_original=_sv_dataset(),
        max_depth=25, x_axis_units=units,
        save_image=False, show=False,
    )


def test_default_unit_is_accepted_everywhere():
    """The default must be in the shared list, or every plot fails by default."""
    import inspect

    for fn in (echogram.plot_sv_echogram,
               echogram.plot_flattened_data_echogram,
               echogram.plot_cluster_echogram,
               echogram.plot_processed_echogram_main):
        default = inspect.signature(fn).parameters["x_axis_units"].default
        assert default in X_AXIS_UNITS, f"{fn.__name__} defaults outside X_AXIS_UNITS"


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError, match="Invalid x_axis_units"):
        echogram.plot_sv_echogram(
            _sv_dataset(), max_depth=25, x_axis_units="clock",
            save_image=False, show=False,
        )
