# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for figure persistence to local and remote (gs://-shaped) outputs.

``memory://`` stands in for ``gs://`` (a non-local fsspec store, no credentials).
"""

from __future__ import annotations

from pathlib import Path

import fsspec
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aa_si_visualization import _artifact_output as ao  # noqa: E402


@pytest.fixture(autouse=True)
def clear_memory_fs():
    mem = fsspec.filesystem("memory")
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]
    yield
    mem.store.clear()
    mem.pseudo_dirs[:] = [""]


@pytest.fixture
def a_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [2, 1, 0])
    yield fig
    plt.close(fig)


def test_is_remote_and_join():
    assert ao._is_remote("gs://b/x") and ao._is_remote("memory://c/x")
    assert not ao._is_remote(Path("x")) and not ao._is_remote(r"C:\t")
    assert ao._join_dir("memory://c/out", "images") == "memory://c/out/images"
    assert ao._join_dir(Path(r"C:\t"), "images") == Path(r"C:\t") / "images"


def test_render_figure_saves_locally(a_figure, tmp_path):
    written = ao.render_figure(
        a_figure,
        default_stem="plot",
        save_dir=str(tmp_path),
        save_formats=["png"],
        show=False,
    )
    assert len(written) == 1
    assert Path(written[0]).exists()
    assert Path(written[0]).name == "plot.png"


def test_render_figure_saves_to_remote_bucket(a_figure):
    written = ao.render_figure(
        a_figure,
        default_stem="plot",
        save_dir="memory://outputs/images",
        save_formats=["png", "svg"],
        show=False,
    )
    assert written == [
        "memory://outputs/images/plot.png",
        "memory://outputs/images/plot.svg",
    ]
    fs = fsspec.filesystem("memory")
    for url in written:
        assert fs.exists(url)
        assert fs.size(url) > 0  # real bytes written


def test_render_figure_remote_via_artifacts_dir_context(a_figure):
    """artifacts_dir from the execution context gets the images/ subfolder."""
    from aa_recipe_manager.executor.runtime_context import execution_context

    with execution_context(
        mode="direct", artifacts_dir="memory://run/outputs", step_id="plot_sv"
    ):
        written = ao.render_figure(
            a_figure, default_stem="fallback", save_formats=["png"], show=False
        )
    # step_id wins over default_stem; images/ subfolder appended to artifacts_dir.
    assert written == ["memory://run/outputs/images/plot_sv.png"]
    assert fsspec.filesystem("memory").exists(written[0])
