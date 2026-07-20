"""Helpers for figure display and persistence across runtime environments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+://")
_LOCAL_PROTOCOLS = frozenset({"file", "local"})


def _is_remote(value: Any) -> bool:
    """True when a directory value is a non-local fsspec URL / remote location."""
    if value is None:
        return False
    is_local = getattr(value, "is_local", None)
    if isinstance(is_local, bool):
        return not is_local
    if isinstance(value, Path):
        return False
    match = _URL_SCHEME_RE.match(str(value))
    if match is None:
        return False
    return str(value)[: match.end() - 3].lower() not in _LOCAL_PROTOCOLS


def _storage_options(value: Any) -> dict | None:
    opts = getattr(value, "storage_options", None)
    return dict(opts) if opts else None


def _join_dir(base: Any, segment: str) -> Path | str:
    """Append a path segment; Path for local bases, URL str for remote."""
    if _is_remote(base):
        return str(base).rstrip("/") + "/" + segment
    return Path(base) / segment


def _save_figure_to(fig, directory: Any, filename: str, *, dpi, fmt) -> Path | str:
    """Save a figure into ``directory`` (local dir or fsspec URL); return its path."""
    if _is_remote(directory):
        import fsspec

        url = str(directory).rstrip("/") + "/" + filename
        with fsspec.open(url, "wb", **(_storage_options(directory) or {})) as fh:
            fig.savefig(fh, dpi=dpi, bbox_inches="tight", format=fmt)
        return url
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
    return path


IMAGE_OUTPUT_DIR = "images"
_SUPPORTED_FORMATS = {
    "png": "png",
    "svg": "svg",
    "jpeg": "jpeg",
    "jpg": "jpg",
    "pdf": "pdf",
}


def _load_execution_context():
    try:
        from aa_recipe_manager.executor.runtime_context import get_execution_context
        return get_execution_context()
    except Exception:
        return None


def configure_matplotlib_backend() -> None:
    """Select a non-interactive backend for direct recipe execution.

    Generated notebooks should keep the notebook/frontend-selected inline
    backend. Direct recipe execution should avoid GUI backends such as TkAgg
    so figures do not create Tk objects or block the pipeline.
    """
    ctx = _load_execution_context()
    mode = getattr(ctx, "mode", None)
    output_dir = getattr(ctx, "output_dir", None)
    if mode != "direct" or output_dir is None:
        return

    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)


def _normalize_formats(save_formats):
    if save_formats is None:
        return None
    normalized = []
    for fmt in save_formats:
        key = str(fmt).strip().lower()
        if key not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported image format {fmt!r}; expected one of "
                f"{sorted(_SUPPORTED_FORMATS)}"
            )
        normalized.append(_SUPPORTED_FORMATS[key])
    return normalized


def render_figure(
    fig,
    *,
    default_stem,
    artifact_suffix=None,
    save_image=None,
    save_formats=None,
    save_dir=None,
    show=None,
    dpi=None,
):
    """Save and/or show a matplotlib figure based on runtime defaults."""
    ctx = _load_execution_context()
    mode = getattr(ctx, "mode", None)
    output_dir = getattr(ctx, "output_dir", None)
    artifacts_dir = getattr(ctx, "artifacts_dir", None)
    step_id = getattr(ctx, "step_id", None)

    normalized_formats = _normalize_formats(save_formats)
    if normalized_formats:
        resolved_save_image = True if save_image is None else bool(save_image)
    elif save_image is not None:
        resolved_save_image = bool(save_image)
    else:
        resolved_save_image = bool(mode == "direct" and output_dir is not None)

    if show is not None:
        resolved_show = bool(show)
    else:
        resolved_show = not (mode == "direct" and output_dir is not None)

    if resolved_save_image:
        resolved_formats = normalized_formats or ["png"]
    else:
        resolved_formats = []

    # An explicit save_dir is used as-is; artifacts_dir/output_dir get the
    # IMAGE_OUTPUT_DIR subfolder. Any of these may be a local path or an fsspec
    # URL (gs://...) — a StorageLocation from the recipe executor, or a raw URL.
    if save_dir is not None:
        resolved_save_dir: Any = save_dir
    elif artifacts_dir is not None:
        resolved_save_dir = _join_dir(artifacts_dir, IMAGE_OUTPUT_DIR)
    elif output_dir is not None:
        resolved_save_dir = _join_dir(output_dir, IMAGE_OUTPUT_DIR)
    else:
        resolved_save_dir = Path.cwd() / IMAGE_OUTPUT_DIR

    stem = step_id or default_stem
    if artifact_suffix:
        stem = f"{stem}_{artifact_suffix}"

    # Record artifacts relative to artifacts_dir only when we saved under it
    # (the recipe-executor case). A save_dir override or an output_dir/cwd
    # fallback is not relativizable, so nothing is recorded — the executor then
    # treats the step as unverifiable and regenerates it.
    artifact_sink = getattr(ctx, "artifact_sink", None)
    record_relative = save_dir is None and artifacts_dir is not None

    written_paths = []
    if resolved_formats:
        resolved_dpi = 300 if dpi is None else dpi
        for fmt in resolved_formats:
            filename = f"{stem}.{fmt}"
            written_paths.append(
                _save_figure_to(
                    fig, resolved_save_dir, filename, dpi=resolved_dpi, fmt=fmt
                )
            )
            if artifact_sink is not None and record_relative:
                artifact_sink.append(f"{IMAGE_OUTPUT_DIR}/{filename}")

    if resolved_show:
        import matplotlib.pyplot as plt

        plt.show()

    import matplotlib.pyplot as plt

    plt.close(fig)
    return written_paths