"""Helpers for figure display and persistence across runtime environments."""

from __future__ import annotations

from pathlib import Path


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

    if save_dir is not None:
        resolved_save_dir = Path(save_dir)
    elif output_dir is not None:
        resolved_save_dir = Path(output_dir) / IMAGE_OUTPUT_DIR
    else:
        resolved_save_dir = Path.cwd() / IMAGE_OUTPUT_DIR

    stem = step_id or default_stem
    if artifact_suffix:
        stem = f"{stem}_{artifact_suffix}"

    written_paths = []
    if resolved_formats:
        resolved_save_dir.mkdir(parents=True, exist_ok=True)
        resolved_dpi = 300 if dpi is None else dpi
        for fmt in resolved_formats:
            path = resolved_save_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=resolved_dpi, bbox_inches="tight", format=fmt)
            written_paths.append(path)

    if resolved_show:
        import matplotlib.pyplot as plt

        plt.show()

    import matplotlib.pyplot as plt

    plt.close(fig)
    return written_paths