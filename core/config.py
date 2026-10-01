"""Load and normalize the plugin configuration.

AstrBot generates the user-facing config from ``_conf_schema.json`` and passes it
to ``Star.__init__``. The fallbacks below only matter when a key is missing or has
the wrong type, which can happen after a plugin update adds new settings. Keep the
defaults here in sync with ``_conf_schema.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger

OUTPUT_MODE_SEPARATE = "separate"
OUTPUT_MODE_GRID = "grid"
OUTPUT_MODES = frozenset({OUTPUT_MODE_SEPARATE, OUTPUT_MODE_GRID})


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: object, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


def _as_str(value: object, fallback: str) -> str:
    return value if isinstance(value, str) else fallback


def _as_str_list(value: object, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)
    items = [str(item).strip() for item in value if str(item).strip()]
    return items


def _as_int(value: object, fallback: int, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if minimum is not None and parsed < minimum:
        return fallback
    return parsed


def _as_float(value: object, fallback: float, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if minimum is not None and parsed < minimum:
        return fallback
    return parsed


@dataclass
class MarkdownImageConfig:
    """Settings for restoring Markdown image syntax into real images."""

    enable: bool = True
    url_whitelist: list[str] = field(default_factory=list)
    download_timeout: int = 10
    max_download_mb: float = 10.0
    placeholder: str = "[图片]"
    keep_alt_text: bool = False
    scan_quoted: bool = True

    @property
    def max_download_bytes(self) -> int:
        """Return the download size limit in bytes."""
        return int(self.max_download_mb * 1024**2)


@dataclass
class AnimationConfig:
    """Settings for splitting animated images into still frames."""

    enable: bool = True
    min_frames: int = 2
    max_frames: int = 4
    interval: int = 0
    ratio: float = 0.0
    skip_first: bool = True
    output_mode: str = OUTPUT_MODE_SEPARATE
    grid_columns: int = 0
    grid_label_frames: bool = True
    max_edge: int = 768
    keep_original: bool = False
    keep_original_copy: bool = False
    keep_original_max_files: int = 200
    keep_original_max_mb: int = 200
    scan_quoted: bool = True

    @property
    def expose_original_path(self) -> bool:
        """Whether the original file path should be handed to the model.

        A persistent copy is useless unless the model is told where it is, so
        enabling the copy also enables the path hint.
        """
        return self.keep_original or self.keep_original_copy


@dataclass
class PluginConfig:
    """Normalized plugin settings used by the message pipeline."""

    enabled: bool = True
    markdown_image: MarkdownImageConfig = field(default_factory=MarkdownImageConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    max_images_per_message: int = 10


def _load_markdown_image(raw: object) -> MarkdownImageConfig:
    """Build the Markdown image settings from raw config values.

    Args:
        raw: The ``markdown_image`` section of the plugin config.

    Returns:
        Normalized settings; missing or invalid values fall back to defaults.
    """
    data = _as_dict(raw)
    return MarkdownImageConfig(
        enable=_as_bool(data.get("enable"), True),
        url_whitelist=_as_str_list(data.get("url_whitelist"), []),
        download_timeout=_as_int(data.get("download_timeout"), 10, minimum=1),
        max_download_mb=_as_float(data.get("max_download_mb"), 10.0, minimum=0.1),
        placeholder=_as_str(data.get("placeholder"), "[图片]"),
        keep_alt_text=_as_bool(data.get("keep_alt_text"), False),
        scan_quoted=_as_bool(data.get("scan_quoted"), True),
    )


def _load_animation(raw: object) -> AnimationConfig:
    """Build the animation settings from raw config values.

    Args:
        raw: The ``animation`` section of the plugin config.

    Returns:
        Normalized settings. The frame bounds are also sorted so that
        ``min_frames`` never exceeds ``max_frames``.
    """
    data = _as_dict(raw)

    min_frames = _as_int(data.get("min_frames"), 2, minimum=1)
    max_frames = _as_int(data.get("max_frames"), 4, minimum=1)
    if min_frames > max_frames:
        logger.warning(
            "[VisionEnhance] min_frames (%d) is greater than max_frames (%d); "
            "using max_frames for both.",
            min_frames,
            max_frames,
        )
        min_frames = max_frames

    output_mode = _as_str(data.get("output_mode"), OUTPUT_MODE_SEPARATE)
    if output_mode not in OUTPUT_MODES:
        logger.warning(
            "[VisionEnhance] Unknown output_mode %r; falling back to %s.",
            output_mode,
            OUTPUT_MODE_SEPARATE,
        )
        output_mode = OUTPUT_MODE_SEPARATE

    ratio = _as_float(data.get("ratio"), 0.0, minimum=0.0)
    if ratio > 1.0:
        logger.warning(
            "[VisionEnhance] ratio %s is greater than 1; clamping to 1.",
            ratio,
        )
        ratio = 1.0

    return AnimationConfig(
        enable=_as_bool(data.get("enable"), True),
        min_frames=min_frames,
        max_frames=max_frames,
        interval=_as_int(data.get("interval"), 0, minimum=0),
        ratio=ratio,
        skip_first=_as_bool(data.get("skip_first"), True),
        output_mode=output_mode,
        grid_columns=_as_int(data.get("grid_columns"), 0, minimum=0),
        grid_label_frames=_as_bool(data.get("grid_label_frames"), True),
        max_edge=_as_int(data.get("max_edge"), 768, minimum=1),
        keep_original=_as_bool(data.get("keep_original"), False),
        keep_original_copy=_as_bool(data.get("keep_original_copy"), False),
        keep_original_max_files=_as_int(
            data.get("keep_original_max_files"), 200, minimum=1
        ),
        keep_original_max_mb=_as_int(data.get("keep_original_max_mb"), 200, minimum=1),
        scan_quoted=_as_bool(data.get("scan_quoted"), True),
    )


def load_config(raw: object) -> PluginConfig:
    """Normalize the raw plugin config into typed settings.

    Args:
        raw: Config object passed to the plugin constructor, normally the dict
            generated from ``_conf_schema.json``.

    Returns:
        Settings with every field validated against its expected type.
    """
    data = _as_dict(raw)
    return PluginConfig(
        enabled=_as_bool(data.get("enabled"), True),
        markdown_image=_load_markdown_image(data.get("markdown_image")),
        animation=_load_animation(data.get("animation")),
        max_images_per_message=_as_int(
            data.get("max_images_per_message"), 10, minimum=1
        ),
    )
