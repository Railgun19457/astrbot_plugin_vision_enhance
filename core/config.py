"""Load and normalize the plugin configuration.

AstrBot generates the user-facing config from ``_conf_schema.json`` and passes it
to ``Star.__init__``. The fallbacks below only matter when a key is missing or has
the wrong type, which can happen after a plugin update adds new settings. Keep the
defaults here in sync with ``_conf_schema.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger

OUTPUT_MODE_SEPARATE = "separate"
OUTPUT_MODE_GRID = "grid"
OUTPUT_MODES = frozenset({OUTPUT_MODE_SEPARATE, OUTPUT_MODE_GRID})

# Placeholders available in the animation hint template. Each name maps to
# ``{description, sample}``, where ``sample`` shows the shape of the values that
# replace it so the fallback matches what users will actually see.
ANIMATION_HINT_PLACEHOLDERS: dict[str, dict[str, str]] = {
    "total_frames": {
        "description": "动图的总帧数",
        "sample": "24",
    },
    "frame_count": {
        "description": "本次抽取的帧数量",
        "sample": "4",
    },
    "frame_indices": {
        "description": "抽出的帧序号，从 1 开始，逗号分隔",
        "sample": "1, 7, 13, 19",
    },
    "output_mode": {
        "description": "输出方式，拼接大图时为 grid，其余为 separate",
        "sample": "separate",
    },
    "duration": {
        "description": "动图总时长（秒，保留两位小数），时长未知时为空",
        "sample": "2.40",
    },
    "original_path": {
        "description": "原图在本地的路径，仅在开启「附带原图路径」时有值",
        "sample": r"E:\\Code\\AstrBot\\data\\temp\\media_image_1.png",
    },
}

# Every placeholder, wrapped in braces, is replaced by the sample value to build
# the fallback template. Keeping one source avoids the two drifting apart.
DEFAULT_ANIMATION_HINT_TEXT = (
    "[Animated image: {total_frames} frames in total, "
    "{frame_count} extracted (frames {frame_indices}).]"
)

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


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
    hint_enable: bool = True
    hint_text: str = DEFAULT_ANIMATION_HINT_TEXT
    scan_quoted: bool = True


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
        hint_enable=_as_bool(data.get("hint_enable"), True),
        hint_text=_as_str(data.get("hint_text"), DEFAULT_ANIMATION_HINT_TEXT),
        scan_quoted=_as_bool(data.get("scan_quoted"), True),
    )


# Characters that carry no information on their own. A rendered hint made up
# only of these is treated as empty, so a template such as ``[{original_path}]``
# does not leak a bare ``[]`` into the conversation when the path is unset.
_DECORATION_CHARS = " \t\r\n[](){}<>【】（）《》「」『』“”\"'.,;:，。；：、-—_"


def render_animation_hint(template: str, values: dict[str, str]) -> str:
    """Fill the animation hint template with frame information.

    Unknown placeholders are replaced with an empty string so that a typo or a
    placeholder belonging to a newer version degrades to missing information
    rather than leaking ``{name}`` into the conversation.

    Args:
        template: User-configured template, possibly empty.
        values: Placeholder values keyed by name.

    Returns:
        The rendered text, stripped. Empty when the template is empty or when
        nothing but surrounding punctuation is left.
    """

    def _replace(match: re.Match[str]) -> str:
        return values.get(match.group(1), "")

    rendered = _PLACEHOLDER_RE.sub(_replace, template).strip()
    if not rendered.strip(_DECORATION_CHARS):
        return ""
    return rendered


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
