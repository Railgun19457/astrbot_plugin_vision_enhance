"""Inbound vision enhancement for AstrBot.

Restores Markdown image syntax into real image messages and splits animated
images into still frames so that vision models can actually see them.
"""

from .config import (
    ANIMATION_HINT_PLACEHOLDERS,
    DEFAULT_ANIMATION_HINT_TEXT,
    AnimationConfig,
    MarkdownImageConfig,
    PluginConfig,
    load_config,
)

__all__ = [
    "ANIMATION_HINT_PLACEHOLDERS",
    "DEFAULT_ANIMATION_HINT_TEXT",
    "AnimationConfig",
    "MarkdownImageConfig",
    "PluginConfig",
    "load_config",
]
