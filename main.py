"""Restore Markdown images and split animations before the model sees them.

Both features rewrite ``event.message_obj.message`` rather than the LLM request,
so the improvements are visible to every other plugin as well as to the model.

Two framework details drive the structure:

* Handler filters cannot express "only when the bot was addressed". AstrBot
  marks an event as woken as soon as any handler filter passes, so without an
  explicit check a group message that merely happens to contain a Markdown
  image would reach the LLM.
* Because that check lives in the handler rather than in a filter, the plugin
  also never depends on the wake prefix.
"""

from __future__ import annotations

import asyncio

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import BaseMessageComponent, Image, Plain
from astrbot.api.star import Context, Star

from .core import chain as chain_ops
from .core import markdown_image
from .core.animation import is_animated, split_animation
from .core.config import PluginConfig, load_config

# Run before other plugins so that they observe the restored images. Higher
# numbers execute first.
HANDLER_PRIORITY = 100


class VisionEnhancePlugin(Star):
    """Inbound vision enhancement for images the model could not otherwise read."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.context = context
        self.raw_config = config if config is not None else {}
        self.config: PluginConfig = load_config(self.raw_config)

    async def terminate(self) -> None:
        """Release anything owned by the plugin when it is unloaded."""
        logger.debug("[VisionEnhance] Plugin unloaded.")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=HANDLER_PRIORITY)
    async def enhance_inbound_images(self, event: AstrMessageEvent):
        """Rewrite inbound images so that the model can actually read them."""
        if not self.config.enabled or not self.should_handle(event):
            return

        try:
            await self._restore_markdown_images(event)
            await self._split_animations(event)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[VisionEnhance] Failed to enhance inbound images for %s",
                event.unified_msg_origin,
            )

    @staticmethod
    def should_handle(event: AstrMessageEvent) -> bool:
        """Decide whether the event is worth processing.

        Args:
            event: Incoming message event.

        Returns:
            True when the bot was addressed, or the message is a private chat.
        """
        return event.is_at_or_wake_command or event.is_private_chat()

    async def _restore_markdown_images(self, event: AstrMessageEvent) -> None:
        """Replace Markdown image syntax with real image components.

        Args:
            event: Event whose message chain is rewritten in place.
        """
        cfg = self.config.markdown_image
        if not cfg.enable:
            return

        budget = self.config.max_images_per_message
        for components in chain_ops.iter_message_chains(
            event.get_messages(), include_quoted=cfg.scan_quoted
        ):
            if budget <= 0:
                break
            budget = await self._restore_chain(event, components, budget)

    async def _restore_chain(
        self,
        event: AstrMessageEvent,
        components: list[BaseMessageComponent],
        budget: int,
    ) -> int:
        """Rewrite one component list.

        Args:
            event: Event that owns the chain.
            components: Component list to rewrite in place.
            budget: Remaining number of images allowed in this message.

        Returns:
            The image budget left after the rewrite.
        """
        cfg = self.config.markdown_image
        for index, component in enumerate(list(components)):
            if budget <= 0:
                break
            if not isinstance(component, Plain):
                continue

            splices = []
            for match in markdown_image.find_markdown_images(component.text, cfg):
                if budget <= 0:
                    break
                path = await markdown_image.download_image(match.group("url"), cfg)
                if not path:
                    continue
                self._track(event, path)
                splices.append(markdown_image.make_splice(match, path, cfg))
                budget -= 1
            if not splices:
                continue

            event.message_str, _ = chain_ops.sync_message_str(
                event.message_str, splices
            )
            components[index : index + 1] = chain_ops.splice_plain(component, splices)
        return budget

    async def _split_animations(self, event: AstrMessageEvent) -> None:
        """Replace animated images with extracted still frames.

        Args:
            event: Event whose message chain is rewritten in place.
        """
        cfg = self.config.animation
        if not cfg.enable:
            return

        budget = self.config.max_images_per_message
        for components in chain_ops.iter_message_chains(
            event.get_messages(), include_quoted=cfg.scan_quoted
        ):
            if budget <= 0:
                break
            budget = await self._split_chain(event, components, budget)

    async def _split_chain(
        self,
        event: AstrMessageEvent,
        components: list[BaseMessageComponent],
        budget: int,
    ) -> int:
        """Split animated images in one component list.

        Args:
            event: Event that owns the chain.
            components: Component list to rewrite in place.
            budget: Remaining number of images allowed in this message.

        Returns:
            The image budget left after the rewrite.
        """
        cfg = self.config.animation
        for index, component in enumerate(list(components)):
            if budget <= 0:
                break
            if not isinstance(component, Image):
                continue

            path = await self._resolve_image_path(event, component)
            if not path or not await asyncio.to_thread(is_animated, path):
                continue

            result = await split_animation(path, cfg)
            if not result.paths:
                logger.info(
                    "[VisionEnhance] Could not split a %d-frame animation; "
                    "keeping the original image.",
                    result.total_frames,
                )
                continue

            replacements = [
                chain_ops.image_component(saved) for saved in result.paths[:budget]
            ]
            for replacement in replacements:
                # Track only the generated frames. A persistent copy is owned by
                # the plugin and must survive this event.
                self._track(event, replacement.file)
            budget -= len(replacements)

            components[index : index + 1] = replacements
            await self._append_original_hint(
                event, components, path, result.frame_indices
            )
        return budget

    async def _resolve_image_path(
        self, event: AstrMessageEvent, component: Image
    ) -> str | None:
        """Materialize an image component into a local file path.

        Args:
            event: Event that owns the component, used to track temp files.
            component: Image component to resolve.

        Returns:
            The local path, or None when the image cannot be resolved.
        """
        try:
            path = await component.convert_to_file_path()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[VisionEnhance] Cannot resolve an image path for %s: %s",
                event.unified_msg_origin,
                exc,
            )
            return None
        self._track(event, path)
        return path

    async def _append_original_hint(
        self,
        event: AstrMessageEvent,
        components: list[BaseMessageComponent],
        original_path: str,
        frame_indices: list[int],
    ) -> None:
        """Tell the model where the untouched original image lives.

        AstrBot deletes received media once the pipeline finishes, so the
        reference stays valid for the current turn only.

        Args:
            event: Event whose chain receives the hint.
            components: Component list the hint is appended to.
            original_path: Local path of the source image.
            frame_indices: Source frame indices that were extracted.
        """
        cfg = self.config.animation
        if not cfg.keep_original or not components or not original_path:
            return

        frames = ", ".join(str(index) for index in frame_indices)
        hint = (
            f"[Animated image: frames {frames} extracted. The original file is "
            f"available at [Image Attachment: path {original_path}]]"
        )
        components.append(Plain(hint))
        event.message_str = f"{event.message_str} {hint}".strip()

    @staticmethod
    def _track(event: AstrMessageEvent, path: str | None) -> None:
        """Register a temporary file for cleanup when the event finishes.

        Args:
            event: Event that owns the file.
            path: Local file path, or None when the component has no file.
        """
        if path:
            event.track_temporary_local_file(path)
