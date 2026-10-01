"""Restore Markdown image syntax into real image messages.

QQ's official bot API delivers some images as Markdown text rather than as
image segments, for example::

    ![图片 #380px #330px](https://qqbot.ugcimg.cn/102076836/.../ee184624...)

The model then only sees a URL. Rewriting the text is not enough on its own,
because the LLM prompt is built from ``event.message_str`` while attachments
come from the message chain, so both have to be updated together.

Guessing the image type from the file extension does not work here: the URLs
above carry no extension at all. Type detection therefore relies on actually
decoding the downloaded bytes.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse

from astrbot.api import logger
from astrbot.core.utils.media_utils import MediaResolver
from PIL import Image as PILImage

from .chain import Splice, image_component
from .config import MarkdownImageConfig

# Matches ![alt](url) and ![alt](url "title"). Only http(s) URLs are accepted so
# that local paths and unrelated bracket syntax are left alone.
MD_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\("
    r"\s*(?P<url>https?://[^\s)>\"']+)"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)


def find_markdown_images(text: str, cfg: MarkdownImageConfig) -> list[re.Match[str]]:
    """Find Markdown image matches that pass the domain whitelist.

    Args:
        text: Plain text to scan.
        cfg: Markdown image settings.

    Returns:
        Matches in source order.
    """
    matches: list[re.Match[str]] = []
    for match in MD_IMAGE_RE.finditer(text):
        url = match.group("url")
        if cfg.url_whitelist and not _host_allowed(url, cfg.url_whitelist):
            logger.debug("[VisionEnhance] Skipping non-whitelisted image %s", url)
            continue
        matches.append(match)
    return matches


def _host_allowed(url: str, whitelist: list[str]) -> bool:
    """Check whether the URL host matches an entry in the whitelist.

    Args:
        url: Image URL.
        whitelist: Allowed hosts. Entries match the host exactly or as a suffix
            separated by a dot, so ``ugcimg.cn`` also covers
            ``qqbot.ugcimg.cn``.

    Returns:
        True when the host is allowed.
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    for entry in whitelist:
        candidate = entry.lower().lstrip(".").removeprefix("*.")
        if not candidate:
            continue
        if host == candidate or host.endswith(f".{candidate}"):
            return True
    return False


async def download_image(url: str, cfg: MarkdownImageConfig) -> str | None:
    """Download an image and verify that it really decodes as an image.

    Args:
        url: Image URL.
        cfg: Markdown image settings providing the size limit.

    Returns:
        The local path of the decoded image, or None when the download failed,
        exceeded the size limit, or did not contain a decodable image.
    """
    try:
        path = await asyncio.wait_for(
            MediaResolver(url, media_type="image").to_path(),
            timeout=cfg.download_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[VisionEnhance] Timed out after %ss downloading %s",
            cfg.download_timeout,
            url,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[VisionEnhance] Failed to download %s: %s", url, exc)
        return None

    if not await _is_decodable_image(path, cfg.max_download_bytes):
        return None
    return path


async def _is_decodable_image(path: str, max_bytes: int) -> bool:
    """Verify the file is a readable image within the size limit.

    Args:
        path: Local file path.
        max_bytes: Maximum accepted file size in bytes.

    Returns:
        True when the file decodes as an image and fits the size limit.
    """
    try:
        size = await asyncio.to_thread(lambda: Path(path).stat().st_size)
    except OSError as exc:
        logger.warning("[VisionEnhance] Cannot stat downloaded image %s: %s", path, exc)
        return False

    if size > max_bytes:
        logger.info(
            "[VisionEnhance] Skipping %s because it is %.1f MB (limit %.1f MB)",
            path,
            size / 1024**2,
            max_bytes / 1024**2,
        )
        return False

    def _probe() -> bool:
        try:
            with PILImage.open(path) as image:
                image.verify()
            return True
        except Exception:  # noqa: BLE001
            return False

    if not await asyncio.to_thread(_probe):
        logger.info(
            "[VisionEnhance] Downloaded file is not a decodable image: %s",
            Path(path).name,
        )
        return False
    return True


def build_placeholder(alt: str, cfg: MarkdownImageConfig) -> str:
    """Build the text left behind at the match position.

    Args:
        alt: The Markdown alt text, which often carries size hints.
        cfg: Markdown image settings.

    Returns:
        The replacement text. Empty when the placeholder is disabled and the
        alt text is not kept.
    """
    parts: list[str] = []
    if cfg.keep_alt_text:
        cleaned = alt.strip()
        if cleaned:
            parts.append(cleaned)
    if cfg.placeholder:
        parts.append(cfg.placeholder)
    return " ".join(parts)


def make_splice(match: re.Match[str], path: str, cfg: MarkdownImageConfig) -> Splice:
    """Create the chain replacement for one Markdown image.

    Args:
        match: Match produced by :data:`MD_IMAGE_RE`.
        path: Local path of the downloaded image.
        cfg: Markdown image settings.

    Returns:
        A splice that removes the Markdown text and inserts the image.
    """
    return Splice(
        start=match.start(),
        end=match.end(),
        matched=match.group(0),
        text=build_placeholder(match.group("alt"), cfg),
        inserts=[image_component(path)],
    )
