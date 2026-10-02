"""Split animated images into still frames.

Vision models generally cannot read animated images: they either take the first
frame or reject the request outright. This module samples frames from GIF, WebP
and APNG files and returns them as ordinary still images.

Every Pillow operation here decodes frames, which is CPU bound and can take a
noticeable amount of time for large animations. Callers must use
:func:`split_animation`, which keeps the work off the event loop.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path

from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
from astrbot.core.utils.datetime_utils import generate_timestamp_id
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from .config import DEFAULT_OUTPUT_MODE, OUTPUT_MODES, AnimationConfig

# Assembled grids are scaled down past this edge. Per-frame scaling already
# respects ``max_edge``, so this only guards against very wide multi-column
# layouts producing an unreasonably large image.
_GRID_MAX_EDGE = 2048

# Label strip height as a fraction of the frame height.
_LABEL_HEIGHT_RATIO = 0.12
_LABEL_MIN_HEIGHT = 14
_LABEL_MAX_HEIGHT = 48


@dataclass(slots=True)
class SplitResult:
    """Outcome of splitting one image.

    Attributes:
        animated: Whether the source was detected as animated.
        paths: Local paths of the produced images. Empty when nothing was
            produced, for example because the source was not animated.
        frame_indices: Source frame index for each produced image. When a grid
            is produced this lists every frame included in the grid.
        total_frames: Total frame count reported by the source.
        is_grid: Whether the output is a single assembled grid image.
        duration: Total playback time in seconds. Zero when the frame delays
            are unavailable or report no duration.
    """

    animated: bool = False
    paths: list[str] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    total_frames: int = 0
    is_grid: bool = False
    duration: float = 0.0


def is_animated(path: str) -> bool:
    """Check whether an image file is animated.

    Args:
        path: Local image path.

    Returns:
        True when the file holds more than one frame. Returns False for
        missing, unreadable, or single-frame files.
    """
    try:
        with PILImage.open(path) as image:
            return bool(_frame_count(image) > 1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[VisionEnhance] Not an animated image %s: %s", path, exc)
        return False


def _frame_count(image: PILImage.Image) -> int:
    """Return the frame count of an opened image.

    Args:
        image: An image opened by Pillow.

    Returns:
        The frame count, at least 1.
    """
    if getattr(image, "is_animated", False):
        return int(getattr(image, "n_frames", 1))
    return int(getattr(image, "n_frames", 1))


def _total_duration_ms(image: PILImage.Image) -> float:
    """Sum the frame delays of an opened image.

    Pillow exposes ``info["duration"]`` as the delay of the *current* frame, so
    the value has to be read after seeking each frame. Reading it once on the
    freshly opened image only yields the first frame's delay.

    Args:
        image: An image opened by Pillow.

    Returns:
        Total playback time in milliseconds. Zero when Pillow cannot report the
        delays, for example when there is no duration metadata at all.
    """
    if not getattr(image, "is_animated", False):
        return 0.0

    total = 0.0
    for index in range(int(getattr(image, "n_frames", 1))):
        try:
            image.seek(index)
            total += float(image.info.get("duration", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[VisionEnhance] Cannot read the delay of frame %d: %s", index, exc
            )
            break
    return total


def build_hint_values(
    result: SplitResult,
    cfg: AnimationConfig,
    original_path: str,
) -> dict[str, str]:
    """Build the placeholder values for the animation hint.

    Args:
        result: Outcome of the split that just happened.
        cfg: Animation settings, used for the effective output mode.
        original_path: Local path of the source image. Only exposed when the
            original path hint is enabled.

    Returns:
        Placeholder values keyed by name. ``original_path`` is empty when the
        path should not be revealed.
    """
    frames = ", ".join(str(index + 1) for index in result.frame_indices)
    duration = f"{result.duration:.2f}" if result.duration > 0 else ""
    mode = cfg.output_mode if cfg.output_mode in OUTPUT_MODES else DEFAULT_OUTPUT_MODE
    return {
        "total_frames": str(result.total_frames),
        "frame_count": str(len(result.frame_indices)),
        "frame_indices": frames,
        "output_mode": mode,
        "duration": duration,
        "original_path": original_path if cfg.keep_original else "",
    }


def select_frame_indices(total_frames: int, cfg: AnimationConfig) -> list[int]:
    """Choose which frames to extract.

    Sampling rules, in order of precedence:

    1. ``interval > 0``: take every Nth frame.
    2. ``0 < ratio < 1``: take that fraction of the frames, spread evenly.
    3. Otherwise: take up to ``max_frames`` frames, spread evenly.

    The result is then clamped to ``[min_frames, max_frames]`` and to the
    available range. ``skip_first`` drops frame 0 before anything else, but is
    ignored when it would leave no frames at all.

    Args:
        total_frames: Frame count reported by the source image.
        cfg: Animation settings.

    Returns:
        Sorted, de-duplicated frame indices.
    """
    if total_frames <= 0:
        return []

    start = 1 if cfg.skip_first and total_frames > 1 else 0
    available = total_frames - start
    if available <= 0:
        return []

    # Normalized config already guarantees min_frames <= max_frames, but this
    # function is also reachable with hand-built settings, so clamp again here.
    min_frames = min(cfg.min_frames, cfg.max_frames)

    if cfg.interval > 0:
        indices = list(range(start, total_frames, cfg.interval))
    elif cfg.ratio > 0:
        target = max(int(round(total_frames * cfg.ratio)), 1)
        indices = _spread_indices(start, total_frames, target)
    else:
        indices = _spread_indices(start, total_frames, cfg.max_frames)

    if len(indices) < min_frames:
        indices = _spread_indices(start, total_frames, min_frames)
    # The upper bound is applied last so that it always wins.
    if len(indices) > cfg.max_frames:
        indices = _spread_from(indices, cfg.max_frames)

    return sorted({index for index in indices if 0 <= index < total_frames})


def _spread_indices(start: int, stop: int, count: int) -> list[int]:
    """Pick ``count`` frames spread evenly across ``[start, stop)``.

    Args:
        start: First frame index that may be selected.
        stop: Exclusive upper bound.
        count: Desired number of frames.

    Returns:
        Frame indices. Returns every available frame when ``count`` is not
        smaller than the available range.
    """
    available = stop - start
    if available <= 0 or count <= 0:
        return []
    if count >= available:
        return list(range(start, stop))
    if count == 1:
        return [start]
    span = available - 1
    return sorted({start + round(index * span / (count - 1)) for index in range(count)})


def _spread_from(indices: list[int], count: int) -> list[int]:
    """Reduce an index list to ``count`` evenly spread entries.

    Args:
        indices: Candidate frame indices, in ascending order.
        count: Desired number of entries.

    Returns:
        The reduced index list, or the input when it is already short enough.
    """
    if count <= 0 or len(indices) <= count:
        return indices
    if count == 1:
        return [indices[0]]
    span = len(indices) - 1
    positions = sorted({round(index * span / (count - 1)) for index in range(count)})
    return [indices[position] for position in positions]


async def split_animation(
    path: str,
    cfg: AnimationConfig,
) -> SplitResult:
    """Split an animated image into still images.

    Args:
        path: Local path of the source image.
        cfg: Animation settings.

    Returns:
        A :class:`SplitResult`. Produced files land in AstrBot's shared temp
        directory so that the framework's own cleaner eventually removes them.
        Failures are reported as a result with ``animated = True`` and no
        paths, so the caller can fall back to the original image.
    """
    try:
        return await asyncio.to_thread(_split_sync, path, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[VisionEnhance] Failed to split animation %s: %s", path, exc, exc_info=True
        )
        return SplitResult(paths=[])


def _split_sync(path: str, cfg: AnimationConfig) -> SplitResult:
    """Run frame extraction synchronously.

    Args:
        path: Local path of the source image.
        cfg: Animation settings.

    Returns:
        The extraction result.
    """
    temp_dir = Path(get_astrbot_temp_path())
    with PILImage.open(path) as image:
        total_frames = _frame_count(image)
        if total_frames <= 1:
            return SplitResult(animated=False, total_frames=total_frames)

        # Summing delays seeks every frame, so it is only worth doing when the
        # configured template actually uses the duration.
        duration = (
            _total_duration_ms(image) / 1000 if "{duration}" in cfg.hint_text else 0.0
        )
        indices = select_frame_indices(total_frames, cfg)
        if not indices:
            return SplitResult(
                animated=True, total_frames=total_frames, duration=duration
            )

        frames = _extract_frames(image, indices, cfg.max_edge)
        if not frames:
            return SplitResult(
                animated=True, total_frames=total_frames, duration=duration
            )

        try:
            if cfg.output_mode == "grid":
                grid = _build_grid(frames, cfg)
                saved = _save_images([grid], temp_dir, "grid")
                return SplitResult(
                    animated=True,
                    paths=saved,
                    frame_indices=indices,
                    total_frames=total_frames,
                    is_grid=True,
                    duration=duration,
                )
            saved = _save_images(frames, temp_dir, "frame")
            return SplitResult(
                animated=True,
                paths=saved,
                frame_indices=indices,
                total_frames=total_frames,
                duration=duration,
            )
        finally:
            for frame in frames:
                frame.close()


def _extract_frames(
    image: PILImage.Image,
    indices: list[int],
    max_edge: int,
) -> list[PILImage.Image]:
    """Decode and scale the requested frames.

    Args:
        image: Opened animated image. Seeked in place.
        indices: Frame indices to decode, in ascending order.
        max_edge: Longest edge of each produced frame, in pixels.

    Returns:
        Detached RGB frames. Undecodable frames are skipped.
    """
    frames: list[PILImage.Image] = []
    for index in indices:
        try:
            image.seek(index)
            frame = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[VisionEnhance] Skipping undecodable frame %d: %s", index, exc
            )
            continue
        frames.append(_fit_within(frame, max_edge))
    return frames


def _fit_within(image: PILImage.Image, max_edge: int) -> PILImage.Image:
    """Scale an image down so its longest edge respects ``max_edge``.

    Args:
        image: Image to scale. Returned unchanged when it already fits.
        max_edge: Longest allowed edge, in pixels.

    Returns:
        The scaled image, either the input itself or a new image.
    """
    longest = max(image.size)
    if longest <= max_edge or longest <= 0:
        return image
    scale = max_edge / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, PILImage.Resampling.LANCZOS)
    image.close()
    return resized


def _build_grid(frames: list[PILImage.Image], cfg: AnimationConfig) -> PILImage.Image:
    """Assemble frames into a single sprite sheet.

    Args:
        frames: Frames to place, in display order.
        cfg: Animation settings providing column count and labelling.

    Returns:
        The assembled grid image.
    """
    count = len(frames)
    columns = cfg.grid_columns if cfg.grid_columns > 0 else math.ceil(math.sqrt(count))
    columns = max(1, min(columns, count))
    rows = math.ceil(count / columns)
    cell_width = max(frame.width for frame in frames)
    cell_height = max(frame.height for frame in frames)

    grid = PILImage.new("RGB", (cell_width * columns, cell_height * rows), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    font = _label_font(cell_height) if cfg.grid_label_frames else None

    for position, frame in enumerate(frames):
        column = position % columns
        row = position // columns
        origin = (column * cell_width, row * cell_height)
        grid.paste(frame, origin)
        if font is not None:
            _draw_label(draw, position + 1, origin, (cell_width, cell_height), font)

    return _cap_grid_size(grid)


def _cap_grid_size(grid: PILImage.Image) -> PILImage.Image:
    """Scale an assembled grid down when it exceeds the safety ceiling.

    Args:
        grid: Assembled grid image.

    Returns:
        The possibly scaled grid.
    """
    longest = max(grid.size)
    if longest <= _GRID_MAX_EDGE:
        return grid
    scale = _GRID_MAX_EDGE / longest
    size = (max(1, round(grid.width * scale)), max(1, round(grid.height * scale)))
    resized = grid.resize(size, PILImage.Resampling.LANCZOS)
    grid.close()
    return resized


def _label_font(cell_height: int) -> ImageFont.ImageFont:
    """Load a font for drawing frame labels.

    Args:
        cell_height: Height of one grid cell, used to size the font.

    Returns:
        A TrueType font when one is available, otherwise Pillow's bitmap font.
    """
    size = max(
        _LABEL_MIN_HEIGHT,
        min(_LABEL_MAX_HEIGHT, round(cell_height * _LABEL_HEIGHT_RATIO)),
    )
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: int,
    origin: tuple[int, int],
    cell: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    """Draw a frame number in the top-left corner of a grid cell.

    Args:
        draw: Draw context bound to the grid image.
        label: Frame number to draw.
        origin: Top-left corner of the cell.
        cell: Cell size in pixels.
        font: Font used for the label.
    """
    text = str(label)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    padding = max(2, text_height // 4)
    strip_height = min(text_height + padding * 2, cell[1])
    strip_width = min(text_width + padding * 2, cell[0])

    draw.rectangle(
        (
            origin[0],
            origin[1],
            origin[0] + strip_width,
            origin[1] + strip_height,
        ),
        fill=(0, 0, 0),
    )
    draw.text(
        (origin[0] + padding, origin[1] + padding - text_box[1]),
        text,
        fill=(255, 255, 255),
        font=font,
    )


def _save_images(
    images: list[PILImage.Image],
    temp_dir: Path,
    prefix: str,
) -> list[str]:
    """Write images to the temp directory as PNG files.

    Args:
        images: Images to save.
        temp_dir: Destination directory.
        prefix: Filename prefix describing the output kind.

    Returns:
        Paths of the saved files. Images that fail to save are skipped.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for image in images:
        path = temp_dir / f"vision_{prefix}_{generate_timestamp_id()}.png"
        try:
            image.save(path, "PNG")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[VisionEnhance] Failed to save frame %s: %s", path, exc)
            continue
        saved.append(str(path))
    return saved
