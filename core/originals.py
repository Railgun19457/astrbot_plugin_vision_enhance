"""Hand the model a usable path to the original image.

AstrBot already stores received images on disk, so the plugin never needs to
duplicate them just to expose a path. Those files are short lived though: the
framework deletes media registered during a message as soon as the pipeline
finishes, and its temp cleaner removes leftovers on a timer. So:

* ``keep_original`` reuses the existing file and tells the model where it is.
  This costs no extra disk space and is valid for the current request only.
* ``keep_original_copy`` copies the file into the plugin data directory, which
  the framework does not prune, so later turns can still read it. That copy
  needs its own limits, applied by :func:`prune_store`.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from astrbot.api import logger
from astrbot.core.utils.datetime_utils import generate_timestamp_id

ORIGINALS_DIRNAME = "originals"


def originals_dir(data_dir: Path) -> Path:
    """Return the directory holding persistent original images.

    Args:
        data_dir: The plugin's data directory.

    Returns:
        Path to the store, which may not exist yet.
    """
    return data_dir / ORIGINALS_DIRNAME


async def copy_into_store(source: str, data_dir: Path) -> str | None:
    """Copy an image into the persistent store.

    Args:
        source: Local path of the image to preserve.
        data_dir: The plugin's data directory.

    Returns:
        The stored path, or None when the source is missing or the copy fails.
    """
    try:
        return await asyncio.to_thread(_copy_sync, source, data_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[VisionEnhance] Failed to preserve original image %s: %s", source, exc
        )
        return None


def _copy_sync(source: str, data_dir: Path) -> str | None:
    """Perform the copy and prune the store.

    Args:
        source: Local path of the image to preserve.
        data_dir: The plugin's data directory.

    Returns:
        The stored path, or None when the source does not exist.
    """
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        return None

    target_dir = originals_dir(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = source_path.suffix or ".img"
    target = target_dir / f"{source_path.stem[:32]}_{generate_timestamp_id()}{suffix}"
    shutil.copyfile(source_path, target)
    return str(target)


def prune_store(
    data_dir: Path,
    max_files: int,
    max_mb: int,
) -> int:
    """Enforce the store limits by deleting the oldest files first.

    Args:
        data_dir: The plugin's data directory.
        max_files: Maximum number of files to keep.
        max_mb: Maximum total size in megabytes.

    Returns:
        The number of deleted files.
    """
    try:
        return _prune_sync(data_dir, max_files, max_mb)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[VisionEnhance] Failed to prune original images: %s", exc)
        return 0


def _prune_sync(data_dir: Path, max_files: int, max_mb: int) -> int:
    """Collect store entries and delete the oldest until both limits hold.

    Args:
        data_dir: The plugin's data directory.
        max_files: Maximum number of files to keep.
        max_mb: Maximum total size in megabytes.

    Returns:
        The number of deleted files.
    """
    target_dir = originals_dir(data_dir)
    if not target_dir.is_dir():
        return 0

    entries: list[tuple[float, int, Path]] = []
    total_bytes = 0
    for path in target_dir.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
        total_bytes += stat.st_size

    max_bytes = max(max_mb, 1) * 1024**2
    keep_limit = max(max_files, 1)
    if len(entries) <= keep_limit and total_bytes <= max_bytes:
        return 0

    entries.sort(key=lambda item: item[0])
    removable = max(len(entries) - keep_limit, 0)
    deleted = 0
    for _mtime, size, path in entries:
        if deleted >= removable and total_bytes <= max_bytes:
            break
        try:
            path.unlink()
        except OSError as exc:
            logger.warning(
                "[VisionEnhance] Failed to delete stored original %s: %s", path, exc
            )
            continue
        total_bytes -= size
        deleted += 1
    return deleted
