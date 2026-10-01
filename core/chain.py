"""Message chain rewrite primitives.

Rewriting the message chain is not enough on its own: the LLM prompt comes from
``event.message_str`` while the image attachments come from
``event.message_obj.message``. Leaving the two out of sync makes the model see
both the original Markdown text and the restored image. Every rewrite therefore
has to update the plain text as well, which is what :class:`Splice` exists for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astrbot.api.message_components import BaseMessageComponent, Image, Plain, Reply


@dataclass
class Splice:
    """A single replacement performed inside one Plain component.

    Attributes:
        start: Inclusive start offset of the matched text in the source Plain.
        end: Exclusive end offset of the matched text in the source Plain.
        matched: The exact matched text, used to remove it from ``message_str``.
        text: Text that replaces the match. Kept as a Plain component when
            non-empty.
        inserts: Components inserted at the match position.
    """

    start: int
    end: int
    matched: str
    text: str = ""
    inserts: list[BaseMessageComponent] = field(default_factory=list)


def splice_plain(component: Plain, splices: list[Splice]) -> list[BaseMessageComponent]:
    """Split a Plain component around the given replacements.

    Args:
        component: The Plain component to rewrite.
        splices: Replacements to apply. They are expected to be sorted by
            ``start`` and not to overlap. Invalid spans are skipped.

    Returns:
        The replacement component list. Returns a single-element list holding
        the original component when nothing applies.
    """
    text = component.text
    valid = [
        splice
        for splice in splices
        if 0 <= splice.start < splice.end <= len(text)
        and text[splice.start : splice.end] == splice.matched
    ]
    if not valid:
        return [component]

    # Text is accumulated and only flushed when an insert breaks the run, so a
    # replacement never splits the surrounding text into extra Plain fragments.
    parts: list[BaseMessageComponent] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        merged = "".join(buffer)
        buffer.clear()
        if merged:
            parts.append(Plain(merged))

    cursor = 0
    for splice in valid:
        if splice.start > cursor:
            buffer.append(text[cursor : splice.start])
        flush()
        parts.extend(splice.inserts)
        if splice.text:
            buffer.append(splice.text)
        cursor = splice.end
    if cursor < len(text):
        buffer.append(text[cursor:])
    flush()
    return parts or [component]


def sync_message_str(message_str: str, splices: list[Splice]) -> tuple[str, int]:
    """Remove rewritten text from ``message_str``.

    Args:
        message_str: The event's plain text representation.
        splices: Replacements that were applied to the chain.

    Returns:
        A tuple of the updated text and how many replacements matched. A
        replacement that cannot be located is skipped so that a partially
        rewritten chain still removes as much text as possible.
    """
    updated = message_str
    applied = 0
    for splice in splices:
        if not splice.matched:
            continue
        replacement = splice.text
        if splice.matched not in updated:
            continue
        updated = updated.replace(splice.matched, replacement, 1)
        applied += 1
    return updated, applied


def iter_message_chains(
    components: list[BaseMessageComponent],
    *,
    include_quoted: bool,
) -> list[list[BaseMessageComponent]]:
    """Collect the component lists that make up a message.

    Args:
        components: Top-level message chain.
        include_quoted: Whether reply chains should be included as well.

    Returns:
        The top-level chain first, followed by each quoted message chain. All
        returned lists are mutable and belong to the event.
    """
    chains: list[list[BaseMessageComponent]] = [components]
    if not include_quoted:
        return chains
    for component in components:
        if isinstance(component, Reply) and component.chain:
            chains.append(component.chain)
    return chains


def count_images(components: list[BaseMessageComponent]) -> int:
    """Count image components in a chain.

    Args:
        components: Components to inspect.

    Returns:
        The number of image components, including those inside reply chains.
    """
    total = 0
    for component in components:
        if isinstance(component, Image):
            total += 1
        elif isinstance(component, Reply) and component.chain:
            total += count_images(component.chain)
    return total


def image_component(path: str) -> Image:
    """Build an Image component pointing at a local file.

    ``Image.convert_to_file_path`` prefers ``url`` over ``file``, while
    ``Image.fromFileSystem`` only sets ``file``. Setting all three fields to the
    local path matches what the framework's own preprocess stage does for
    received images and avoids surprises in either lookup order.

    Args:
        path: Local image path.

    Returns:
        The image component.
    """
    return Image(file=path, path=path, url=path)
