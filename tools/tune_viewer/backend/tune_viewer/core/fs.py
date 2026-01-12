from __future__ import annotations

from collections import deque
from pathlib import Path


class PathTraversalError(ValueError):
    pass


def resolve_under_root(*, root: Path, rel_path: str) -> Path:
    """
    Resolve a *relative* path under a trusted root directory, rejecting traversal.
    """
    root_abs = root.resolve()
    rel = Path(str(rel_path).strip())
    if rel.is_absolute():
        raise PathTraversalError("Absolute paths are not allowed.")

    target = (root_abs / rel).resolve()
    try:
        if not target.is_relative_to(root_abs):
            raise PathTraversalError("Path traversal detected.")
    except AttributeError:
        # Python < 3.9 fallback (shouldn't happen in this repo, but keep safe).
        if str(root_abs) not in str(target):
            raise PathTraversalError("Path traversal detected.")

    return target


def read_text_capped(path: Path, *, max_bytes: int) -> str:
    data = path.read_bytes()
    if len(data) > int(max_bytes):
        data = data[: int(max_bytes)]
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("utf-8", errors="replace")


def tail_text_lines(path: Path, *, tail_lines: int, max_bytes: int) -> str:
    """
    Read last N lines (best-effort) with a total byte cap.
    """
    # Byte cap first to avoid pathological log sizes.
    # Read from end in chunks.
    chunk_size = 64 * 1024
    tail_lines = max(1, int(tail_lines))
    max_bytes = max(1, int(max_bytes))

    with path.open("rb") as f:
        f.seek(0, 2)
        end = f.tell()
        remaining = min(end, max_bytes)
        pos = end

        chunks: list[bytes] = []
        while remaining > 0:
            n = min(chunk_size, remaining)
            pos -= n
            f.seek(pos)
            chunks.append(f.read(n))
            remaining -= n
            # Stop early if we already have plenty of newlines.
            if b"\n" in chunks[-1] and b"\n".join(chunks).count(b"\n") > tail_lines * 2:
                break

    data = b"".join(reversed(chunks))
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("utf-8", errors="replace")

    lines = deque(text.splitlines(), maxlen=tail_lines)
    return "\n".join(lines)

