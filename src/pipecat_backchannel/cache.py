"""On-disk clip cache.

Clips are produced at most once, then read from disk forever after. Nothing here
runs on the hot path: the cache is filled at pipeline start, and the player only
ever plays bytes that are already in memory. That's the whole point — no LLM, no
per-utterance network round trip, no touching the live TTS service's state
machinery to say "mhm".

The :class:`ClipCache` protocol is two methods on purpose. Storage is the least
interesting decision in this library, and the consumer only ever needs "do you
have this?" and "keep this".
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable


def clip_filename(text: str, sample_rate: int) -> str:
    """Return the cache filename for a clip.

    Exported so you can pre-record clips by hand: name your ``.wav`` files with
    this, drop them in the cache directory, and they will be used without ever
    calling a synthesizer or recording anything.

    The name carries a short fingerprint of the exact text, because the readable
    part alone cannot: ``"Hm."``, ``"Hm,"`` and ``"Hm..."`` are three different
    prosodies — the whole point of the inventory — and all collapse to ``hm``.

    Args:
        text: The clip text, e.g. ``"Mhm."``.
        sample_rate: Sample rate in Hz.

    Returns:
        A filename such as ``"mhm_5c9e6a_24000hz.wav"``.
    """
    safe = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
    tag = hashlib.sha1(text.encode()).hexdigest()[:6]
    return f"{safe}_{tag}_{sample_rate}hz.wav"


@runtime_checkable
class ClipCache(Protocol):
    """Keeps clip audio between runs."""

    def get(self, text: str, sample_rate: int) -> bytes | None:
        """Return cached PCM for a clip, or ``None`` if it isn't stored.

        Args:
            text: The clip text.
            sample_rate: Sample rate in Hz. Part of the key — clips stored at one
                rate must never be handed back for another.

        Returns:
            Raw mono 16-bit little-endian PCM, or ``None``.
        """
        ...

    def put(self, text: str, sample_rate: int, pcm: bytes) -> None:
        """Store PCM for a clip.

        Args:
            text: The clip text.
            sample_rate: Sample rate in Hz.
            pcm: Raw mono 16-bit little-endian PCM.
        """
        ...


class FileClipCache:
    """Stores clips as ``.wav`` files in a directory.

    Delete the directory to force everything to be produced again, e.g. after
    changing voice.
    """

    def __init__(self, directory: str | Path = ".clip_cache"):
        """Initialize the cache.

        Args:
            directory: Where the ``.wav`` files live. Created if missing.
        """
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        """The directory holding the cached ``.wav`` files."""
        return self._directory

    def path_for(self, text: str, sample_rate: int) -> Path:
        """Return the full path this cache would use for a clip."""
        return self._directory / clip_filename(text, sample_rate)

    def get(self, text: str, sample_rate: int) -> bytes | None:
        """Read a clip from disk. See :class:`ClipCache`."""
        path = self.path_for(text, sample_rate)
        if not path.exists():
            return None
        with wave.open(str(path), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())
        return pcm

    def put(self, text: str, sample_rate: int, pcm: bytes) -> None:
        """Write a clip to disk. See :class:`ClipCache`."""
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(text, sample_rate)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
