"""The clip library: what clips exist, and what they sound like.

One object owns both halves, because they are one fact. Splitting them is how a
custom inventory silently drifts out of sync with the audio on disk — the
processor picks a clip that was never produced, and the bot goes quiet with no
error anywhere.

Nothing here needs a sample rate from the caller. The library learns it from the
pipeline's ``StartFrame`` and keys the cache on it, so clips can never be played
back at a rate they weren't produced at.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from loguru import logger

from pipecat_backchannel.cache import ClipCache, FileClipCache
from pipecat_backchannel.clips import DEFAULT_CLIP_GROUPS, flatten_clips
from pipecat_backchannel.synth import ClipSynthesizer


def _validate(groups: Mapping[str, Sequence[str]]) -> None:
    if not groups:
        raise ValueError("clip_groups is empty — there would be nothing to say.")
    for name, clips in groups.items():
        if not clips:
            raise ValueError(f"clip group {name!r} is empty — remove it or give it clips.")
        for clip in clips:
            if not clip or not clip.strip():
                raise ValueError(f"clip group {name!r} contains an empty clip text.")


class ClipLibrary:
    """The clips a backchannel can play, and the audio behind them.

    Filled at pipeline start, from whichever source is available: the cache
    first, then a :class:`~pipecat_backchannel.synth.ClipSynthesizer` if one was
    given. With neither, clips stay missing until something calls :meth:`store`
    — which is what :class:`~pipecat_backchannel.recorder.ClipRecorder` does,
    using the pipeline's own TTS service.
    """

    def __init__(
        self,
        *,
        groups: Mapping[str, Sequence[str]] | None = None,
        synthesizer: ClipSynthesizer | None = None,
        cache: ClipCache | None = None,
    ):
        """Initialize the library.

        Args:
            groups: Clip inventory grouped by conversational function. Defaults
                to :data:`~pipecat_backchannel.clips.DEFAULT_CLIP_GROUPS`.
            synthesizer: Produces clips the cache doesn't have. ``None`` means
                the clips are recorded from the pipeline's TTS instead.
            cache: Where clips are kept between runs. Defaults to
                :class:`~pipecat_backchannel.cache.FileClipCache`.
        """
        groups = DEFAULT_CLIP_GROUPS if groups is None else groups
        _validate(groups)
        self._groups = groups
        self._texts = flatten_clips(groups)
        self._synthesizer = synthesizer
        self._cache = cache or FileClipCache()

        self._clips: dict[str, bytes] = {}
        self._sample_rate: int | None = None
        self._lock = asyncio.Lock()

    @property
    def groups(self) -> Mapping[str, Sequence[str]]:
        """The clip inventory, grouped by conversational function."""
        return self._groups

    @property
    def texts(self) -> list[str]:
        """Every clip text, deduplicated."""
        return list(self._texts)

    @property
    def synthesizer(self) -> ClipSynthesizer | None:
        """What produces clips the cache doesn't have, if anything."""
        return self._synthesizer

    @property
    def sample_rate(self) -> int | None:
        """The rate the loaded clips are at, or ``None`` before the first load."""
        return self._sample_rate

    @property
    def ready(self) -> bool:
        """Whether every clip has audio. The gate stays shut until this is true."""
        return bool(self._sample_rate) and len(self._clips) == len(self._texts)

    def missing(self) -> list[str]:
        """Clip texts that still have no audio."""
        return [t for t in self._texts if t not in self._clips]

    def get(self, text: str) -> bytes | None:
        """Return a clip's PCM, or ``None`` if it isn't loaded."""
        return self._clips.get(text)

    def store(self, text: str, pcm: bytes) -> None:
        """Add a clip's audio and write it to the cache.

        Args:
            text: The clip text. Must be part of this library's inventory.
            pcm: Raw mono 16-bit little-endian PCM at :attr:`sample_rate`.

        Raises:
            RuntimeError: Called before the library knows its sample rate.
            KeyError: ``text`` is not in the inventory.
        """
        if self._sample_rate is None:
            raise RuntimeError("ClipLibrary.store() called before load().")
        if text not in self._texts:
            raise KeyError(f"{text!r} is not in this library's clip inventory.")
        self._clips[text] = pcm
        self._cache.put(text, self._sample_rate, pcm)

    async def load(self, sample_rate: int) -> None:
        """Fill the library at ``sample_rate``, using the cache then a synthesizer.

        Idempotent: calling it again at the same rate does nothing. Calling it at
        a *different* rate discards what's loaded and reloads, because a clip
        produced at one rate must never be played at another.

        Missing clips are left missing rather than raising — the library reports
        that through :attr:`ready`, and something else may fill them in later.
        """
        async with self._lock:
            if sample_rate == self._sample_rate:
                if not self.missing():
                    return
            else:
                if self._sample_rate is not None:
                    logger.debug(
                        f"Backchannel clips: sample rate changed "
                        f"{self._sample_rate} -> {sample_rate}, reloading"
                    )
                self._clips.clear()
                self._sample_rate = sample_rate

            from_cache = 0
            synthesized = 0
            for text in self.missing():
                pcm = self._cache.get(text, sample_rate)
                if pcm is not None:
                    from_cache += 1
                elif self._synthesizer is not None:
                    pcm = await self._synthesizer(text, sample_rate)
                    self._cache.put(text, sample_rate, pcm)
                    synthesized += 1
                if pcm is not None:
                    self._clips[text] = pcm
            if synthesized:
                logger.info(
                    f"Backchannel: synthesized {synthesized} clip(s) (one-time, cached after)"
                )
            if from_cache:
                logger.debug(f"Backchannel: loaded {from_cache} clip(s) from cache")
