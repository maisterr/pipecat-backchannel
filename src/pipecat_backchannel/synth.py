"""Producing clip audio without the pipeline's own TTS service.

Only needed to give the backchannels a different voice from the bot's. By
default there is no synthesizer at all: the clips are recorded from the TTS
service already in the pipeline (see
:mod:`pipecat_backchannel.recorder`), which needs no keys and no configuration.

The protocol is deliberately just "text in, PCM out" rather than a pipecat
``TTSService``. A ``TTSService`` looks like the obvious seam and is not one:
websocket-flavored services — most streaming providers — deliver audio
out-of-band through a receive task that only runs inside a live pipeline, so
``run_tts()`` yields nothing when called directly. An abstraction that only fits
the HTTP-flavored half of the services is the implementation wearing a hat.

Because this is one async call with no lifecycle, a synthesizer that holds a
connection should open and close it within the call. Nothing here will tear one
down for you.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ClipSynthesizer(Protocol):
    """Turns a clip text into raw audio, once, ahead of time.

    Never called on the hot path: :class:`~pipecat_backchannel.library.ClipLibrary`
    calls it at pipeline start, for cache misses only.
    """

    async def __call__(self, text: str, sample_rate: int) -> bytes:
        """Synthesize one clip.

        Args:
            text: The clip text, e.g. ``"Mhm."``.
            sample_rate: Output sample rate in Hz. Must match the sample rate of
                the live TTS service in the pipeline, so the clips sound like the
                same voice at the same speed.

        Returns:
            Raw mono 16-bit little-endian PCM at ``sample_rate``.
        """
        ...
