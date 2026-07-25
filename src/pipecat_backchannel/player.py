"""Plays pre-cached backchannel clips as raw audio."""

from __future__ import annotations

import array

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_backchannel.frames import PlayCachedClipFrame
from pipecat_backchannel.library import ClipLibrary


class ClipPlayer(FrameProcessor):
    """Turns :class:`~pipecat_backchannel.frames.PlayCachedClipFrame` markers
    into audio.

    Place this **after** the live TTS service and **before**
    ``transport.output()``. Sitting downstream of TTS means it never touches the
    TTS service's own context or state machinery — it just emits the same
    ``TTSStarted`` / ``TTSAudioRaw`` / ``TTSStopped`` shape real TTS output would,
    so ``BotStartedSpeaking`` / ``BotStoppedSpeaking`` and audio pacing work
    identically for a backchannel and for a real reply.
    """

    def __init__(self, *, clips: ClipLibrary | None = None, volume: float = 0.6, **kwargs):
        """Initialize the player.

        Args:
            clips: The clip library, shared with the
                :class:`~pipecat_backchannel.processor.BackchannelProcessor`.
                The player loads it on ``StartFrame``, at the pipeline's own
                output sample rate — you never supply one.
            volume: Linear scale applied to the clip's samples on the way out,
                where ``1.0`` is exactly as recorded. Below 1.0 by default: a
                real backchannel sits under the person still talking, and one
                delivered at full presenting volume sounds like an interruption
                even when its timing is right.
        """
        super().__init__(**kwargs)
        self._clips = clips or ClipLibrary()
        self._volume = volume

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Play cached clips; pass everything else through untouched."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._clips.load(frame.audio_out_sample_rate)

        elif isinstance(frame, PlayCachedClipFrame):
            await self._play(frame.text)
            return  # the marker itself must never leak downstream

        await self.push_frame(frame, direction)

    async def _play(self, text: str):
        pcm = self._clips.get(text)
        if not pcm:
            # Silence here would be indistinguishable from the gate deciding not
            # to fire, which makes a broken inventory impossible to debug.
            logger.warning(
                f"Backchannel clip {text!r} has no audio in the library — nothing played. "
                "The clip inventory and the loaded clips are out of sync."
            )
            return

        # append_to_context defaults to True, and the assistant aggregator opens
        # an assistant turn on any TTSStartedFrame carrying it. That would record
        # the bot as having taken the floor — and the user talking on would then
        # count as barging in on that turn, so every clip would manufacture an
        # interruption. A backchannel is not a turn; say so explicitly.
        await self.push_frame(TTSStartedFrame(append_to_context=False), FrameDirection.DOWNSTREAM)
        await self.push_frame(
            TTSAudioRawFrame(
                audio=_at_volume(pcm, self._volume),
                sample_rate=self._clips.sample_rate,
                num_channels=1,
            ),
            FrameDirection.DOWNSTREAM,
        )
        await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)


def _at_volume(pcm: bytes, volume: float) -> bytes:
    """Scale mono 16-bit PCM, saturating rather than wrapping.

    Applied here rather than baked into the cached clip, so the cache stays a
    faithful recording of the voice and the level can be retuned without
    re-recording anything.

    Roughly 1ms for a half-second clip, which is why this stays a plain loop
    instead of gaining a numpy dependency.
    """
    if volume == 1.0:
        return pcm

    samples = array.array("h")
    samples.frombytes(pcm)
    for i, sample in enumerate(samples):
        # Overflowing an s16 wraps to full negative, which is an audible click.
        samples[i] = max(-32768, min(32767, int(sample * volume)))
    return samples.tobytes()
