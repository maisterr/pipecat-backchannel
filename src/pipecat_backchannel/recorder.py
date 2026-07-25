"""Records backchannel clips from a TTS service, before anything else runs.

This is what makes the default setup need no API key, no voice ID, and no
provider code: a pipeline already contains a TTS service, and this processor sits
directly downstream of it. So it asks that service to say "Mhm." — once — and
keeps the audio instead of passing it on.

**Why upstream, not ``run_tts()``.** Websocket-flavored services (which is most
streaming providers) can't be driven outside a pipeline at all: ``run_tts()``
yields ``None`` and the audio arrives asynchronously through machinery that only
runs while the service is live. Inside a pipeline that machinery is already
running, so a ``TTSSpeakFrame`` pushed upstream works for every service, HTTP or
websocket, without naming a single vendor.

**Why it holds the StartFrame.** A pipeline is only "ready" once the
``StartFrame`` reaches the end of it, and nothing — no greeting, no client-ready
handler, no reply — can run before that. Holding it here therefore buys a window
in which this processor is the only thing that can possibly be using the TTS. No
guessing whether arriving audio is the clip we asked for or the bot answering
someone; nothing else exists yet.

Waiting for gaps *during* a conversation was tried and is wrong: a client is
usually connected by the time the pipeline starts, so the first greeting lands
microseconds later, and any clip already in flight leaks to the user's speakers.

**Why nothing leaks.** The request carries ``append_to_context=False``, so the
service emits no assistant-aggregation frame and the text never reaches the LLM
context. The audio is swallowed here, upstream of ``transport.output()``, so
there is no sound and no bot-speaking frames. And the ``StartFrame`` is released
before anything else can speak, so the two never overlap.

Use :meth:`~pipecat_backchannel.plugin.Backchannel.prewarm` to do this once at
application startup instead of inside the first session.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_backchannel.library import ClipLibrary

#: Frames a TTS service emits for an utterance. While the StartFrame is held,
#: nothing else in the pipeline can be speaking, so these are always ours.
_TTS_OUTPUT = (TTSStartedFrame, TTSAudioRawFrame, TTSTextFrame, TTSStoppedFrame)


class ClipRecorder(FrameProcessor):
    """Fills a :class:`~pipecat_backchannel.library.ClipLibrary` from the TTS
    service upstream of it, while holding the pipeline in startup.

    Place this immediately **after** the live TTS service. Only needed when the
    library has no synthesizer and no pre-recorded clips — which is the default,
    and the reason the default needs no configuration.

    Completely transparent once every clip is stored, and on every later run,
    since the clips are read from cache and nothing is missing.
    """

    def __init__(
        self,
        *,
        clips: ClipLibrary,
        start_timeout_s: float = 10.0,
        chunk_timeout_s: float = 5.0,
        budget_s: float = 120.0,
        **kwargs,
    ):
        """Initialize the recorder.

        Args:
            clips: The library to fill, shared with the processor and player.
            start_timeout_s: How long to wait for the TTS to begin an utterance
                that was asked for.
            chunk_timeout_s: How long to wait for the next audio chunk of an
                utterance already in progress.
            budget_s: Hard ceiling on the whole recording pass. The pipeline is
                held open for this long at worst, then starts regardless — a
                misbehaving TTS must never wedge an app at startup.
        """
        super().__init__(**kwargs)
        self._clips = clips
        self._start_timeout_s = start_timeout_s
        self._chunk_timeout_s = chunk_timeout_s
        self._budget_s = budget_s

        self._recording = False
        self._inbox: asyncio.Queue[Frame] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._held_start: StartFrame | None = None
        self._deferred: list[Frame] = []
        self._rate_confirmed = False
        self._done = asyncio.Event()

    @property
    def recording(self) -> bool:
        """Whether the pipeline is still being held open to record clips."""
        return self._recording

    async def wait_until_done(self):
        """Block until recording has finished (or was never needed)."""
        await self._done.wait()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold the pipeline open to record; pass everything else through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._begin(frame, direction)
            return

        if self._recording and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, (EndFrame, CancelFrame)):
                # Shutting down mid-recording: give up and let startup complete,
                # so the frame that ends the pipeline isn't stuck behind us.
                await self._finish("pipeline shutting down")
            elif isinstance(frame, _TTS_OUTPUT):
                self._inbox.put_nowait(frame)
                return  # ours by construction — nothing else can be speaking
            else:
                # Everything upstream of here has already started, so a connected
                # client's audio keeps arriving while we hold the StartFrame.
                # Letting it past would deliver frames to processors that have
                # not been started yet. Hold the order instead.
                self._deferred.append(frame)
                return

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Release the pipeline if it is still being held."""
        await self._finish(None)
        await super().cleanup()

    async def _begin(self, frame: StartFrame, direction: FrameDirection):
        await self._clips.load(frame.audio_out_sample_rate)
        if not self._clips.missing():
            self._done.set()
            await self.push_frame(frame, direction)
            return

        # Hold it. Downstream stays un-started and the pipeline stays "not
        # ready", so no greeting, reply or client-ready handler can run yet.
        self._held_start = frame
        self._recording = True
        self._task = self.create_task(self._record_all())

    async def _finish(self, reason: str | None):
        if not self._recording:
            self._done.set()
            return
        self._recording = False
        if reason:
            logger.warning(f"Backchannel: clip recording stopped ({reason})")
        if self._task and not self._task.done():
            await self.cancel_task(self._task)
        self._task = None
        await self._release()

    async def _release(self):
        held, self._held_start = self._held_start, None
        if held is not None:
            await self.push_frame(held, FrameDirection.DOWNSTREAM)
        deferred, self._deferred = self._deferred, []
        for frame in deferred:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)
        self._done.set()

    async def _record_all(self):
        count = len(self._clips.missing())
        logger.info(
            f"Backchannel: recording {count} clip(s) from the TTS before the pipeline opens. "
            "One time only — they are cached on disk after this."
        )
        deadline = time.monotonic() + self._budget_s
        recorded_count = 0
        try:
            # Re-read what's missing each time round: confirming the true sample
            # rate can re-key the library, which changes what still needs doing.
            while pending := self._clips.missing():
                if time.monotonic() > deadline:
                    logger.warning(
                        f"Backchannel: gave up after {self._budget_s}s with "
                        f"{len(pending)} clip(s) left. Backchannels stay off; recording "
                        "retries on the next run."
                    )
                    return
                text = pending[0]
                recorded = await self._record_one(text)
                if recorded is None:
                    logger.warning(
                        f"Backchannel: the TTS produced no audio for {text!r}. Backchannels "
                        "stay off; recording retries on the next run. To avoid recording "
                        "entirely, pass a synthesizer or pre-record the clips."
                    )
                    return
                pcm, rate = recorded
                await self._confirm_rate(rate)
                self._clips.store(text, pcm)
                recorded_count += 1
            logger.info(f"Backchannel: clips ready ({recorded_count} recorded and cached)")
        finally:
            self._recording = False
            await self._release()

    async def _confirm_rate(self, rate: int):
        """Re-key the library if the TTS emits at a rate the StartFrame didn't predict.

        A service constructed with an explicit ``sample_rate`` ignores the
        pipeline's output rate, so the first recorded frame is the only reliable
        source of truth. Done once, before anything is stored.
        """
        if self._rate_confirmed:
            return
        self._rate_confirmed = True
        if rate != self._clips.sample_rate:
            await self._clips.load(rate)

    async def _record_one(self, text: str) -> tuple[bytes, int] | None:
        while not self._inbox.empty():
            self._inbox.get_nowait()

        await self.push_frame(
            TTSSpeakFrame(text=text, append_to_context=False), FrameDirection.UPSTREAM
        )

        chunks: list[bytes] = []
        rate: int | None = None
        while True:
            timeout = self._chunk_timeout_s if chunks else self._start_timeout_s
            try:
                frame = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            except TimeoutError:
                return None

            if isinstance(frame, TTSAudioRawFrame):
                rate = frame.sample_rate
                chunks.append(frame.audio)
            elif isinstance(frame, TTSStartedFrame):
                chunks.clear()  # a fresh utterance supersedes anything buffered
            elif isinstance(frame, TTSStoppedFrame):
                if not chunks or rate is None:
                    return None
                return b"".join(chunks), rate
