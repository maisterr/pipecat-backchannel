"""The backchannel processor: decides *when* to say "mhm".

The rule is deliberately literal: a pause, plus a turn-detector that says the
user isn't finished, means inject a continuer. No hand-rolled filler-word regex
in the gating decision — that's pipecat's real trained turn-detector answering
the question it was trained to answer.

**Two VAD instances, deliberately.** The main pipeline's VAD (on
``LLMUserAggregatorParams``) drives turn-start and barge-in interruption, where a
*longer* ``start_secs`` is better: it stops coughs and stray noise from
interrupting the bot mid-reply. Backchannel timing wants the opposite — fast
reaction to real pauses. Tying both to one VAD forces a bad tradeoff, so this
processor runs its own. ``VADAnalyzer`` supports standalone use directly:
``set_sample_rate()`` once, then ``analyze_audio()`` per chunk returns a
debounced ``VADState``.

The same applies to the turn analyzer, for a different reason: it holds one
audio buffer, and ``analyze_end_of_turn()`` wipes it whenever the verdict is
COMPLETE. A backchannel asking at its own 0.2s pauses would therefore destroy
the segment ``LLMUserAggregator`` still needs for the real end-of-turn decision.
So this runs a second, independent analyzer — the same question asked of the
same model at a different moment.

The *model* is shared even so: see :mod:`pipecat_backchannel.turn`, which gives
each pipeline its own analyzer over one process-wide ONNX session, so only one
copy stays resident. That saves memory, not inference — running a second VAD and
a second smart-turn model still roughly doubles inference cost during speech.
Fine for a handful of concurrent sessions; worth measuring before a large
deployment.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_backchannel.clips import ClipSelector, HeuristicClipSelector
from pipecat_backchannel.frames import PlayCachedClipFrame
from pipecat_backchannel.library import ClipLibrary


@dataclass
class BackchannelParams:
    """Tuning for :class:`BackchannelProcessor`.

    Parameters:
        min_speech_before_eligible_s: Ignore pauses this soon after a *turn*
            starts, measured from the first speech onset after the bot last
            spoke — not reset on every micro-resume within an ongoing turn.
            Natural halting speech is full of short bursts; resetting per burst
            means this threshold is almost never reached.
        cooldown_s: Minimum time between two backchannels. This and
            ``fire_probability`` are the two dials for how present the listener
            feels; between them they cap the rate at roughly one every few
            seconds of continuous speech.
        fire_probability: Chance of firing at an otherwise eligible moment.
            Below 1.0 because a bot that responds to *every* eligible pause
            sounds mechanical; humans skip some of them.
        vad_start_secs: ``start_secs`` for this processor's own VAD. Short, on
            purpose — see the module docstring.
        vad_stop_secs: ``stop_secs`` for this processor's own VAD. Doubles as the
            gate's patience: no pause is reported, and so nothing fires, until
            the user has been quiet this long.
    """

    min_speech_before_eligible_s: float = 0.7
    cooldown_s: float = 2.5
    fire_probability: float = 0.8
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2


#: How long a bot-speaking frame stays attributable to a clip this processor
#: fired. Long enough to cover the trip through the player and out to the
#: transport; short enough that a clip which never got there stops shadowing
#: real replies.
_OWN_CLIP_ATTRIBUTION_S = 3.0


class BackchannelProcessor(FrameProcessor):
    """Fires backchannel clips on turn-incomplete pauses.

    Passes every frame through unchanged and never touches the LLM context — the
    bot's conversation history has no idea this happened, which is exactly right:
    a continuer isn't a turn and shouldn't be remembered as one.

    :class:`~pipecat_backchannel.plugin.Backchannel` builds and places one of
    these for you. Constructing it directly means placing it yourself: after your
    STT service, with a :class:`~pipecat_backchannel.player.ClipPlayer` after
    your TTS service sharing the same
    :class:`~pipecat_backchannel.library.ClipLibrary`.
    """

    def __init__(
        self,
        *,
        params: BackchannelParams | None = None,
        clips: ClipLibrary | None = None,
        selector: ClipSelector | None = None,
        vad_analyzer: VADAnalyzer | None = None,
        turn_analyzer: BaseTurnAnalyzer | None = None,
        recent_clip_memory: int = 3,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            params: Tuning. Defaults to :class:`BackchannelParams`.
            clips: The clip library, shared with the
                :class:`~pipecat_backchannel.player.ClipPlayer`. Supplies both
                the inventory to choose from and the readiness signal — the gate
                stays shut until every clip has audio.
            selector: Clip-choice strategy. Defaults to
                :class:`~pipecat_backchannel.clips.HeuristicClipSelector`.
            vad_analyzer: This processor's own VAD, separate from the pipeline's.
                Defaults to a ``SileroVADAnalyzer`` built from ``params``.
            turn_analyzer: The end-of-turn classifier. Defaults to
                ``LocalSmartTurnAnalyzerV3``.
            recent_clip_memory: How many recent clips to avoid repeating, so it
                won't say "Yeah, yeah." after every pause.
        """
        super().__init__(**kwargs)
        self._params = params or BackchannelParams()
        self._clips = clips or ClipLibrary()
        self._selector = selector or HeuristicClipSelector()

        #: Set to ``False`` to go quiet without touching the pipeline, e.g.
        #: while the bot is walking someone through a form.
        self.enabled = True
        self._turn_analyzer = turn_analyzer or LocalSmartTurnAnalyzerV3()
        self._vad = vad_analyzer or SileroVADAnalyzer(
            params=VADParams(
                start_secs=self._params.vad_start_secs,
                stop_secs=self._params.vad_stop_secs,
            )
        )
        self._recent_clip_memory = recent_clip_memory

        self._in_speech = False
        self._speech_started_at = 0.0
        self._awaiting_fresh_turn = True
        self._bot_speaking = False
        self._awaiting_own_clip = False
        self._classifying = False
        self._last_fired_at = float("-inf")
        self._pause_tasks: set[asyncio.Task] = set()
        self._last_partial = ""
        self._recent_clips: deque[str] = deque(maxlen=recent_clip_memory)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Watch audio and bot state; pass every frame through untouched."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
            self._vad.set_sample_rate(frame.audio_in_sample_rate)
            # Read off the VAD that actually runs, not off params: the VAD is
            # injectable, and the analyzer uses this to align its buffer with
            # speech onset. Two sources for one fact means they can disagree.
            self._turn_analyzer.update_vad_start_secs(self._vad.params.start_secs)

        elif isinstance(frame, InputAudioRawFrame):
            await self._on_audio(frame)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if not self._is_own_clip():
                self._awaiting_fresh_turn = True
                # The user's turn is over, so drop the audio buffered for it.
                # Without this the buffer only empties when a classification
                # happens to come back COMPLETE — and the gate skips most
                # classifications — so it would accumulate the whole call.
                # Pipecat's own turn strategy clears at exactly this boundary.
                self._turn_analyzer.clear()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._awaiting_own_clip = False

        elif isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            text = getattr(frame, "text", "")
            if text:
                self._last_partial = text

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Cancel any in-flight pause classification."""
        for task in list(self._pause_tasks):
            await self.cancel_task(task)
        self._pause_tasks.clear()
        await super().cleanup()

    async def _on_audio(self, frame: InputAudioRawFrame):
        state = await self._vad.analyze_audio(frame.audio)
        self._turn_analyzer.append_audio(frame.audio, state == VADState.SPEAKING)

        # Tracked as "has speech been confirmed since the last pause" rather than
        # as a state pair, because the VAD reaches QUIET two different ways:
        # SPEAKING -> STOPPING -> QUIET is the user pausing mid-thought, while
        # STARTING -> QUIET is a cough the VAD talked itself out of. Only the
        # first is worth classifying, and only the first ever passes through
        # SPEAKING — so matching on the pair alone gets both cases wrong.
        if state == VADState.SPEAKING and not self._in_speech:
            self._in_speech = True
            await self._on_speech_started()
        elif state == VADState.QUIET and self._in_speech:
            self._in_speech = False
            await self._on_pause_started()

    async def _on_speech_started(self):
        # Only reset the "just started talking" clock at the start of a fresh
        # turn (right after the bot spoke, or session start) — not on every
        # micro-resume within one ongoing turn.
        if self._awaiting_fresh_turn:
            self._speech_started_at = time.monotonic()
            self._awaiting_fresh_turn = False

    def _pick_clip(self) -> str:
        clip = self._selector.select(self._last_partial, self._clips.groups, self._recent_clips)
        self._recent_clips.append(clip)
        return clip

    def _is_own_clip(self) -> bool:
        """Whether the bot-speaking frame just seen is a clip this gate fired.

        The output transport builds those frames from *any* audio it receives
        and broadcasts them upstream, so a backchannel and a real reply arrive
        here looking identical. The difference matters: a real reply ends the
        user's turn, while a clip plays in the middle of one — clearing the turn
        buffer on a clip throws away the context the next classification needs.

        Attribution is by recency, and bounded, so a clip that never reached the
        transport cannot shadow real replies for the rest of the call.
        """
        return self._awaiting_own_clip and (
            time.monotonic() - self._last_fired_at < _OWN_CLIP_ATTRIBUTION_S
        )

    def _eligible(self) -> bool:
        """Cheap checks, all of them, before any inference is paid for.

        Every one of these is a pure comparison; the classifier below is ~100ms
        of ONNX inference on a CPU the real turn-detector is also using. Asking
        the model and *then* discarding the answer on a coin flip — which is what
        this did originally — spends that on most pauses of every turn, for
        nothing. Ordering matters far more here than it looks.
        """
        if not self.enabled:
            return False
        if not self._clips.ready:
            return False  # clips still being prepared; nothing to play yet
        if self._bot_speaking:
            return False
        if self._classifying:
            # Inference is serialized on one worker thread, so a second request
            # would queue behind the first and answer a moment already gone.
            return False

        now = time.monotonic()
        if now - self._speech_started_at < self._params.min_speech_before_eligible_s:
            logger.debug("Backchannel: skip, too soon after turn started")
            return False
        if now - self._last_fired_at < self._params.cooldown_s:
            logger.debug("Backchannel: skip, cooldown")
            return False
        if random.random() > self._params.fire_probability:
            logger.debug("Backchannel: skip, probability gate")
            return False
        return True

    async def _on_pause_started(self):
        if not self._eligible():
            return
        # Only now is a task worth allocating. Everything above was a comparison;
        # everything below is inference, and must not block the audio path.
        self._classifying = True
        task = self.create_task(self._classify())
        self._pause_tasks.add(task)
        task.add_done_callback(self._pause_tasks.discard)

    async def _classify(self):
        try:
            state, _ = await self._turn_analyzer.analyze_end_of_turn()
        finally:
            self._classifying = False
        logger.debug(f"Backchannel: pause classified {state.name}")

        if state != EndOfTurnState.INCOMPLETE:
            return  # turn-detector says complete -> let the real response happen
        if self._bot_speaking:
            return  # the bot started talking while the classifier was running
        if self._in_speech:
            # The user carried on while the classifier was thinking, so the pause
            # it was asked about is over. This is the race a delay before firing
            # used to guard against, answered from state we already have rather
            # than by making every clip late.
            logger.debug("Backchannel: skip, user resumed during classification")
            return

        self._last_fired_at = time.monotonic()
        self._awaiting_own_clip = True
        clip = self._pick_clip()
        logger.debug(f"Backchannel: firing {clip!r}")
        await self.push_frame(PlayCachedClipFrame(clip), FrameDirection.DOWNSTREAM)
