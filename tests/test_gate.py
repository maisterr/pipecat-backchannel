"""The gate: when to say "mhm", and when to stay silent.

The hard part of the whole library — the only decision that can ruin the
experience. Covers every eligibility branch, the turn-analyzer economics
(inference is ~100ms on a shared CPU), and the own-clip attribution that keeps
the gate from mistaking its own audio for a bot reply.
"""

import asyncio

from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from conftest import (
    FakeSTT,
    FakeTTSService,
    FakeTurnAnalyzer,
    FakeVAD,
    MemoryCache,
    _audio,
    _fired,
    _make,
    _pause,
    _speak,
    _transports,
    _wire,
)
from pipecat_backchannel import Backchannel, BackchannelParams
from pipecat_backchannel.clips import DEFAULT_CLIP_GROUPS, flatten_clips
from pipecat_backchannel.library import ClipLibrary
from pipecat_backchannel.processor import BackchannelProcessor


class BlockingTurnAnalyzer(FakeTurnAnalyzer):
    """Classifier that stays in flight until the test lets it finish."""

    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def analyze_end_of_turn(self):
        self.analyzed += 1
        await self.release.wait()
        return self.state, None


async def _bot_says_something(processor):
    """What the output transport broadcasts upstream whenever audio plays.

    Identical frames whether the audio was a real reply or a backchannel clip:
    ``base_output.py`` builds a fresh ``BotStartedSpeakingFrame`` from any
    ``OutputAudioRawFrame`` it receives and pushes it UPSTREAM.
    """
    await processor.queue_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)


async def _bot_finishes(processor):
    await processor.queue_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)


def _record_spawned_tasks(processor) -> list:
    """Capture every task the processor spawns, so a test can see skipped work."""
    spawned = []
    original = processor.create_task

    def counting(coro, *args, **kwargs):
        spawned.append(coro)
        return original(coro, *args, **kwargs)

    processor.create_task = counting
    return spawned


# ------------------------------------------------------------------ fire or not


async def test_fires_on_incomplete_pause():
    processor, vad, _ = await _make()
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    fired = _fired(sink)
    assert len(fired) == 1
    assert fired[0].text in flatten_clips(DEFAULT_CLIP_GROUPS)


async def test_silent_when_turn_is_complete():
    processor, vad, _ = await _make(turn_state=EndOfTurnState.COMPLETE)
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert _fired(sink) == []


async def test_passes_every_frame_through():
    from pipecat.frames.frames import InputAudioRawFrame

    processor, vad, _ = await _make()
    sink = await _wire(processor)

    await _speak(processor, vad, chunks=3)
    audio = [f for f in sink.frames if isinstance(f, InputAudioRawFrame)]
    assert len(audio) == 3


async def test_cooldown_blocks_second_fire():
    processor, vad, _ = await _make()
    processor._params.cooldown_s = 60.0
    sink = await _wire(processor)

    for _ in range(2):
        await _speak(processor, vad)
        await _pause(processor, vad)
        await asyncio.sleep(0.15)

    assert len(_fired(sink)) == 1


async def test_zero_probability_never_fires():
    processor, vad, _ = await _make()
    processor._params.fire_probability = 0.0
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert _fired(sink) == []


async def test_first_fire_is_not_blocked_by_the_cooldown_near_boot():
    # Larger than any real machine's uptime, so with the old 0.0 sentinel
    # ``now - 0.0`` is always inside the cooldown and this fails on every host;
    # with the -inf sentinel the elapsed time is infinite and it passes.
    processor, vad, _ = await _make()
    processor._params.cooldown_s = 1e12
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert len(_fired(sink)) == 1


async def test_suppressed_while_bot_speaking():
    processor, vad, _ = await _make()
    sink = await _wire(processor)

    await processor.queue_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.02)
    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert _fired(sink) == []


async def test_min_speech_gate_ignores_early_pause():
    processor, vad, _ = await _make()
    processor._params.min_speech_before_eligible_s = 60.0
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert _fired(sink) == []


async def test_disabled_never_fires():
    processor, vad, _ = await _make()
    processor.enabled = False
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert _fired(sink) == []


async def test_silent_until_clips_are_ready():
    """A half-filled library must keep the gate shut, not play nothing loudly."""
    vad = FakeVAD()
    processor = BackchannelProcessor(
        params=BackchannelParams(
            min_speech_before_eligible_s=0.0,
            cooldown_s=0.0,
            fire_probability=1.0,
        ),
        clips=ClipLibrary(cache=MemoryCache(prefilled=False)),
        vad_analyzer=vad,
        turn_analyzer=FakeTurnAnalyzer(),
    )
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert _fired(sink) == []


# --------------------------------------- the gate's own clip echoing back at it


async def test_own_clip_does_not_wipe_the_turn_buffer():
    """The clip comes back as a BotStartedSpeakingFrame, and clearing on it is wrong.

    A backchannel plays *during* the user's turn. Treating the echo as the bot
    taking the floor throws away the audio the next classification needs — so
    every fired clip degraded the very decision it was fired to support.
    """
    processor, vad, turn = await _make()
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)
    assert len(_fired(sink)) == 1

    await _bot_says_something(processor)

    assert turn.cleared == 0


async def test_a_real_reply_still_wipes_the_turn_buffer():
    """The other half of the same decision: a genuine reply does end the turn."""
    processor, vad, turn = await _make(fire_probability=0.0)
    await _wire(processor)

    await _speak(processor, vad)
    await _bot_says_something(processor)

    assert turn.cleared == 1


async def test_own_clip_does_not_restart_the_min_speech_clock():
    """The clip echo also marked the next speech as a fresh turn.

    So ``min_speech_before_eligible_s`` restarted mid-turn, and the user had to
    re-earn eligibility they had already earned.
    """
    processor, vad, _ = await _make(min_speech_before_eligible_s=0.3)
    sink = await _wire(processor)

    await _speak(processor, vad)
    await asyncio.sleep(0.35)  # past the threshold, so the first pause is eligible
    await _pause(processor, vad)
    await asyncio.sleep(0.15)
    assert len(_fired(sink)) == 1

    await _bot_says_something(processor)
    await _bot_finishes(processor)

    await _speak(processor, vad)  # same turn, the user simply carried on
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert len(_fired(sink)) == 2


async def test_a_clip_that_never_reached_the_transport_stops_shadowing_replies():
    """Attribution is by recency, so it has to expire.

    Without a bound, one clip swallowed by a misconfigured pipeline would make
    every later reply look like the backchannel's own audio, forever.
    """
    processor, vad, turn = await _make()
    sink = await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)
    assert len(_fired(sink)) == 1

    processor._last_fired_at -= 3600  # the clip never arrived; an hour passes

    await _bot_says_something(processor)

    assert turn.cleared == 1


# --------------------------------------------------- paying for inference wisely


async def test_gate_does_not_pay_for_inference_it_will_discard():
    """The classifier costs ~100ms on a CPU the real turn-detector shares.

    Running it before the cooldown and probability gates means spending that on
    most pauses of every turn and throwing the answer away — the cause of the
    whole pipeline going sluggish while a backchannel is installed.
    """
    processor, vad, turn = await _make()
    processor._params.cooldown_s = 60.0
    await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)
    after_first = turn.analyzed

    for _ in range(3):  # all inside the cooldown
        await _speak(processor, vad)
        await _pause(processor, vad)
        await asyncio.sleep(0.15)

    assert after_first == 1  # the firing pause is classified
    assert turn.analyzed == 1  # the blocked ones cost nothing


async def test_a_blocked_pause_spawns_no_task_at_all():
    """The gate's cheap checks are pure comparisons; they do not need a task.

    Every pause used to allocate one just to discover it was in cooldown.
    """
    processor, vad, _ = await _make(cooldown_s=60.0)
    await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    spawned = _record_spawned_tasks(processor)
    for _ in range(3):  # all inside the cooldown
        await _speak(processor, vad)
        await _pause(processor, vad)
        await asyncio.sleep(0.1)

    assert spawned == []


async def test_only_one_classification_runs_at_a_time():
    """Inference is serialized on a single worker thread.

    Queueing a second one behind the first buys a verdict about a moment that
    has already passed.
    """
    turn = BlockingTurnAnalyzer()
    processor, vad, _ = await _make(turn=turn)
    await _wire(processor)

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.05)
    assert turn.analyzed == 1  # in flight, blocked

    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.05)

    assert turn.analyzed == 1
    turn.release.set()


async def test_gate_does_not_classify_while_the_bot_is_speaking():
    processor, vad, turn = await _make()
    await _wire(processor)

    await processor.queue_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.02)
    await _speak(processor, vad)
    await _pause(processor, vad)
    await asyncio.sleep(0.15)

    assert turn.analyzed == 0


async def test_turn_buffer_is_released_when_the_user_turn_ends():
    """Skipped classifications skip the _clear that came with them.

    The analyzer only empties itself on a COMPLETE verdict or 3s of unbroken
    silence, so a gate that declines most classifications would otherwise let it
    accumulate the whole call's audio.
    """
    processor, vad, turn = await _make()
    await _wire(processor)

    await _speak(processor, vad)
    assert turn.cleared == 0

    await processor.queue_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)

    assert turn.cleared == 1


# ------------------------------------------------------- reading the VAD's truth


async def test_a_false_start_is_not_a_pause():
    """QUIET -> STARTING -> QUIET is a cough, not the user pausing mid-thought.

    The real VAD reaches QUIET from STARTING as well as from STOPPING, and the
    gate could not tell those apart.
    """
    processor, vad, turn = await _make()
    sink = await _wire(processor)

    await _audio(processor, vad, VADState.STARTING)
    await _audio(processor, vad, VADState.QUIET)
    await asyncio.sleep(0.15)

    assert turn.analyzed == 0
    assert _fired(sink) == []


async def test_a_real_pause_still_arrives_through_stopping():
    """...and the real VAD never goes straight from SPEAKING to QUIET.

    It passes through STOPPING for ``stop_secs`` first, so a fix that matched on
    the state pair alone would silence the gate completely.
    """
    processor, vad, turn = await _make()
    sink = await _wire(processor)

    await _audio(processor, vad, VADState.SPEAKING, chunks=3)
    await _audio(processor, vad, VADState.STOPPING, chunks=2)
    await _audio(processor, vad, VADState.QUIET)
    await asyncio.sleep(0.15)

    assert turn.analyzed == 1
    assert len(_fired(sink)) == 1


async def test_turn_analyzer_follows_the_vad_that_actually_runs():
    """The analyzer aligns its buffer to speech onset using this value.

    It was read from ``BackchannelParams`` even when the VAD was injected, so an
    injected VAD left the two describing different audio.
    """
    vad = FakeVAD(start_secs=0.5)
    turn = FakeTurnAnalyzer()
    processor, _, _ = await _make(vad=vad, turn=turn, vad_start_secs=0.2)

    await _wire(processor)

    assert turn.vad_start_secs == 0.5


# ------------------------------------------------------- the shared turn model


def test_shared_turn_analyzer_reuses_one_model_with_separate_state():
    """Two analyzers need two buffers, but only one set of weights."""
    from pipecat_backchannel.turn import shared_turn_analyzer

    a, b = shared_turn_analyzer(), shared_turn_analyzer()

    assert a is not b
    assert a._session is b._session  # one ONNX model for the process
    assert a._audio_buffer is not b._audio_buffer  # independent conversations


def test_shared_turn_analyzer_keeps_configurations_apart():
    """Different arguments may mean a different model — never share those."""
    from pipecat_backchannel.turn import shared_turn_analyzer

    default = shared_turn_analyzer()
    tuned = shared_turn_analyzer(cpu_count=2)

    assert default._session is not tuned._session
    assert tuned._session is shared_turn_analyzer(cpu_count=2)._session


def test_shared_turn_analyzer_accepts_configured_params():
    """``turn_factory`` is a documented seam, and any configuration hit a TypeError.

    ``SmartTurnParams`` is a pydantic model, so it is unhashable, and the shared
    session was keyed on the kwargs themselves.
    """
    from pipecat_backchannel.turn import shared_turn_analyzer

    first = shared_turn_analyzer(params=SmartTurnParams(stop_secs=1.0))
    second = shared_turn_analyzer(params=SmartTurnParams(stop_secs=1.0))

    assert first is not second
    assert first._session is second._session


def test_shared_turn_analyzer_still_separates_different_params():
    from pipecat_backchannel.turn import shared_turn_analyzer

    one = shared_turn_analyzer(params=SmartTurnParams(stop_secs=1.0))
    other = shared_turn_analyzer(params=SmartTurnParams(stop_secs=2.0))

    assert one._session is not other._session


def test_plugin_gives_each_pipeline_its_own_analyzer_over_one_model():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    backchannel = Backchannel()

    first = next(
        p for p in backchannel([tin, stt, tts, tout]) if isinstance(p, BackchannelProcessor)
    )
    second = next(
        p for p in backchannel([tin, stt, tts, tout]) if isinstance(p, BackchannelProcessor)
    )

    assert first._turn_analyzer is not second._turn_analyzer
    assert first._turn_analyzer._session is second._turn_analyzer._session


def test_plugin_turn_factory_is_injectable():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    built = []

    def factory():
        analyzer = FakeTurnAnalyzer()
        built.append(analyzer)
        return analyzer

    out = Backchannel(turn_factory=factory)([tin, stt, tts, tout])

    gate = next(p for p in out if isinstance(p, BackchannelProcessor))
    assert gate._turn_analyzer is built[0]
