"""Startup: recording clips through the pipeline's TTS, and prewarming.

Everything slow happens here, ahead of the conversation, so the live decision
stays instant. The recorder holds the pipeline shut while it works; prewarm
moves the same work to app start.
"""

import asyncio

from pipecat.frames.frames import (
    CancelFrame,
    InputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from conftest import (
    CHUNK,
    PCM,
    SAMPLE_RATE,
    FakeTTS,
    MemoryCache,
    Sink,
    _setup,
)
from pipecat_backchannel import Backchannel
from pipecat_backchannel.library import ClipLibrary
from pipecat_backchannel.recorder import ClipRecorder


async def _wire_recorder(tts: FakeTTS, recorder: ClipRecorder) -> Sink:
    sink = Sink()
    await _setup(tts, recorder, sink)
    await tts.queue_frame(
        StartFrame(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.wait_for(recorder.wait_until_done(), timeout=5)
    await asyncio.sleep(0.05)
    return sink


def _started(sink: Sink) -> list[StartFrame]:
    return [f for f in sink.frames if isinstance(f, StartFrame)]


# ---------------------------------------------------------------------- recorder


async def test_recorder_fills_the_library_from_upstream_tts():
    cache = MemoryCache(prefilled=False)
    library = ClipLibrary(groups={"continue": ["Mhm.", "Yeah."]}, cache=cache)
    tts = FakeTTS()
    recorder = ClipRecorder(clips=library)

    sink = await _wire_recorder(tts, recorder)

    assert [f.text for f in tts.spoken] == ["Mhm.", "Yeah."]
    assert library.ready
    assert library.get("Mhm.") == PCM * 2
    assert cache.stored[("Yeah.", SAMPLE_RATE)] == PCM * 2
    # Nothing recorded may reach the transport.
    assert not any(
        isinstance(f, (TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame)) for f in sink.frames
    )
    # ...and the pipeline is released only once, afterwards.
    assert len(_started(sink)) == 1


async def test_recorder_holds_the_pipeline_closed_until_the_clips_are_done():
    """The StartFrame is the pipeline's "ready" signal.

    Downstream must not see it while recording, or a client-ready handler fires
    and a greeting starts talking over the clip that is still being recorded.
    """
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    tts = FakeTTS(answer=False)  # never responds, so the hold persists
    recorder = ClipRecorder(clips=library, start_timeout_s=0.2)
    sink = Sink()
    await _setup(tts, recorder, sink)

    await tts.queue_frame(
        StartFrame(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    assert recorder.recording
    assert _started(sink) == []  # held

    await asyncio.wait_for(recorder.wait_until_done(), timeout=5)
    await asyncio.sleep(0.05)
    assert len(_started(sink)) == 1  # released even though recording failed


async def test_recorder_does_not_let_frames_overtake_the_held_start_frame():
    """Everything upstream of the recorder has already started.

    So a connected client's audio kept flowing while the StartFrame was held,
    reaching processors that had not been started yet.
    """
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    tts = FakeTTS(answer=False)  # never responds, so the hold persists
    recorder = ClipRecorder(clips=library, start_timeout_s=0.2)
    sink = Sink()
    await _setup(tts, recorder, sink)

    await tts.queue_frame(
        StartFrame(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    assert recorder.recording

    await recorder.queue_frame(
        InputAudioRawFrame(audio=CHUNK, sample_rate=SAMPLE_RATE, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    assert sink.frames == []  # nothing may pass while the pipeline is held shut

    await asyncio.wait_for(recorder.wait_until_done(), timeout=5)
    await asyncio.sleep(0.05)

    kinds = [type(f) for f in sink.frames]
    assert kinds.index(StartFrame) < kinds.index(InputAudioRawFrame)


async def test_recorder_never_writes_to_the_llm_context():
    """A clip request must not be appendable to the conversation."""
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    tts = FakeTTS()
    await _wire_recorder(tts, ClipRecorder(clips=library))

    assert tts.spoken[0].append_to_context is False


async def test_recorder_is_inert_when_clips_are_cached():
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=True))
    tts = FakeTTS()
    recorder = ClipRecorder(clips=library)

    sink = await _wire_recorder(tts, recorder)

    assert tts.spoken == []
    assert recorder.recording is False
    assert library.ready
    assert len(_started(sink)) == 1  # never held when there is nothing to do


async def test_recorder_adopts_the_rate_the_tts_actually_emits():
    """An explicitly-configured TTS ignores the pipeline rate; the clips follow it."""
    cache = MemoryCache(prefilled=False)
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=cache)
    tts = FakeTTS(sample_rate=24000)

    await _wire_recorder(tts, ClipRecorder(clips=library))

    assert library.sample_rate == 24000
    assert ("Mhm.", 24000) in cache.stored


async def test_recorder_finishes_clips_re_keyed_by_the_rate_change():
    """Re-keying mid-run must not leave clips stranded outside the work list.

    Some clips cached at the pipeline rate, a TTS that emits at another: the
    re-key makes every clip missing again, including ones that were never in the
    original batch. Missing those would leave the library permanently unready
    and re-record the same clips on every single run.
    """
    cache = MemoryCache(prefilled=False)
    cache.put("Mhm.", SAMPLE_RATE, PCM)  # cached at the pipeline rate only
    library = ClipLibrary(groups={"continue": ["Mhm.", "Yeah."]}, cache=cache)
    tts = FakeTTS(sample_rate=24000)

    await _wire_recorder(tts, ClipRecorder(clips=library))

    assert library.sample_rate == 24000
    assert library.ready
    assert ("Mhm.", 24000) in cache.stored
    assert ("Yeah.", 24000) in cache.stored


async def test_recorder_releases_the_pipeline_when_shutting_down():
    """A disconnect mid-recording must not strand the held StartFrame."""
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    tts = FakeTTS(answer=False)
    recorder = ClipRecorder(clips=library, start_timeout_s=30)
    sink = Sink()
    await _setup(tts, recorder, sink)
    await tts.queue_frame(
        StartFrame(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    assert recorder.recording

    await tts.queue_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.1)

    assert recorder.recording is False
    assert len(_started(sink)) == 1
    assert any(isinstance(f, CancelFrame) for f in sink.frames)


async def test_recorder_stops_after_a_silent_tts():
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    tts = FakeTTS(answer=False)
    recorder = ClipRecorder(clips=library, start_timeout_s=0.05)

    sink = await _wire_recorder(tts, recorder)

    assert library.ready is False
    assert recorder.recording is False
    assert len(_started(sink)) == 1  # startup completes regardless


async def test_recorder_gives_up_within_its_budget():
    """A misbehaving TTS must never wedge an app at startup."""
    library = ClipLibrary(
        groups={"continue": ["Mhm.", "Yeah.", "Right."]}, cache=MemoryCache(prefilled=False)
    )
    tts = FakeTTS(answer=False)
    recorder = ClipRecorder(clips=library, start_timeout_s=0.05, budget_s=0.1)

    sink = await _wire_recorder(tts, recorder)

    assert not library.ready
    assert len(_started(sink)) == 1


# ----------------------------------------------------------------------- prewarm


async def test_prewarm_fills_the_library_in_a_throwaway_pipeline():
    """The app-start path: real Pipeline, real PipelineTask, no conversation."""
    from pipecat_backchannel.prewarm import prewarm_library

    cache = MemoryCache(prefilled=False)
    library = ClipLibrary(groups={"continue": ["Mhm.", "Yeah."]}, cache=cache)
    tts = FakeTTS()

    ready = await prewarm_library(library, tts=tts, sample_rate=SAMPLE_RATE, budget_s=10)

    assert ready
    assert library.ready
    assert [f.text for f in tts.spoken] == ["Mhm.", "Yeah."]
    assert cache.stored[("Mhm.", SAMPLE_RATE)] == PCM * 2


async def test_prewarm_returns_false_rather_than_hanging():
    from pipecat_backchannel.prewarm import prewarm_library

    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    tts = FakeTTS(answer=False)

    ready = await prewarm_library(library, tts=tts, sample_rate=SAMPLE_RATE, budget_s=0.2)

    assert ready is False
    assert not library.ready


async def test_plugin_prewarm_records_through_the_tts_it_is_given():
    """The optional app-start path, reached through the object you already built."""
    cache = MemoryCache(prefilled=False)
    backchannel = Backchannel(clip_groups={"continue": ["Mhm.", "Yeah."]}, cache=cache)

    assert await backchannel.prewarm(tts=FakeTTS(), sample_rate=SAMPLE_RATE, budget_s=10)
    assert backchannel.clips.ready
    assert cache.stored[("Mhm.", SAMPLE_RATE)] == PCM * 2


async def test_plugin_prewarm_is_a_no_op_when_cached():
    backchannel = Backchannel(clip_groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=True))
    tts = FakeTTS()

    assert await backchannel.prewarm(tts=tts, sample_rate=SAMPLE_RATE)
    assert tts.spoken == []


async def test_prewarm_takes_the_rate_from_the_service():
    """A configured service knows its own rate; making the caller repeat it invites drift."""
    cache = MemoryCache(prefilled=False)
    backchannel = Backchannel(clip_groups={"continue": ["Mhm."]}, cache=cache)

    assert await backchannel.prewarm(tts=FakeTTS(sample_rate=8000), budget_s=10)

    assert backchannel.clips.sample_rate == 8000
    assert ("Mhm.", 8000) in cache.stored


def test_sample_rate_of_reads_a_configured_service_before_it_starts():
    """`sample_rate` only reports after start(), so the configured value is all there is."""
    from pipecat.services.tts_service import TTSService

    from pipecat_backchannel.prewarm import sample_rate_of

    class Configured(TTSService):
        async def run_tts(self, text, context_id):
            yield None

    assert Configured(sample_rate=44100).sample_rate == 0  # not started yet
    assert sample_rate_of(Configured(sample_rate=44100)) == 44100
    assert sample_rate_of(Configured()) is None  # negotiates from the pipeline
