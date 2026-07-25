"""Shared fakes and wiring helpers for the test suite.

The VAD and turn analyzer are injected as fakes so the gate can be driven
deterministically, without ONNX inference or real audio. The TTS service is a
fake too — the recorder's whole contract is "ask upstream, keep what comes
back", and that is testable without a vendor.

The fakes carry the full pipecat surface the real components expose
(``update_vad_start_secs``, VAD params, ...) — a test double that quietly
lacks the method under test is how a real bug once survived: the processor
read ``vad_start_secs`` from its params instead of from the injected VAD, and
no fake was positioned to notice.
"""

from __future__ import annotations

import asyncio

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.asyncio.task_manager import TaskManager

from pipecat_backchannel import BackchannelParams
from pipecat_backchannel.frames import PlayCachedClipFrame
from pipecat_backchannel.library import ClipLibrary
from pipecat_backchannel.processor import BackchannelProcessor

SAMPLE_RATE = 16000
CHUNK = b"\x00\x00" * 160  # 10ms of silence; the fake VAD ignores content
PCM = b"\x01\x02" * 10


class MemoryCache:
    """In-memory ClipCache. ``prefilled`` makes every lookup a hit."""

    def __init__(self, prefilled: bool = False):
        self.prefilled = prefilled
        self.stored: dict[tuple[str, int], bytes] = {}

    def get(self, text: str, sample_rate: int) -> bytes | None:
        if (text, sample_rate) in self.stored:
            return self.stored[(text, sample_rate)]
        return PCM if self.prefilled else None

    def put(self, text: str, sample_rate: int, pcm: bytes) -> None:
        self.stored[(text, sample_rate)] = pcm


class FakeVAD:
    """VAD whose state the test sets directly."""

    def __init__(self, *, start_secs: float = 0.2):
        self.state = VADState.QUIET
        self.sample_rate = None
        # The processor reads its VAD's own params rather than trusting
        # BackchannelParams to agree with an injected analyzer.
        self._params = VADParams(start_secs=start_secs, stop_secs=0.2)

    @property
    def params(self) -> VADParams:
        return self._params

    def set_sample_rate(self, sample_rate: int):
        self.sample_rate = sample_rate

    async def analyze_audio(self, buffer: bytes) -> VADState:
        return self.state


class FakeTurnAnalyzer(BaseTurnAnalyzer):
    """Turn analyzer that returns whatever the test asks for."""

    def __init__(self, state=EndOfTurnState.INCOMPLETE):
        super().__init__(sample_rate=SAMPLE_RATE)
        self.state = state
        self.appended = 0
        self.analyzed = 0
        self.cleared = 0
        self.vad_start_secs: float | None = None
        self._params = BaseTurnParams()

    @property
    def params(self) -> BaseTurnParams:
        return self._params

    @property
    def speech_triggered(self) -> bool:
        return True

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        self.appended += 1
        return EndOfTurnState.INCOMPLETE

    def update_vad_start_secs(self, vad_start_secs: float):
        self.vad_start_secs = vad_start_secs

    async def analyze_end_of_turn(self):
        self.analyzed += 1
        return self.state, None

    def clear(self):
        self.cleared += 1


class Sink(FrameProcessor):
    """Collects everything pushed downstream."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.frames.append(frame)


class FakeTTS(FrameProcessor):
    """Answers a TTSSpeakFrame the way a real TTS service does: audio downstream."""

    def __init__(self, *, chunks: int = 2, sample_rate: int = SAMPLE_RATE, answer: bool = True):
        super().__init__()
        self.chunks = chunks
        self.sample_rate = sample_rate
        self.answer = answer
        self.spoken: list[TTSSpeakFrame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame):
            self.spoken.append(frame)
            if self.answer:
                await self.push_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
                for _ in range(self.chunks):
                    await self.push_frame(
                        TTSAudioRawFrame(audio=PCM, sample_rate=self.sample_rate, num_channels=1),
                        FrameDirection.DOWNSTREAM,
                    )
                await self.push_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
            return

        await self.push_frame(frame, direction)


class FakeSTT(STTService):
    async def run_stt(self, audio: bytes):
        yield None


class FakeTTSService(TTSService):
    async def run_tts(self, text: str, context_id: str):
        yield None


def _transports():
    return BaseInputTransport(TransportParams()), BaseOutputTransport(TransportParams())


async def _setup(*processors: FrameProcessor):
    task_manager = TaskManager(loop=asyncio.get_running_loop())
    setup = FrameProcessorSetup(
        clock=SystemClock(), task_manager=task_manager, pipeline_worker=None
    )
    for p in processors:
        await p.setup(setup)
    for a, b in zip(processors, processors[1:]):
        a.link(b)


async def _wire(processor: FrameProcessor) -> Sink:
    """setup() + start() a processor standalone, with a sink attached."""
    sink = Sink()
    await _setup(processor, sink)
    await processor.queue_frame(
        StartFrame(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    return sink


async def _audio(processor, vad: FakeVAD, state: VADState, chunks: int = 1):
    """Push audio while the fake VAD reports ``state``."""
    vad.state = state
    for _ in range(chunks):
        await processor.queue_frame(
            InputAudioRawFrame(audio=CHUNK, sample_rate=SAMPLE_RATE, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
    await asyncio.sleep(0.05)


async def _speak(processor, vad: FakeVAD, chunks: int = 5):
    """Simulate the user talking."""
    await _audio(processor, vad, VADState.SPEAKING, chunks)


async def _pause(processor, vad: FakeVAD):
    """Simulate the user going quiet."""
    await _audio(processor, vad, VADState.QUIET)


def _fired(sink: Sink) -> list[PlayCachedClipFrame]:
    return [f for f in sink.frames if isinstance(f, PlayCachedClipFrame)]


async def _ready_library(**kwargs) -> ClipLibrary:
    library = ClipLibrary(cache=MemoryCache(prefilled=True), **kwargs)
    await library.load(SAMPLE_RATE)
    return library


async def _make(*, vad: FakeVAD | None = None, turn: FakeTurnAnalyzer | None = None, **kwargs):
    vad = vad or FakeVAD()
    turn = turn or FakeTurnAnalyzer(kwargs.pop("turn_state", EndOfTurnState.INCOMPLETE))
    tuning = {
        "min_speech_before_eligible_s": 0.0,
        "cooldown_s": 0.0,
        "fire_probability": 1.0,
    }
    processor = BackchannelProcessor(
        params=BackchannelParams(**(tuning | kwargs)),
        clips=await _ready_library(),
        vad_analyzer=vad,
        turn_analyzer=turn,
    )
    return processor, vad, turn
