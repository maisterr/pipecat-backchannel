"""The one thing you configure, and the one thing you call."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.processors.frame_processor import FrameProcessor

from pipecat_backchannel.cache import ClipCache
from pipecat_backchannel.clips import ClipSelector, HeuristicClipSelector
from pipecat_backchannel.library import ClipLibrary
from pipecat_backchannel.placement import AutoPlacement
from pipecat_backchannel.player import ClipPlayer
from pipecat_backchannel.prewarm import prewarm_library
from pipecat_backchannel.processor import BackchannelParams, BackchannelProcessor
from pipecat_backchannel.recorder import ClipRecorder
from pipecat_backchannel.synth import ClipSynthesizer
from pipecat_backchannel.turn import shared_turn_analyzer


class Backchannel:
    """Adds backchannels to a pipeline.

    Build one, call it with your processor list, and use what comes back::

        backchannel = Backchannel()

        pipeline = Pipeline(backchannel([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]))

    With no arguments the clips are recorded from your own TTS service, once,
    and cached — so the voice matches, and there is nothing to configure. Every
    argument below replaces one part of that.

    **Build one per process, not per session.** It holds configuration and the
    clip library, both of which are worth sharing; each call builds fresh
    processors for that pipeline, which is where the per-conversation state
    lives.
    """

    def __init__(
        self,
        *,
        params: BackchannelParams | None = None,
        clip_groups: Mapping[str, Sequence[str]] | None = None,
        selector: ClipSelector | None = None,
        turn_factory: Callable[[], BaseTurnAnalyzer] | None = None,
        synthesizer: ClipSynthesizer | None = None,
        cache: ClipCache | None = None,
        volume: float = 0.6,
        placement: AutoPlacement | None = None,
    ):
        """Initialize the backchannel.

        Args:
            params: Timing and frequency of the gate. Defaults to
                :class:`~pipecat_backchannel.processor.BackchannelParams`.
            clip_groups: The clip inventory, grouped by conversational function.
            selector: Which clip to play at a given moment. Defaults to
                :class:`~pipecat_backchannel.clips.HeuristicClipSelector`, built
                here and shared by every pipeline — selectors are stateless, and
                the default one loads a WordNet corpus the first time.
            turn_factory: Builds the end-of-turn classifier for each pipeline.
                Defaults to :func:`~pipecat_backchannel.turn.shared_turn_analyzer`,
                which gives every pipeline its own analyzer over one
                process-wide ONNX model. A *factory*, not an instance: the
                analyzer holds per-conversation audio state and cannot be shared
                between concurrent calls.
            synthesizer: Produces clips instead of recording them from a TTS
                service. Only needed to use a different voice than the bot's own.
            cache: Where clips are kept between runs. Defaults to
                :class:`~pipecat_backchannel.cache.FileClipCache`.
            volume: How loud the clips play, relative to the recording. Below
                1.0 by default, because a backchannel belongs underneath the
                person who still has the floor.
            placement: How the processors find their positions in the pipeline.
        """
        self._library = ClipLibrary(groups=clip_groups, synthesizer=synthesizer, cache=cache)
        self._volume = volume
        self._placement = placement or AutoPlacement()
        self._params = params
        # Stateless, so one instance serves every pipeline — and building it here
        # pays the default selector's one-time WordNet load at construction,
        # rather than inside whichever session happens to be first.
        self._selector = selector or HeuristicClipSelector()
        self._turn_factory = turn_factory or shared_turn_analyzer
        # Clips can fill themselves from a synthesizer; without one they have to
        # be recorded from a TTS service, which needs a processor in the pipeline.
        self._needs_recorder = self._library.synthesizer is None

    @property
    def clips(self) -> ClipLibrary:
        """The clip library shared by every pipeline this is added to."""
        return self._library

    async def prewarm(
        self, *, tts, sample_rate: int | None = None, budget_s: float = 120.0
    ) -> bool:
        """Record the clips now, so no session has to.

        Entirely optional. Without it the first session records them itself,
        held in startup while it does — correct, but a wait for whoever connects
        first. Call this at app startup to move that cost out of the request
        path. Returns straight away once the clips are cached, so it is cheap to
        call unconditionally.

        Args:
            tts: A pipecat ``TTSService``, used and then shut down. Build a
                throwaway instance for this — do not pass one a session will use.
            sample_rate: Rate to record at. Taken from ``tts`` when omitted, so
                the clips match what that service produces. Only pass it when
                the service negotiates its rate from the pipeline instead of
                being configured with one.
            budget_s: Hard ceiling, so a misbehaving service can't hang startup.

        Returns:
            Whether every clip ended up with audio.
        """
        return await prewarm_library(
            self._library, tts=tts, sample_rate=sample_rate, budget_s=budget_s
        )

    def __call__(self, processors: Sequence[FrameProcessor]) -> list[FrameProcessor]:
        """Return a new processor list with a backchannel inserted.

        Call this once per pipeline. Each call builds its own processors — they
        hold per-conversation state — over the one shared clip library, so the
        clips are loaded once per process however many sessions you run.

        Args:
            processors: Your pipeline, as you'd otherwise pass it to
                ``Pipeline``. Not modified.

        Returns:
            A new list, ready for ``Pipeline(...)``.

        Raises:
            ~pipecat_backchannel.placement.BackchannelPlacementError: The
                positions couldn't be worked out from the list.
        """
        listener = BackchannelProcessor(
            params=self._params,
            clips=self._library,
            selector=self._selector,
            turn_analyzer=self._turn_factory(),
        )
        speakers: list[FrameProcessor] = []
        if self._needs_recorder:
            speakers.append(ClipRecorder(clips=self._library))
        speakers.append(ClipPlayer(clips=self._library, volume=self._volume))

        return self._placement.insert(processors, listener, speakers)
