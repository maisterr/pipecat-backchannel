"""Where the backchannel processors go in a pipeline.

The two positions are not a preference — they are the only two that work, and
which one is which follows from what each processor needs. Making the caller
restate that on every pipeline is the library leaking its own invariant, so it
is worked out here instead, from the services already in the list.

Every failure raises. A backchannel that placed itself wrongly makes no sound
and reports nothing, which is indistinguishable from one that simply decided not
to fire — the single worst failure mode this library has.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

_MANUAL_HINT = (
    "Place them by hand instead, sharing one ClipLibrary between them:\n"
    "    from pipecat_backchannel.processor import BackchannelProcessor  # after your STT\n"
    "    from pipecat_backchannel.recorder import ClipRecorder           # after your TTS\n"
    "    from pipecat_backchannel.player import ClipPlayer               # then this"
)


class BackchannelPlacementError(RuntimeError):
    """Raised when the backchannel can't work out where to insert itself."""


def _last_index(processors: Sequence[FrameProcessor], kind: type) -> int | None:
    found = None
    for i, p in enumerate(processors):
        if isinstance(p, kind):
            found = i
    return found


class AutoPlacement:
    """Finds both positions from the services already in the pipeline.

    - The listener goes after the last ``STTService``, so it sees transcripts.
      With no STT — a speech-to-speech pipeline — it goes after the input
      transport, where it still has the audio it actually needs.
    - The speakers go after the last ``TTSService``, so they are downstream of
      real speech and never touch the TTS service's own state. With no TTS they
      go immediately before the output transport.
    """

    def insert(
        self,
        processors: Sequence[FrameProcessor],
        listener: FrameProcessor,
        speakers: Sequence[FrameProcessor],
    ) -> list[FrameProcessor]:
        """Return a new list with ``listener`` and ``speakers`` inserted.

        Args:
            processors: The pipeline as written by the caller. Not modified.
            listener: The processor that decides when to backchannel.
            speakers: The processors that turn that decision into audio, in the
                order they must appear.

        Returns:
            A new list.

        Raises:
            BackchannelPlacementError: No anchor found, or the positions come out
                in an order that can't work.
        """
        out = list(processors)

        stt_at = _last_index(out, STTService)
        if stt_at is None:
            stt_at = _last_index(out, BaseInputTransport)
        if stt_at is None:
            raise BackchannelPlacementError(
                "No STT service or input transport found in the pipeline, so there is "
                "nowhere to listen from. Anchors are only looked for at the top level — a "
                "service nested inside a ParallelPipeline is invisible here.\n" + _MANUAL_HINT
            )
        listener_at = stt_at + 1

        tts_at = _last_index(out, TTSService)
        if tts_at is not None:
            speakers_at = tts_at + 1
        else:
            output_at = _last_index(out, BaseOutputTransport)
            if output_at is None:
                raise BackchannelPlacementError(
                    "No TTS service or output transport found in the pipeline, so there is "
                    "nowhere to play clips. Anchors are only looked for at the top level — a "
                    "service nested inside a ParallelPipeline is invisible here.\n" + _MANUAL_HINT
                )
            speakers_at = output_at

        if listener_at > speakers_at:
            raise BackchannelPlacementError(
                f"The listening position (index {listener_at}) comes after the playing "
                f"position (index {speakers_at}), so the clip decision would never reach the "
                "player. Check the order of your STT and TTS services.\n" + _MANUAL_HINT
            )

        # Insert the later position first: it doesn't shift the earlier index.
        # When both land on the same index, this still leaves the listener first.
        out[speakers_at:speakers_at] = list(speakers)
        out.insert(listener_at, listener)
        return out
