"""Record the clips once, at application startup.

Used by :meth:`~pipecat_backchannel.plugin.Backchannel.prewarm`, which is how
you normally reach this. Call it directly only when driving a
:class:`~pipecat_backchannel.library.ClipLibrary` without a ``Backchannel``.

The in-pipeline :class:`~pipecat_backchannel.recorder.ClipRecorder` holds the
first session open while it records, which is correct but pays the cost inside a
conversation someone is waiting on. Recording from your app's
startup instead moves that cost before the server accepts anything at all:
sessions then find the clips already on disk and start instantly.

It works by running a throwaway two-processor pipeline — your TTS service and a
recorder — which is the same mechanism the in-pipeline path uses, just without a
conversation attached. Give it a TTS instance built for this and thrown away
afterwards, not the one a session will use: recording ends by tearing the service
down.
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.workers.runner import WorkerRunner

from pipecat_backchannel.library import ClipLibrary
from pipecat_backchannel.recorder import ClipRecorder


#: Used only when a service has no configured rate to read. Pipecat's own
#: default output rate, so clips match what an unconfigured pipeline produces.
DEFAULT_SAMPLE_RATE = 24000


def sample_rate_of(tts) -> int | None:
    """Return the rate a TTS service is configured to output at, if it says.

    A service only reports its rate through ``sample_rate`` once the pipeline
    has started it, so before that the configured value is the only thing to go
    on. Returns ``None`` when the service was left to negotiate its rate from
    the pipeline, since there is no pipeline here to negotiate with.
    """
    for candidate in (getattr(tts, "sample_rate", None), getattr(tts, "_init_sample_rate", None)):
        if candidate:
            return int(candidate)
    return None


async def prewarm_library(
    clips: ClipLibrary,
    *,
    tts,
    sample_rate: int | None = None,
    budget_s: float = 120.0,
) -> bool:
    """Fill ``clips`` from ``tts`` now, so no session has to.

    Returns immediately if the clips are already cached, which is every run after
    the first, so this is cheap to call unconditionally at startup.

    Args:
        clips: The library to fill. Pass the same one to ``Backchannel``.
        tts: A pipecat ``TTSService``, used and then shut down. Build a
            throwaway instance for this — do not pass one a session will use.
        sample_rate: Output rate for the throwaway pipeline. Taken from ``tts``
            when omitted, so the clips match what that service produces. Only
            pass it when the service negotiates its rate from the pipeline and
            therefore can't say.
        budget_s: Hard ceiling. Startup continues regardless after this, with
            backchannels off, rather than hanging on a misbehaving service.

    Returns:
        Whether the library ended up with audio for every clip.
    """
    if sample_rate is None:
        sample_rate = sample_rate_of(tts)
        if sample_rate is None:
            sample_rate = DEFAULT_SAMPLE_RATE
            logger.debug(
                f"Backchannel: {type(tts).__name__} has no configured sample rate; "
                f"recording clips at {DEFAULT_SAMPLE_RATE}Hz."
            )

    await clips.load(sample_rate)
    if clips.ready:
        return True

    recorder = ClipRecorder(clips=clips, budget_s=budget_s)
    worker = PipelineWorker(
        Pipeline([tts, recorder]),
        params=PipelineParams(audio_out_sample_rate=sample_rate),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    running = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(recorder.wait_until_done(), timeout=budget_s + 10)
    except TimeoutError:
        logger.warning("Backchannel: prewarm timed out; starting without backchannels")
    finally:
        await worker.cancel()
        await running

    return clips.ready
