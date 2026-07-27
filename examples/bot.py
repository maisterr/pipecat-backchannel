import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cerebras.llm import CerebrasLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_backchannel import Backchannel, BackchannelParams

# Relative to this file, not the shell's cwd, so the example runs from anywhere.
load_dotenv()


def make_tts() -> CartesiaTTSService:
    return CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        sample_rate=24000,
        settings=CartesiaTTSService.Settings(
            voice=os.getenv(
                "CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"
            ),
            model="sonic-3.5",
        ),
    )


# One per process, prewarmed below and then called for every session. Sharing the
# object rather than building a second one is what keeps the clip inventory, the
# cache and the sample rate identical between the two — separate instances only
# ever agree by both happening to find the same files on disk.
backchannel = Backchannel(
    volume=0.6,
    params=BackchannelParams(
        min_speech_before_eligible_s=1.5, fire_probability=0.6, cooldown_s=6
    ),
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = make_tts()

    llm = CerebrasLLMService(
        api_key=os.getenv("CEREBRAS_API_KEY"),
        settings=CerebrasLLMService.Settings(
            model=os.getenv("CEREBRAS_MODEL", "gemma-4-31b"),
            system_instruction=(
                "You are a friendly, concise English-speaking voice assistant. Your "
                "responses will be spoken aloud, so avoid emojis, bullet points, or "
                "other formatting that can't be spoken. Keep replies short and "
                "conversational, the way people actually talk out loud."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    start_secs=float(os.getenv("MAIN_VAD_START_SECS", "0.4")),
                    stop_secs=float(os.getenv("MAIN_VAD_STOP_SECS", "0.4")),
                )
            )
        ),
    )

    pipeline = Pipeline(
        backchannel(
            [
                transport.input(),
                stt,
                user_aggregator,
                llm,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        )
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message(
            {"role": "developer", "content": "Start by concisely introducing yourself."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    # Record the clips before the server accepts a single connection, so no
    # caller ever waits for them, at whatever rate make_tts() is configured for.
    # Returns straight away once .clip_cache/ is populated, so this only costs
    # anything on the very first run. Every session then plays them from memory,
    # since run_bot uses this same object.
    asyncio.run(backchannel.prewarm(tts=make_tts()))

    main()
