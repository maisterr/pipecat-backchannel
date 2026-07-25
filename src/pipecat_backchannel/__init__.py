"""Backchannels for Pipecat voice agents.

Injects natural continuers ("mhm", "yeah", "right") while the user is still
talking, so the agent sounds like it's listening instead of waiting. Clips are
recorded once from your own TTS service and played from disk, so saying "mhm"
costs no LLM call and no network round trip.

Basic usage — no keys, no sample rates, no wiring::

    from pipecat_backchannel import Backchannel

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

This module holds what you need to *use* it. What you need to *change* it lives
one import deeper:

===================================  ========================================
``pipecat_backchannel.clips``        the inventory and how a clip is chosen
``pipecat_backchannel.synth``        producing clips without the pipeline's TTS
``pipecat_backchannel.cache``        where clips are kept between runs
``pipecat_backchannel.library``      what clips exist and what they sound like
``pipecat_backchannel.placement``    how the processors find their positions
``pipecat_backchannel.turn``         the end-of-turn model behind the gate
``pipecat_backchannel.processor``    the gate that decides when to fire
``pipecat_backchannel.player``       turning decisions into audio
``pipecat_backchannel.recorder``     recording clips from the pipeline's TTS
``pipecat_backchannel.frames``       the marker frame the two exchange
===================================  ========================================
"""

from pipecat_backchannel.plugin import Backchannel
from pipecat_backchannel.processor import BackchannelParams

__version__ = "0.2.0"

__all__ = [
    "Backchannel",
    "BackchannelParams",
]
