"""Turn analyzers for the backchannel gate.

The backchannel needs its *own* turn analyzer — it cannot share the one
``LLMUserAggregator`` uses. That instance is not reentrant: both consumers would
call ``append_audio()`` into a single buffer (doubling every chunk), both would
write ``update_vad_start_secs()`` with their own VAD's value, and
``analyze_end_of_turn()`` wipes the buffer whenever it returns ``COMPLETE`` —
so a backchannel asking at a 0.2s pause would destroy the audio segment the
aggregator still needs for the real end-of-turn decision.

What *can* be shared is the model itself. Two analyzers need two sets of
buffers, but only one set of weights, and an ONNX Runtime session is safe to
call concurrently. :func:`shared_turn_analyzer` gives each pipeline its own
analyzer over one process-wide session, so N concurrent calls cost one model in
memory instead of N.

Measured on pipecat 1.6.0: identical verdicts to unshared analyzers, and no
concurrency penalty (8 concurrent inferences: 223ms unshared, 225ms shared).

What this saves is **memory**, and only that. Inference is unchanged — the two
analyzers are asked about different audio at different moments, so the model
runs just as often either way. Startup is unchanged too: pipecat builds the
session inside ``__init__`` before there is anything to graft onto, so each
analyzer still spends ~30ms loading a session that is then discarded. Avoiding
that would mean intercepting ONNX Runtime globally, which is not worth 30ms of
per-session setup that happens off the conversation's critical path.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

#: One ONNX Runtime session per distinct analyzer configuration, for the life of
#: the process. Never invalidated: the weights are read-only.
_sessions: dict[str, Any] = {}


def _config_key(kwargs: dict) -> str:
    """Identify an analyzer configuration.

    Keyed on the rendered arguments rather than the arguments themselves,
    because the interesting ones aren't hashable: ``SmartTurnParams`` is a
    pydantic model, so a tuple of kwargs raises ``TypeError`` the moment anyone
    configures the analyzer at all.
    """
    return repr(sorted((name, repr(value)) for name, value in kwargs.items()))


def shared_turn_analyzer(**kwargs) -> BaseTurnAnalyzer:
    """Build a ``LocalSmartTurnAnalyzerV3`` that reuses one process-wide model.

    Fresh buffers, shared weights. Behaves exactly like constructing the
    analyzer directly, except only one copy of the model stays resident however
    many pipelines are running.

    Args:
        **kwargs: Passed straight to ``LocalSmartTurnAnalyzerV3``. Analyzers
            built with different arguments get different sessions, since they
            may not be the same model.

    Returns:
        A turn analyzer with its own state.
    """
    analyzer = LocalSmartTurnAnalyzerV3(**kwargs)

    # Reaching into the analyzer's session is the only route pipecat offers —
    # LocalSmartTurnAnalyzerV3 builds it in __init__ from a path and exposes no
    # seam. If a future version renames it, every analyzer simply keeps its own
    # weights: more memory, identical behaviour.
    session = getattr(analyzer, "_session", None)
    if session is None:
        logger.debug(
            "Backchannel: this pipecat version exposes no ONNX session on "
            f"{type(analyzer).__name__}; the turn model will be loaded per pipeline."
        )
        return analyzer

    key = _config_key(kwargs)
    cached = _sessions.get(key)
    if cached is None:
        _sessions[key] = session
    else:
        analyzer._session = cached

    return analyzer
