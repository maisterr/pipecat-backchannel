"""The plugin and placement: one object, called per pipeline.

Covers where the processors land in the list, what one ``Backchannel`` shares
between pipelines, and the package-level promises (vendor-free core, typed,
small public surface).
"""

from pathlib import Path

import pytest

import pipecat_backchannel
from conftest import (
    PCM,
    SAMPLE_RATE,
    FakeSTT,
    FakeTTS,
    FakeTTSService,
    MemoryCache,
    Sink,
    _transports,
)
from pipecat_backchannel import Backchannel
from pipecat_backchannel.placement import AutoPlacement, BackchannelPlacementError
from pipecat_backchannel.player import ClipPlayer
from pipecat_backchannel.processor import BackchannelProcessor
from pipecat_backchannel.recorder import ClipRecorder


# --------------------------------------------------------------------- placement


def test_placement_puts_listener_after_stt_and_speakers_after_tts():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    listener, player = Sink(), Sink()

    out = AutoPlacement().insert([tin, stt, tts, tout], listener, [player])

    assert out == [tin, stt, listener, tts, player, tout]


def test_placement_uses_the_last_service_when_there_are_several():
    tin, tout = _transports()
    stt_a, stt_b, tts = FakeSTT(), FakeSTT(), FakeTTSService()
    listener, player = Sink(), Sink()

    out = AutoPlacement().insert([tin, stt_a, stt_b, tts, tout], listener, [player])

    assert out.index(listener) == out.index(stt_b) + 1


def test_placement_keeps_speaker_order():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    listener, recorder, player = Sink(), Sink(), Sink()

    out = AutoPlacement().insert([tin, stt, tts, tout], listener, [recorder, player])

    assert out == [tin, stt, listener, tts, recorder, player, tout]


def test_placement_falls_back_to_the_transports_without_stt_or_tts():
    tin, tout = _transports()
    listener, player = Sink(), Sink()

    out = AutoPlacement().insert([tin, tout], listener, [player])

    assert out == [tin, listener, player, tout]


def test_placement_raises_without_an_input_anchor():
    _, tout = _transports()
    with pytest.raises(BackchannelPlacementError, match="nowhere to listen"):
        AutoPlacement().insert([tout], Sink(), [Sink()])


def test_placement_raises_without_an_output_anchor():
    tin, _ = _transports()
    with pytest.raises(BackchannelPlacementError, match="nowhere to play"):
        AutoPlacement().insert([tin], Sink(), [Sink()])


def test_placement_raises_when_the_order_cannot_work():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    with pytest.raises(BackchannelPlacementError, match="comes after"):
        AutoPlacement().insert([tin, tts, tout, stt], Sink(), [Sink()])


# ------------------------------------------------------------------------ plugin


def test_plugin_inserts_listener_recorder_and_player():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()

    out = Backchannel()([tin, stt, tts, tout])

    kinds = [type(p).__name__ for p in out]
    assert kinds == [
        "BaseInputTransport",
        "FakeSTT",
        "BackchannelProcessor",
        "FakeTTSService",
        "ClipRecorder",
        "ClipPlayer",
        "BaseOutputTransport",
    ]


def test_plugin_skips_the_recorder_when_a_synthesizer_is_given():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()

    async def synth(text: str, sample_rate: int) -> bytes:
        return PCM

    out = Backchannel(synthesizer=synth)([tin, stt, tts, tout])

    assert not any(isinstance(p, ClipRecorder) for p in out)
    assert sum(isinstance(p, ClipPlayer) for p in out) == 1


def test_plugin_does_not_mutate_the_list_it_is_given():
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    original = [tin, stt, tts, tout]

    Backchannel()(list(original))

    assert original == [tin, stt, tts, tout]


def test_plugin_serves_many_pipelines_from_one_instance():
    """Built once per process: fresh processors per pipeline, one shared library."""
    tin, tout = _transports()
    stt, tts = FakeSTT(), FakeTTSService()
    backchannel = Backchannel()

    first = backchannel([tin, stt, tts, tout])
    second = backchannel([tin, stt, tts, tout])

    inserted_first = [p for p in first if p not in (tin, stt, tts, tout)]
    inserted_second = [p for p in second if p not in (tin, stt, tts, tout)]
    assert len(inserted_first) == 3
    # No processor is reused — they hold per-conversation state.
    assert not set(map(id, inserted_first)) & set(map(id, inserted_second))
    # ...but the expensive parts are shared: one clip library, one selector
    # (stateless, and the default one loads a WordNet corpus at build time).
    for p in inserted_first + inserted_second:
        assert p._clips is backchannel.clips
    gates = [
        p
        for p in inserted_first + inserted_second
        if isinstance(p, BackchannelProcessor)
    ]
    assert gates[0]._selector is gates[1]._selector
