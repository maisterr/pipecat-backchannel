"""The clips: what plays, how loud, and which one fits the moment.

Covers the player (audio out, TTS-shaped, never a turn), the library (readiness
and rate discipline), the on-disk cache, and the selector heuristics.
"""

import array
import asyncio

import pytest
from loguru import logger
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from conftest import (
    PCM,
    SAMPLE_RATE,
    FakeSTT,
    FakeTTSService,
    MemoryCache,
    _ready_library,
    _transports,
    _wire,
)
from pipecat_backchannel import Backchannel
from pipecat_backchannel.cache import FileClipCache, clip_filename
from pipecat_backchannel.clips import (
    DEFAULT_CLIP_GROUPS,
    HeuristicClipSelector,
    flatten_clips,
)
from pipecat_backchannel.frames import PlayCachedClipFrame
from pipecat_backchannel.library import ClipLibrary
from pipecat_backchannel.player import ClipPlayer


def _samples(pcm: bytes) -> list[int]:
    values = array.array("h")
    values.frombytes(pcm)
    return list(values)


# ------------------------------------------------------------------------ player


async def test_clip_player_emits_tts_shape():
    library = await _ready_library()
    player = ClipPlayer(clips=library, volume=1.0)  # level is its own test, below
    sink = await _wire(player)

    await player.queue_frame(PlayCachedClipFrame("Mhm."), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)

    kinds = [type(f) for f in sink.frames]
    assert TTSStartedFrame in kinds
    assert TTSStoppedFrame in kinds
    audio = [f for f in sink.frames if isinstance(f, TTSAudioRawFrame)]
    assert audio[0].audio == PCM
    assert audio[0].sample_rate == SAMPLE_RATE
    # The marker itself must not leak downstream.
    assert not any(isinstance(f, PlayCachedClipFrame) for f in sink.frames)


async def test_clip_player_loads_at_the_pipeline_sample_rate():
    library = ClipLibrary(cache=MemoryCache(prefilled=True))
    player = ClipPlayer(clips=library)
    await _wire(player)

    assert library.sample_rate == SAMPLE_RATE
    assert library.ready


async def test_clip_player_plays_clips_quieter_than_the_bot():
    """A real backchannel sits under the speaker, not level with them."""
    player = ClipPlayer(clips=await _ready_library(), volume=0.5)
    sink = await _wire(player)

    await player.queue_frame(PlayCachedClipFrame("Mhm."), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)

    played = next(f for f in sink.frames if isinstance(f, TTSAudioRawFrame))
    assert _samples(played.audio) == [s // 2 for s in _samples(PCM)]


async def test_clips_are_quieter_than_the_bot_by_default():
    assert ClipPlayer()._volume < 1.0


async def test_clip_player_clamps_instead_of_wrapping_around():
    """Overflowing a 16-bit sample wraps to full negative — an audible click."""
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=MemoryCache(prefilled=False))
    await library.load(SAMPLE_RATE)
    library.store("Mhm.", array.array("h", [30000] * 10).tobytes())

    player = ClipPlayer(clips=library, volume=2.0)
    sink = await _wire(player)

    await player.queue_frame(PlayCachedClipFrame("Mhm."), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)

    played = next(f for f in sink.frames if isinstance(f, TTSAudioRawFrame))
    assert _samples(played.audio) == [32767] * 10


async def test_clip_player_warns_instead_of_dropping_silently():
    """Silence here is indistinguishable from the gate declining to fire."""
    warnings: list[str] = []
    handler = logger.add(lambda m: warnings.append(m), level="WARNING")
    try:
        library = await _ready_library()
        player = ClipPlayer(clips=library)
        sink = await _wire(player)

        await player.queue_frame(PlayCachedClipFrame("Never recorded."), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.05)
    finally:
        logger.remove(handler)

    assert not any(isinstance(f, TTSAudioRawFrame) for f in sink.frames)
    assert any("Never recorded." in w for w in warnings)


async def test_a_clip_never_opens_an_assistant_turn():
    """``TTSStartedFrame.append_to_context`` defaults to True.

    Left at the default, the assistant aggregator opens an assistant turn for
    every clip played — the bot recorded as having taken the floor. The user
    talking on then counts as barging in on that turn, so each backchannel
    manufactured an interruption out of nothing.
    """
    player = ClipPlayer(clips=await _ready_library())
    sink = await _wire(player)

    await player.queue_frame(PlayCachedClipFrame("Mhm."), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)

    started = next(f for f in sink.frames if isinstance(f, TTSStartedFrame))
    assert started.append_to_context is False


def test_plugin_passes_the_volume_through_to_the_player():
    tin, tout = _transports()
    out = Backchannel(volume=0.25)([tin, FakeSTT(), FakeTTSService(), tout])

    player = next(p for p in out if isinstance(p, ClipPlayer))
    assert player._volume == 0.25


# ------------------------------------------------------------------------ library


async def test_library_is_ready_only_when_every_clip_has_audio():
    library = ClipLibrary(
        groups={"continue": ["Mhm.", "Yeah."]}, cache=MemoryCache(prefilled=False)
    )
    await library.load(SAMPLE_RATE)
    assert not library.ready
    assert library.missing() == ["Mhm.", "Yeah."]

    library.store("Mhm.", PCM)
    assert not library.ready

    library.store("Yeah.", PCM)
    assert library.ready


async def test_library_synthesizes_misses_once():
    calls = []

    async def synth(text: str, sample_rate: int) -> bytes:
        calls.append(text)
        return PCM

    library = ClipLibrary(
        groups={"continue": ["Mhm."]}, synthesizer=synth, cache=MemoryCache(prefilled=False)
    )
    await library.load(SAMPLE_RATE)
    await library.load(SAMPLE_RATE)

    assert calls == ["Mhm."]
    assert library.ready


async def test_library_reloads_when_the_sample_rate_changes():
    cache = MemoryCache(prefilled=False)
    library = ClipLibrary(groups={"continue": ["Mhm."]}, cache=cache)
    await library.load(SAMPLE_RATE)
    library.store("Mhm.", PCM)

    await library.load(24000)

    assert library.sample_rate == 24000
    assert not library.ready  # 16kHz audio must never be played back as 24kHz


async def test_library_rejects_a_clip_outside_its_inventory():
    library = await _ready_library(groups={"continue": ["Mhm."]})
    with pytest.raises(KeyError):
        library.store("Not mine.", PCM)


def test_library_rejects_an_empty_inventory():
    with pytest.raises(ValueError, match="empty"):
        ClipLibrary(groups={})
    with pytest.raises(ValueError, match="empty"):
        ClipLibrary(groups={"continue": []})
    with pytest.raises(ValueError, match="empty clip text"):
        ClipLibrary(groups={"continue": ["  "]})


# -------------------------------------------------------------------------- cache


def test_file_cache_round_trips(tmp_path):
    cache = FileClipCache(tmp_path)
    assert cache.get("Mhm.", SAMPLE_RATE) is None

    cache.put("Mhm.", SAMPLE_RATE, PCM)

    assert cache.get("Mhm.", SAMPLE_RATE) == PCM
    assert cache.get("Mhm.", 24000) is None  # rate is part of the key
    assert (tmp_path / clip_filename("Mhm.", SAMPLE_RATE)).exists()


def test_file_cache_keeps_prosody_variants_apart(tmp_path):
    """"Hm.", "Hm," and "Hm..." are three clips, not one filename.

    The readable part of the name drops punctuation — which is exactly what
    distinguishes these clips. Colliding them makes every variant come back
    from disk sounding like whichever one was recorded last.
    """
    cache = FileClipCache(tmp_path)
    variants = {"Hm.": PCM, "Hm,": b"\x03\x04" * 10, "Hm...": b"\x05\x06" * 10}

    for text, pcm in variants.items():
        cache.put(text, SAMPLE_RATE, pcm)

    for text, pcm in variants.items():
        assert cache.get(text, SAMPLE_RATE) == pcm


# ----------------------------------------------------------------------- selector


def test_selector_routes_by_transcript():
    selector = HeuristicClipSelector(expand_synonyms=False)
    assert selector.pick_group("we need the, uh...") == "thinking"
    assert selector.pick_group("it turns out we shipped it") == "surprise"


def test_selector_avoids_recent_clips():
    selector = HeuristicClipSelector(expand_synonyms=False)
    groups = {"thinking": ["Hmm.", "Hm."]}
    clip = selector.select("uh, like", groups, recent=["Hmm."])
    assert clip == "Hm."


def test_selector_falls_back_for_custom_inventory():
    selector = HeuristicClipSelector(expand_synonyms=False)
    groups = {"only": ["Right."]}
    assert selector.select("uh, like", groups, recent=[]) == "Right."


def test_selector_never_says_the_same_thing_twice_in_a_row():
    """Once the group is exhausted the selector widens — but never onto the last clip.

    Saying "Yeah. ... Yeah." back to back is the one repetition a listener
    actually notices; two clips apart nobody hears.
    """
    selector = HeuristicClipSelector(expand_synonyms=False)
    groups = {"only": ["Mhm.", "Yeah."]}

    for _ in range(50):
        assert selector.select("", groups, recent=["Mhm.", "Yeah."]) == "Mhm."


def test_selector_repeats_only_when_the_group_holds_nothing_else():
    selector = HeuristicClipSelector(expand_synonyms=False)
    groups = {"only": ["Mhm."]}

    assert selector.select("", groups, recent=["Mhm."]) == "Mhm."


def test_every_default_group_can_avoid_repeating_itself():
    """A one-clip group repeats by construction, whatever the selector does."""
    thin = {name: clips for name, clips in DEFAULT_CLIP_GROUPS.items() if len(clips) < 2}
    assert thin == {}


def test_flatten_clips_dedupes():
    groups = {"a": ["Mhm.", "Yeah."], "b": ["Yeah.", "Yep."]}
    assert flatten_clips(groups) == ["Mhm.", "Yeah.", "Yep."]
