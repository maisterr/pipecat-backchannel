"""Frames emitted by the backchannel processor."""

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame


@dataclass
class PlayCachedClipFrame(DataFrame):
    """Tells the downstream :class:`~pipecat_backchannel.player.ClipPlayer` to
    play a pre-cached backchannel clip.

    Parameters:
        text: The clip text, used as the key into the loaded clip dict.
    """

    text: str
