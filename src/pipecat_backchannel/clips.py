"""Backchannel clip inventory and clip-choice strategies.

Clip *choice* is a separate, much lower-stakes decision than the fire/no-fire
gate in :mod:`pipecat_backchannel.processor`: which flavor of "mhm" fits the
last thing the user said, and don't repeat the same one twice in a row. A wrong
guess here just sounds slightly off; it never affects *whether* a clip plays.

Clips are grouped by conversational function. The default
:class:`HeuristicClipSelector` picks a group with two regexes over the latest
ASR partial, then avoids recently-played clips within that group.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from loguru import logger

#: Clips grouped by conversational function. Pass a modified copy as
#: ``Backchannel(clip_groups=...)`` to change the inventory; the audio for
#: whatever you list is produced at startup.
#:
#: Real backchannels are a small, closed set of vocalizations — *uh-huh, yeah,
#: mm-hmm, right, okay, oh, huh, hm* covers nearly all of them. Assessments like
#: "absolutely" or "makes sense" read as backchannels on the page but in real
#: speech they arrive at turn boundaries, not over someone mid-sentence, and a
#: bot that says them while you're still talking sounds like it's answering.
#:
#: So variety comes from **prosody, not vocabulary**: the same few sounds spelled
#: and punctuated differently, because that is the handle a TTS gives you.
#: ``"Mhm."`` falls, ``"Mhm,"`` stays up and hands the floor back, ``"Hm..."``
#: draws out. One sound, three meanings.
#:
#: Every group needs at least two clips, or it repeats itself no matter what the
#: selector does.
DEFAULT_CLIP_GROUPS: dict[str, list[str]] = {
    # Plain continuer — the safe default, fits almost any mid-thought pause.
    "continue": [
        "Mhm.",
        "Mhm,",
        "Mm-hmm.",
        "Mm-hmm,",
        "Mm.",
        "Mm,",
        "Okay.",
    ],
    # Agreement/confirmation — fits when they just stated something concrete.
    "affirm": [
        "Yeah.",
        "Yeah,",
        "Yep.",
        "Yup.",
        "Uh-huh, yeah.",
        "Yeah, yeah.",
        "Okay, yeah.",
    ],
    # Hesitation-empathy — fits when they're visibly searching for words. The
    # trailing ellipsis is doing the work: it keeps the sound unresolved, which
    # is what makes it read as thinking along rather than answering.
    "thinking": [
        "Hmm.",
        "Hm.",
        "Hm...",
        "Hmm...",
        "Mm...",
        "Yeah...",
        "Right...",
    ],
    # Realization/surprise — fits when they're introducing something notable.
    # Kept mild on purpose: "wow" and "no way" are strong enough to read as
    # taking the floor, which is the one thing a backchannel must never do.
    "surprise": [
        "Oh.",
        "Oh yeah.",
        "Oh, yeah?",
        "Oh really.",
        "Really.",
        "Huh.",
    ],
}


def flatten_clips(groups: Mapping[str, Sequence[str]]) -> list[str]:
    """Flatten grouped clips into the deduplicated list of texts to synthesize.

    Args:
        groups: Clip groups, e.g. :data:`DEFAULT_CLIP_GROUPS`.

    Returns:
        Every clip text across all groups, in order, without duplicates.
    """
    seen: dict[str, None] = {}
    for group in groups.values():
        for clip in group:
            seen.setdefault(clip, None)
    return list(seen)


@runtime_checkable
class ClipSelector(Protocol):
    """Chooses which clip to play for a given moment."""

    def select(
        self,
        text: str,
        groups: Mapping[str, Sequence[str]],
        recent: Sequence[str],
    ) -> str:
        """Pick a clip.

        Called at the moment the clip fires, so implementations must not do
        network or disk I/O here.

        Args:
            text: The latest ASR partial for the user's current turn ("" if none).
            groups: The processor's clip groups.
            recent: Recently played clips, most recent last. Avoid these where
                you can, and avoid ``recent[-1]`` unless the group holds nothing
                else — saying the same thing twice in a row is the only
                repetition a listener reliably notices.

        Returns:
            A clip text present in ``groups``.
        """
        ...


# Hesitation markers are discourse fillers ("um", "like", "I mean"), not content
# words — WordNet has no synonym set for them, so this stays a hand list rather
# than a library expansion.
_HESITATION_RE = re.compile(
    r"\b(um+|uh+|like|you know|i mean|i don'?t know|not sure|kind of|sort of)\b", re.I
)

_SURPRISE_SEED_WORDS = [
    "surprising",
    "shocking",
    "unexpected",
    "astonishing",
    "remarkable",
]
_SURPRISE_MARKERS = {
    "turns out",
    "apparently",
    "wow",
    "no way",
    "really",
    "honestly",
}
# WordNet's similar-adjective graph drifts into unrelated territory from these
# seeds (e.g. "outrage", "scandalous", "disgraceful" — shock-adjacent, not
# surprise). Automatic expansion still needs this manual denylist.
_SURPRISE_DENYLIST = {
    "storm",
    "floor",
    "disgraceful",
    "appal",
    "appall",
    "offend",
    "outrage",
    "scandalise",
    "scandalize",
    "scandalous",
    "shameful",
    "traumatise",
    "traumatize",
    "upset",
    "unthought",
    "unthought-of",
    "unprovided for",
    "unhoped",
    "lurid",
    "singular",
}


def expand_via_wordnet(
    seed_words: Sequence[str], denylist: set[str] = _SURPRISE_DENYLIST
) -> set[str]:
    """Expand content-word seeds through WordNet's synonym/similarity graph.

    Only worth doing for real content words with real synonym sets. On the first
    call this downloads the WordNet corpus (~10MB, cached under ``~/nltk_data/``);
    if that fails (offline, sandboxed), it falls back to the seeds alone rather
    than raising.

    Args:
        seed_words: Words to expand from.
        denylist: Words to drop from the result — WordNet drifts, and expansion
            is not fire-and-forget.

    Returns:
        The expanded word set, minus ``denylist``.
    """
    try:
        from nltk.corpus import wordnet as wn

        try:
            wn.synsets("test")
        except LookupError:
            import nltk

            nltk.download("wordnet", quiet=True)

        words = set()
        for seed in seed_words:
            for synset in wn.synsets(seed):
                for lemma in synset.lemmas():
                    words.add(lemma.name().replace("_", " ").lower())
                if synset.pos() == "a":
                    for similar in synset.similar_tos():
                        for lemma in similar.lemmas():
                            words.add(lemma.name().replace("_", " ").lower())
        return words - denylist
    except Exception as e:
        logger.warning(f"WordNet expansion unavailable ({e}), using seed words only")
        return set(seed_words)


class HeuristicClipSelector:
    """Picks a clip group with two regexes over the latest ASR partial.

    A rough proxy for the real problem (which filler fits this context). Good
    enough because the stakes are low — see the module docstring.
    """

    def __init__(
        self, *, affirm_probability: float = 0.4, expand_synonyms: bool = True
    ):
        """Initialize the selector.

        Any synonym expansion happens here, at construction time — never during
        :meth:`select`, which runs at the moment the clip fires.

        Args:
            affirm_probability: Chance of "affirm" over "continue" when neither
                regex matches.
            expand_synonyms: Expand the surprise vocabulary via WordNet. Costs a
                one-time ~10MB corpus download on first use; set ``False`` to
                use the seed words alone.
        """
        self._affirm_probability = affirm_probability
        surprise_words = set(_SURPRISE_SEED_WORDS)
        if expand_synonyms:
            surprise_words = expand_via_wordnet(_SURPRISE_SEED_WORDS)
        surprise_words |= _SURPRISE_MARKERS
        self._surprise_re = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in sorted(surprise_words)) + r")\b",
            re.I,
        )

    def pick_group(self, text: str) -> str:
        """Choose a clip group name for the given ASR partial."""
        if self._surprise_re.search(text):
            return "surprise"
        if _HESITATION_RE.search(text):
            return "thinking"
        return "affirm" if random.random() < self._affirm_probability else "continue"

    def select(
        self,
        text: str,
        groups: Mapping[str, Sequence[str]],
        recent: Sequence[str],
    ) -> str:
        """Pick a clip, avoiding recent ones within the chosen group."""
        group_name = self.pick_group(text)
        # Custom inventories may not define every default group name.
        if group_name not in groups:
            group_name = next(iter(groups))
        group = groups[group_name]

        candidates = [c for c in group if c not in recent]
        if not candidates:
            # Every clip in this group has been used recently. Widen back to the
            # whole group, but never onto the one that just played: a repeat two
            # clips apart passes unnoticed, back to back never does.
            last = recent[-1] if recent else None
            candidates = [c for c in group if c != last] or list(group)
        clip = random.choice(candidates)
        logger.debug(f"Backchannel: picked {clip!r} (group={group_name!r})")
        return clip
