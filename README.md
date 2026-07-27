<div align="center">
  <a href="https://github.com/pipecat-ai/pipecat">
    <img alt="Pipecat" width="220px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png">
  </a>
  <p><em>A community integration for <a href="https://github.com/pipecat-ai/pipecat">Pipecat</a></em></p>
</div>

# pipecat-backchannel

[![PyPI](https://img.shields.io/pypi/v/pipecat-backchannel?cacheSeconds=3600)](https://pypi.org/project/pipecat-backchannel)
[![Python](https://img.shields.io/pypi/pyversions/pipecat-backchannel?cacheSeconds=3600)](https://pypi.org/project/pipecat-backchannel)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue)](LICENSE)
[![Pipecat](https://img.shields.io/badge/Pipecat-1.6.0-brightgreen)](https://github.com/pipecat-ai/pipecat)

Your Pipecat agent listens back: quiet "mhm"s and "yeah"s while the user is
still talking.

## Demo

https://github.com/user-attachments/assets/6c38d03b-4eb6-42bd-8f0d-47a1cfdb1bfc

## Why

Listen to two people on a phone call. While one tells a story, the other keeps
feeding back small signals: "mhm", "yeah". Linguists call these
**backchannels**. They mean _keep going, I'm with you_, and the speaker talks
straight through them.

Talk to a voice agent and you get silence until you finish. Silence on a call
reads as a dropped connection, so you hesitate, trail off, or ask "hello?".

This library adds the listening sounds. The agent keeps the floor with you the
whole time: a backchannel takes no turn and leaves no trace in your LLM
context.

## Install

```bash
uv add pipecat-backchannel
```

## Usage

```python
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
```

One wrapper call is the whole setup. The default configuration works without
an API key, a voice ID, or a sample rate, and places every processor for you.

The first run records the clips through **your own TTS**, so they come out in
your bot's voice, and caches them in `.clip_cache/`. Every later run starts
from the cache. Delete that directory after changing voice.

## Record before the first call

The first session waits a few seconds while the recorder fills the clip cache.
Move that wait to app startup if you want to avoid delay for first caller:

```python
backchannel = Backchannel()          # one per process

async def run_bot(...):
    pipeline = Pipeline(backchannel([... tts ...]))

if __name__ == "__main__":
    asyncio.run(backchannel.prewarm(tts=make_tts()))   # before the server opens
    main()
```

## Tuning

```python
Backchannel(
    volume=0.6,                              # how loud, relative to the recording
    params=BackchannelParams(
        fire_probability=0.8,                # chance of speaking up at a good moment
        cooldown_s=2.5,                      # minimum gap between backchannels
        min_speech_before_eligible_s=0.7,    # ignore pauses right after a turn starts
    ),
)
```

`fire_probability` and `cooldown_s` set how present the listener feels. Raise
the first and lower the second for a more active one. Keep some slack: a bot
that reacts to each eligible pause sounds mechanical, because human listeners
skip some of them.

`volume` sits below 1.0 by default so clips play under the user's voice.

Set `loguru` to `DEBUG` to see each fire/skip decision and its reason.

### Going quiet mid-session

Some stretches of a call want a silent listener: the user dictating a card
number, or the bot walking through a form where each pause belongs to a
field, and an "mhm" would read as confirmation.

The component that decides when to fire is the `BackchannelProcessor`, one
per pipeline. `backchannel([...])` returns your processor list with it
inserted, so keep that list, build the `Pipeline` from it, and pull the gate
out of it:

```python
from pipecat_backchannel.processor import BackchannelProcessor

processors = backchannel([transport.input(), stt, llm, tts, transport.output()])
pipeline = Pipeline(processors)

gate = next(p for p in processors if isinstance(p, BackchannelProcessor))
```

Then flip it from any event handler, as many times as the call needs:

```python
gate.enabled = False   # stop firing; every frame still passes through
...
gate.enabled = True    # resume
```

`enabled` is a plain attribute, safe to set at any moment. While it is
`False` the processor keeps passing audio through and keeps tracking speech,
so nothing downstream notices and re-enabling picks up mid-turn. It fires no
clips and runs no end-of-turn classification in that state. The pipeline, the
LLM context, and the other sessions' gates stay untouched.

## Custom clips

The library groups clips by conversational function: continuer, agreement,
hesitation, surprise.

```python
Backchannel(clip_groups={"continue": ["Mhm.", "Right."], "affirm": ["Yeah.", "Yep."]})
```

Two rules for a custom inventory:

- **Give each group at least two clips.** A one-clip group repeats itself.
- **Get variety from punctuation, and keep the vocabulary small.** Real
  backchannels draw on a handful of sounds: _uh-huh, yeah, mm-hmm, right,
  okay, oh, huh, hm_. A written phrase like "absolutely" reads fine on the
  page, then sounds like an answer when spoken over someone mid-sentence.
  Punctuation is the handle your TTS gives you: `"Mhm."` falls, `"Mhm,"` stays
  up, `"Hm..."` draws out.

Pass `synthesizer=` to use a different voice, or `cache=` to supply your own
recorded `.wav` files. See `pipecat_backchannel.synth` and `.cache`.

## Run the example

```bash
cp examples/.env.template examples/.env   # DEEPGRAM_API_KEY, CARTESIA_API_KEY, CEREBRAS_API_KEY
uv run examples/bot.py
```

Open `http://localhost:7860/client`, talk in mid-length sentences, and pause
mid-thought. A quiet "mhm" lands on clause-complete pauses. Word-search pauses
("we need the, uh...") and finished questions get silence, and the questions
get a real reply.

## Limitations

- **Pauses only.** Humans backchannel during speech too, cued by pitch. This
  library reacts to pauses. The roadmap covers the rest.
- **About 2× inference while the user speaks.** It runs a second VAD and
  turn-detector on purpose: the pipeline's copies are tuned for turn-taking,
  and pause timing needs the opposite settings. A handful of concurrent
  sessions runs fine; measure before a large deployment.
- **English only.** The clips and the regexes are English. The underlying
  models handle other languages.

## Roadmap

- [ ] **Fire during speech.** Real listeners backchannel mid-utterance, cued
      by pitch and rhythm rather than silence. That takes Voice Activity
      Projection; [MaAI](https://github.com/MaAI-Kyoto/MaAI) ships a usable
      model. It reads a different signal, so it slots in beside the pause
      gate rather than replacing it, and it is the largest quality jump left.
- [ ] **Match the speaker's prosody.** The selector picks a clip at random
      within its group. Choosing the one whose pitch and energy fit the last
      utterance would likely want a small prediction model at fire time,
      trained to map the tail of the user's speech to a clip. Playback stays
      free — the clips already sit on disk — so the only new cost is that one
      lightweight inference.

## License

BSD 2-Clause © Taras Maister, matching Pipecat's own license. Not affiliated
with Daily or the Pipecat core team.
