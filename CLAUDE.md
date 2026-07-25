# pipecat-backchannel

## The idea

Human conversation runs on two channels at once. One person holds the floor and talks. The other, without ever taking the floor, keeps feeding back short signals — "mhm", "yeah", "right". These are **backchannels**, and they mean *keep going, I'm with you*.

They are not turns. The speaker doesn't stop for them, doesn't answer them, and doesn't remember them as things the listener said. They're a parallel stream of evidence that someone is still on the other end.

Voice agents have no such channel. They are silent for as long as you speak, then reply once you stop. From the user's side that silence is indistinguishable from a hung connection, so they compensate: trailing off, repeating themselves, cutting a thought short to check the agent is still there. The conversation stops feeling live.

This library adds that missing channel to a voice agent, and nothing else. It is not a feature of the agent's intelligence — the agent never knows it happened. It's a thin behavioral layer that makes the agent *sound* like it's listening while it listens.

## Why it's shaped the way it is

The problem decomposes into decisions that are genuinely independent, and the architecture's whole job is to keep them that way:

- **When to signal.** A timing decision, made from audio alone. This is the hard part and the only one that can ruin the experience.
- **What to signal.** Which flavor of acknowledgment fits the moment.
- **Where the sound comes from.** Synthesis, recording, a vendor — an infrastructure concern.
- **How it reaches the user.** Turning a decision into audio in the pipeline.

Collapsing any two of these together is the mistake this design exists to prevent. They change for different reasons, at different rates, under different constraints.

Three properties fall out of the domain and are not negotiable:

**Timing is a hard deadline, not a goal.** A backchannel is only correct inside a narrow window. Slightly late is worse than never — an acknowledgment that lands after the moment has passed sounds like an interruption. Anything slow must therefore happen ahead of time, never in the moment. Work is precomputed at startup so the live decision is pure and instant.

**The decisions differ wildly in stakes.** Getting the timing wrong makes the agent talk over people. Getting the flavor wrong makes it sound very slightly off. These must not be coupled, and the cheap, sloppy decision must never be allowed to influence the expensive, careful one.

**Listening is not speaking.** A backchannel must leave no trace in the agent's memory of the conversation. The moment it's recorded as something the agent said, it has become a turn, and the whole premise is gone.

Read those three as the acceptance criteria for any change here. A change that's elegant but violates one of them is wrong.

---

# Engineering principles

## Dependency inversion — the one that matters most

**Depend on abstractions, never on concretions.** High-level policy must not know which low-level implementation serves it. If a module names a vendor, a library, or a driver, it is welded to it.

**Inject dependencies; never construct them inside.** A component that builds its own collaborators has silently made them mandatory and untestable. Pass them in, with a sensible default where there's an obvious one.

This is what buys everything else:

- **Swappability.** A new provider is a new implementation, not an edit.
- **Testability.** Inject a fake and the test is fast, offline, deterministic. If something is hard to test, that's the design telling you it's too coupled — fix the coupling, don't mock harder.
- **Blast radius.** Concrete details stay at the edges. A change out there doesn't reach the core.

The abstraction must be defined by the **consumer's** need, not the implementation's shape. An interface that mirrors one vendor's API is that vendor wearing a hat — the second implementation will not fit it.

## Simple first

Write the obvious thing. Complexity should arrive because something demanded it, not because you predicted it would. Speculative generality is paid for now, for a benefit that usually never comes.

You cannot guess the right abstraction from one use case. Wait for the second. Duplication is cheaper than the wrong abstraction — it's easy to merge later, while a bad abstraction is hard to remove once everything depends on it.

## Architecturally correct first

Simple is not sloppy. Get the **seams** right early — where things plug together is cheap to place now and expensive to move later. Everything else can be crude and improved in place.

A wrong abstraction is worse than none.

## SOLID

- **S** — one reason to change. If describing a component needs "and", split it.
- **O** — extend by adding, not by editing. New behavior shouldn't mean touching working code.
- **L** — a subtype must be usable anywhere its base is, without surprises.
- **I** — many small interfaces beat one fat one. Don't force a caller to depend on methods it ignores.
- **D** — see above.

## DRY

Deduplicate **knowledge, not text.** Two pieces of code that look alike but change for different reasons are not duplicates — merging them couples them forever. One fact should have one home; that's the point.

## Abstraction

An abstraction earns its place by hiding something the caller genuinely shouldn't know. If it only forwards calls, delete it — it's a layer, not an abstraction.

Name things for what they mean, not how they work. The name is the abstraction; if you can't name it clearly, the boundary is wrong.

## KISS

The reader is the constraint. Code is read far more than written, and the next reader has none of your context. Clever costs more than it saves.

Prefer boring, obvious, explicit. Fewer moving parts, fewer states, fewer ways to hold it wrong.
