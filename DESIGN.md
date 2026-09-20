# Design invariants

Off Book scores how accurately a singer reproduces a song's lyrics, from memory, with no
lyrics on screen. Ground truth is the publisher's **reference vocal** — an audio file —
transcribed at runtime by the same speech recognizer that transcribes the singer. This
page states what the engine guarantees about that audio and where each guarantee is
enforced in code, so that the claims can be checked rather than trusted.

Vocabulary, used exactly: the **reference vocal** is the licensed vocal recording,
analysed and never played; the **live vocal** is the mic; a **track** is one reference
vocal and any number of performances.

## 1. No lyric text exists anywhere in the system

Nothing ingests, stores, derives or displays written lyrics. The reference vocal is
transcribed in memory, at runtime, and its tokens are consumed by the aligner and
discarded.

- There is no lyric input of any kind — no LRC, no text field, no subtitle track. The
  only inputs are audio files (`offbook.audio.sources`).
- The session record does not contain token text by default. Timing, confidence,
  verdicts and scores are recorded; the words are not (`config.record_transcripts`,
  default `false`; `schema_version: 2`). The reference tokens are the lyrics, and so are
  the live tokens for every `MATCH`, so a record with text would be a lyric sheet.
  Recording text is an explicit opt-in for tuning work on audio you are licensed to
  transcribe. See [docs/session-record.md](docs/session-record.md).
- The stage view (`/stage`) shows only the *live* stream's confirmed words — what the
  singer actually sang — and never the reference stream. The control panel shows both
  streams transiently, to the operator, for diagnosis; nothing it shows is persisted.
- Session output (`sessions/`) and all audio are gitignored, so none of it can reach the
  repository by accident.

## 2. The reference vocal never reaches an output device

If the reference were audible the mic would capture it and every score would come back
near-perfect while measuring nothing. So the reference has no play path, by type.

- The reference is decoded to `ReferencePCM` (`offbook.audio.decode`), which exposes no
  array interface and no method that yields playable samples. The only object that can
  feed an output stream is `_BackingPlayer`, which accepts only `BackingPCM`.
- Every `sounddevice` output entry point is wrapped so that nothing outside
  `offbook.audio.graph` can open one (`offbook.audio.guard`).
- `AudioGraph.assert_reference_has_no_output_route()` runs at every session start.
- Replay sessions refuse to open an output device at all.
- `tests/test_no_output_route.py` tries the ways this could be violated and asserts each
  one is rejected.

The one thing the engine cannot enforce is acoustics: the backing track goes to the
singer's headphones, and if it leaks into the mic the live stream is polluted. Hence the
closed-back-headphone requirement in the README.

## 3. Reference and live are different types

The two streams are processed identically — same recognizer, same resampler, same
chunker — but they must never be confused. `Reference` and `Live` are phantom types
(`offbook.roles`); `Frames[Role]`, `Token[Role]` and `Recognizer[Role]` are generic in
them, and the aligner's inputs are typed so that a reference token cannot be passed
where a live token is expected, or vice versa.

- `RecognizerSpec` (implementation, unit, model id, revision, weight hash, dtype,
  chunking) must be identical on both streams or the session refuses to start
  (`RecognizerMismatchError`).
- `tests/test_roles_typing.py` runs mypy over a fixture that tries every way of swapping
  the roles and asserts each one is a type error. The invariant is a CI failure, not a
  convention.

## 4. Fully local and deterministic

Nothing about a performance leaves the machine, and the same audio always yields the
same score.

- The only network access is the model download (`offbook check-models`, or the first
  run), from Hugging Face, at revisions pinned in `models.lock.json` and verified against
  recorded weight hashes at load. The model hash is written into every session record.
- Decoding is greedy. Recognizer windows are cut on transport sample counts, never wall
  clock. Seeds are fixed.
- `tests/test_replay_models.py` replays the same audio twice and asserts identical
  tokens, timestamps, confidences and score.
- The transport clock is the mic's frame counter; the reference is advanced by exactly
  the number of frames the mic delivered (`ReferencePacer`, `PacerLockstepError`), so the
  two streams are sample-aligned by construction rather than by timing.

## What this design does not claim

- It does not judge pitch, tone or performance quality. It judges whether the words
  were sung, and roughly when. See [docs/how-it-works.md](docs/how-it-works.md) for the
  verdicts and the score.
- It does not prevent a person with the software and a licensed reference vocal from
  transcribing it by other means. It guarantees that *this system* never does so on
  their behalf, and never writes the result down.
- Encryption of reference vocals at rest is a distribution concern, outside this
  repository. The engine reads audio files; how they arrived and how they are protected
  on disk is the deploying application's responsibility.
