# How it works

## Audio graph

```
 reference.wav ─► decode ─► ReferencePCM ─► ReferencePacer ─► Frames[Reference] ─► Recognizer[Reference] ─┐
                                               ▲ advanced by exactly N frames                            ├─► Aligner ─► Score
 mic ──────────► input callback (N frames) ────┴─► Frames[Live] ────────────────► Recognizer[Live] ──────┘
                        │                                        (transport clock = mic frame counter)
                        └─► performance.wav
 backing.wav ──► decode ─► BackingPCM ─► output device (headphones) ─► DriftMonitor (DAC clock vs transport)
```

The mic input callback is the transport clock. On every block of N frames it advances the
reference pacer by exactly N frames, so the two analysis streams are sample-aligned by
construction (a non-zero residual raises `PacerLockstepError`). The backing track runs on
the output device's clock; `DriftMonitor` uses PortAudio's ADC/DAC timestamps to measure
where the backing was, in its own clock, at the instant each mic block was captured, and
logs the difference every second. When mic and headphones are one device the graph opens
a single duplex stream and drift is zero by design.

Both streams then pass through identical soxr resamplers to 16 kHz and identical chunkers.

## Recognizers

Pluggable `Backend` (window of PCM → tokens with times and confidence). Two ship:

| | A. word | B. phoneme |
|---|---|---|
| model | `mlx-community/parakeet-tdt-0.6b-v2` via parakeet-mlx, greedy TDT | `facebook/wav2vec2-lv-60-espeak-cv-ft`, greedy CTC, MLX port in `offbook/asr/mlx_wav2vec2` |
| token | normalized word, timestamps from TDT, confidence = mean piece confidence | IPA phoneme, 20 ms frame span, confidence = mean posterior |

`RecognizerSpec` (impl, unit, model id, revision, weight hash, dtype, chunking) must be
identical on both streams or the session refuses to start (`RecognizerMismatchError`).
Two backend instances are loaded (one per stream); `--share-weights` shares tensors.

Streaming: every `hop_s`, the last `window_s` of audio is re-decoded. A token is emitted,
once, when it ends before `window_end − resolve_margin_s`, starts past the window's
left-edge guard, and the previous decode produced the same token within `agree_s`;
anything overlapping an already-emitted token is dropped as a re-spelling. Both instances
are serviced from one inference thread in a fixed order, so MLX is never entered
concurrently. See [tuning.md](tuning.md) for why the window is 30 s.

## Comparison and score

Tokens are held until *decidable* — a reference token once the live stream has resolved
past `start + tolerance + max_lag`, a live token once the reference stream has resolved
past `start + tolerance`. "Resolved" is each recognizer's promise that no further token
will start before that time; it trails the decode by `frontier_lag_s`, so the verdict for
a word lands roughly `margin + frontier_lag + hop` ≈ 9 s after it is sung. That delay is
the price of not deciding a region one stream may still add a token to — deciding early
is how a late-surfacing token turns one MATCH into a MISSED plus an INSERTED. The pending window is re-aligned with Levenshtein DP whose
substitution cost carries lexical distance (0/1 for words, normalized edit distance over
IPA characters for phonemes) and a timing penalty; unpairable pairs (outside
`[−tolerance, tolerance + max_lag]`) cost ∞. A beat-late singer is still paired.

| verdict | meaning |
|---|---|
| `MATCH` | same token (word mode: same spelling or same sound), \|dt\| ≤ tolerance |
| `FAIL_TIMING` | same token, late by up to tolerance + max_lag or early by up to tolerance + max_lead; worth `1 − timing_weight` |
| `FAIL_LEXICAL` | paired in time, different token |
| `FAIL_MISSED` | reference token with no live counterpart |
| `FAIL_INSERTED` | live token with no reference counterpart |

```
match_rate = Σ points / Σ verdicts
score      = 100 · match_rate · min(1, transport_position / reference_duration)
```

The score climbs from zero through the song and lands on `100 · match_rate`.

Two policies that keep the score honest:

- **After the reference vocal's last word, nothing is scored.** A live token that no
  reference token seen so far could pair with waits until the reference stream has
  finished (a later reference phrase may still arrive — an ad-lib during an instrumental
  break *is* inserted once the next phrase shows up); if the reference ends without one,
  the token is ignored and counted in the record's
  `live_tokens_ignored_after_reference_end`. Talking over the outro does not cost points.
  `score_past_reference_end` turns the old behaviour back on.
- **Breath tokens are not words.** The word recognizer emits "uh"/"hmm"-type tokens on
  sung intakes; both streams drop them in normalization.
- **Homophones are the same word.** The word recognizer spells a sound however its
  language model leans that moment — `for` on the reference, `four` on the singer. In
  word mode two words match if their spellings or their Metaphone sound keys agree
  (`offbook/compare/sound.py`; pure rules, no dictionary). Applied to both streams alike,
  so it cannot favour either. `word_match: exact` turns it off.

What the score does *not* do by default: reward timing inside the tolerance. Two singers
who both land every word within 0.75 s score the same even if one is dead on and the
other consistently half a second late. `graded_timing` changes that — inside the
tolerance a match earns `1 − timing_weight·|dt|/tolerance`, continuous with FAIL_TIMING
at the boundary — and it is the knob to turn if timing precision should count. On two
real takes of the same 45 s song by two singers, one on the beat and one drifting
0.2–0.6 s late then rushing the last phrase: flat scoring 84.7 vs 70.0, graded 79.9 vs
62.0.

## Config knobs

All exposed on `run`/`replay`; defaults in `offbook/config.py`.

| flag | default | meaning |
|---|---|---|
| `--tolerance` | 0.75 s (phoneme: 0.4) | \|dt\| within which a lexical match is also a timing match |
| `--max-lag` | 1.5 s | extra lateness a live token may have and still be paired |
| `max_lead_s` (config/web) | 1.5 s | extra earliness a live token may have and still be paired — a singer who rushes gets FAIL_TIMING, not a phantom insertion plus a missed word |
| `word_match` (config/web) | `sound` | word mode: `sound` also accepts homophones via a Metaphone key (`for`/`four`, `there`/`their`); `exact` compares spellings |
| `graded_timing` (config/web) | off | inside the tolerance a MATCH earns `1 − timing_weight·|dt|/tolerance` instead of a flat 1 |
| `--window-tokens` | 12 (phoneme: 48) | reference tokens held in the edit-distance window |
| `--timing-weight` | 0.5 | weight of a `FAIL_TIMING` relative to a lexical failure (0–1) |
| `--window-s` / `--hop-s` / `--margin-s` | 30 / 1 / 2 s | recognizer window, hop, and how much of the window's tail stays tentative |
| `--agree-s` | 0.3 s | two consecutive decodes must agree on a token (same text, start within this) before it is emitted |
| `--edge-guard-s` | 1.0 s | tokens starting this close to a window's left edge are ignored (cut-off phrases decode badly) |
| `confirm_timeout_s` (config/web) | 2 s | a token unconfirmed this long past the resolve line is emitted anyway |
| `frontier_lag_s` (config/web) | 2 s | how far the stream's resolved frontier trails the resolve line, so late-surfacing tokens still land ahead of it |
| `score_past_reference_end` (config/web) | off | count what is sung after the reference vocal's last word as inserted; off = ignore it |
| `--dtype` | float32 | MLX weight dtype (`bfloat16` halves memory) |
| `--share-weights` | off | one weight set shared by the two recognizer instances |
