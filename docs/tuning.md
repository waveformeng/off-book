# Tuning transcription

Trust the reference transcript before trusting the score; the live vocal goes through
exactly the same path. The diagnostic:

```sh
uv run offbook transcribe path/to/reference_vocal.wav --full            # word recognizer
uv run offbook transcribe path/to/reference_vocal.wav --full --phoneme
uv run offbook transcribe path/to/reference_vocal.wav --full --seconds 120 --window-s 20 --hop-s 1
```

It prints the streaming chunker's tokens with timestamps, then (`--full`) the recognizer's
one-shot decode of the whole file — the best that model can do on that audio — and the
chunker's token error rate against it. That separates the two things that can be wrong:

- **Error rate high, one-shot transcript good** → the streaming path is losing tokens;
  tune `--window-s`, `--hop-s`, `--margin-s`, `--agree-s`, `--edge-guard-s`.
- **One-shot transcript itself poor** → the model can't read this vocal; try the other
  recognizer, or a cleaner reference stem.

How the defaults were chosen, on a real 5-minute sung publisher vocal (error rate of the
streaming chunker against the one-shot decode; lower is better):

| window / hop / margin | first 120 s | full song |
|---|---|---|
| 6 / 1 / 1 with the original frontier chunker | ≈ 0.6 | — |
| 15 / 1 / 1.5 | 0.169 | — |
| 20 / 2 / 2 | 0.091 | 0.191 |
| 30 / 2 / 2 | 0.065 | 0.136 |
| **30 / 2 / 3** (default) | 0.065 | 0.143 |

Two things did the work: long windows (a 6 s window cuts phrases and starves the model of
context) and *agreement* (recognizer timestamps jitter by 80–160 ms between overlapping
windows; the original chunker's hard frontier dropped any token that re-decoded a hair
earlier than the previous cut). parakeet-mlx's own cached streaming mode was tried and
rejected: error rate above 0.8 on the same audio. With the defaults, replaying that vocal
against itself scores 100.0 over 197 tokens. The phoneme recognizer through the same
chunker sits at 0.10–0.17 against 30–60 s one-shot decodes (its one-shot decode of a
5-minute file is itself degraded — wav2vec2 does not like long inputs — so compare it on
`--seconds 60`).

On a second, shorter vocal (45 s, clean diction) the default chunker matches the one-shot
decode exactly: error rate 0.000, 56/56 tokens. Replaying a real 45 s performance of it
against that reference: 87.7, where every failure is a genuine word difference between
what was sung and what the reference has.

Cost of the 30 s window: about 0.35 s of inference per 2 s hop per stream on the M4, so
roughly a third of real time for both streams. A word's tokens are emitted `margin + hop`
≈ 5 s after it is sung and its verdict lands ≈ 9 s after (see [how-it-works.md](how-it-works.md#comparison-and-score));
the comparison aligns on transport timestamps, so this delays the readout, not the
result.

## Measured on this machine (MacBook Pro M4, 24 GB)

Two recognizer instances resident, decoding a 30 s window each (the per-hop cost):

| recognizer | dtype | MLX peak | decode per instance |
|---|---|---|---|
| parakeet | float32 | 5.8 GB | 334 ms |
| parakeet | bfloat16 | 3.6 GB | 404 ms |
| w2v2-phoneme | float32 | 3.8 GB | 497 ms |
| w2v2-phoneme | bfloat16 | 2.7 GB | 516 ms |

With the default 2 s hop that is 0.35–0.5 s of inference per second of audio for both
streams together. Built-in mic ↔ built-in output drift measured at −3 µs over 8 s (shared
clock). Replay of the same audio, three runs each in float32 and bfloat16: identical
tokens, timestamps, confidences and score. A 5-minute sung reference vocal replayed
against itself: 100.0, 197/197 MATCH.

M2 Air / 16 GB: not measured here. The float32 parakeet configuration needs ~6 GB of
unified memory for the two instances; `--dtype bfloat16` or `--share-weights` are the
degrade paths. Run `uv run offbook replay` on a recorded performance there and watch that
the `ref ▸ / live ▸` lines keep pace with the transport time.
