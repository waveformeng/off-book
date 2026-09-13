# Off Book — dual-stream scoring engine

Waveform Karaoke's *Off Book* mode: a singer performs from memory, no lyrics on screen.
This is the engine underneath. It plays nothing but the backing track, transcribes the
publisher's **reference vocal** and the singer's **live vocal** through the *same*
recognizer at the same time, aligns the two token streams on a shared transport clock,
and keeps a running score.

Vocabulary, used exactly:

| term | meaning |
|---|---|
| **REFERENCE VOCAL** | the publisher's licensed vocal demo. Ground truth. Read from file, decoded in software, **never played** |
| **LIVE VOCAL** | the mic. The thing being scored |
| **TRACK** | a song: one reference vocal, many performances |

## Hard constraints, and where they are enforced

1. **No lyric text.** Nothing in this repo ingests, stores, derives or outputs written
   lyrics. Ground truth is the reference *audio*, transcribed at runtime by the same
   recognizer that transcribes the singer. Session records contain the runtime
   transcripts of both streams; `sessions/` is gitignored.
2. **The reference vocal never reaches an output device.** It is decoded to an in-memory
   `ReferencePCM` that has no array interface and no play path. The only object that can
   feed an output stream is `_BackingPlayer`, which accepts only `BackingPCM`. The graph
   asserts this at session start (`AudioGraph.assert_reference_has_no_output_route`), and
   every `sounddevice` output entry point is wrapped so nothing outside
   `offbook.audio.graph` can open one (`offbook.audio.guard`). If the reference were ever
   audible the mic would capture it and every score would come back near-perfect while
   measuring nothing — hence the closed-headphone requirement below.
3. **Fully local.** MLX on Apple Silicon. The only network access is `offbook check-models`
   (or the first run), which downloads the pinned model revisions from Hugging Face.
4. **Deterministic.** Greedy decoding only; model revisions and weight hashes pinned in
   `models.lock.json` and verified at load; recognizer windows are cut on sample counts,
   never wall clock; fixed seeds. The model hash is written into every session record.
   `tests/test_replay_models.py` replays the same audio twice and asserts identical tokens,
   timestamps, confidences and score.
5. **Audio files are inputs, never committed.** See `.gitignore`.

## Setup

Requires macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
brew install uv
uv sync                      # Python 3.12 + all dependencies, locked
uv run offbook check-models  # downloads the two pinned models (~3.7 GB) and verifies hashes
uv run offbook web           # control panel at http://127.0.0.1:8765 — see "Web interface"
```

The first phoneme-model run converts the PyTorch checkpoint to MLX safetensors under
`.models/` (no torch involved; see `offbook/asr/mlx_wav2vec2/checkpoint.py`).

## The closed-headphone requirement

**Run live sessions with closed-back headphones. Never with speakers.**

The backing track goes to the headphones. The reference vocal goes nowhere. The mic must
hear only the singer: if the mic hears the backing track the live stream is polluted, and
if anything ever leaked the reference vocal the system would grade ground truth against
itself. The engine measures ADC↔DAC drift, not acoustic leakage — leakage is on you.

## Web interface

The quickest way to drive the engine: a local control panel for file locations,
devices, config and start/stop, with the live readout in the browser.

### Spin it up

```sh
uv run offbook web
#   off book control panel → http://127.0.0.1:8765
```

Open <http://127.0.0.1:8765>. The server binds to localhost only and stays in the
foreground; `Ctrl-C` shuts it down (a running session is stopped and its record written).

Options:

```sh
uv run offbook web --port 9000            # different port
uv run offbook web --host 0.0.0.0         # reachable from other machines on the LAN —
                                          # only do this on a network you trust; there is no auth
```

The terminal that runs `offbook web` still prints the console readout for every session,
so you can watch either place.

### Using it

1. **Mode** — *Live (mic)* to score a performance now, *Replay* to re-score a recorded
   `performance.wav` (no audio devices are opened; deterministic).
2. **Files** — type paths or click **Browse**. The browser is server-side: it lists
   folders and audio files on the machine running `offbook web`, because a web page
   cannot read paths off your disk. Fields:
   - *Reference vocal* — analysed, never played.
   - *Backing track* (live only) — goes to the headphones. Optional.
   - *Recorded performance* (replay only).
   - *Session output directory* — where `sessions/<id>/session.json` and
     `performance.wav` land. Defaults to `sessions/` under the current directory.
3. **Devices** (live only) — pick the mic and the headphones. The hint under the pickers
   says whether that pairing is one CoreAudio device (duplex stream, drift zero by design)
   or two (split streams, drift measured and logged). Put on closed-back headphones.
4. **Recognizer** — *A · Word* (Parakeet) or *B · Phoneme* (wav2vec2). Switching loads
   that recognizer's defaults; every alignment and chunking knob from the table below is
   editable, **Reset to defaults** puts them back.
5. **Start session**. The status pill goes `loading` (models coming resident, ~5 s) →
   `running` → `done`. **Stop** ends a session early; the record is still written with
   `"error": "stopped"`.

While it runs, the right-hand column shows the score climbing, match rate, transport
position against the reference duration, ADC↔DAC drift, the model id/revision/hash, both
streams appending as tokens resolve, and the verdict table (colour-coded, `dt` positive =
late). When it finishes you get the counts and a link to the session JSON.

**Past sessions** lists every record in the output directory with its JSON, and a
*replay* link on live sessions that flips the form to replay mode with the reference and
performance paths filled in — the fastest way to try a different tolerance or the other
recognizer on the same take.

One session at a time: a second Start while one is loading or running is refused. A
browser refresh mid-song reconnects and replays the readout so far.

### Endpoints, if you want to script it

| | |
|---|---|
| `GET /api/devices` | audio devices |
| `GET /api/browse?path=` | folders + audio files |
| `GET /api/config/defaults` | word / phoneme default configs |
| `POST /api/session/start` | body: `{mode, reference, backing?, performance?, input_device?, output_device?, out_dir, config}` |
| `POST /api/session/stop` | |
| `GET /api/session/status` | `status`, `error`, `result` |
| `GET /api/events` | server-sent events: `status`, `header`, `resolved`, `tick`, `verdict`, `drift`, `final` |
| `GET /api/sessions?out_dir=` · `GET /api/sessions/{id}/record` | past records |

## Run from the terminal

```sh
# list devices (index, name, channels, default rate)
uv run offbook devices

# live session: reference vocal (analysed, never played) + backing track (headphones) + mic
uv run offbook run path/to/reference_vocal.wav \
    --backing path/to/backing.wav \
    --input-device 0 --output-device 1

# phoneme recognizer instead of words
uv run offbook run path/to/reference_vocal.wav --backing path/to/backing.wav --phoneme

# re-score a recorded performance; opens no audio devices, fully deterministic
uv run offbook replay path/to/reference_vocal.wav sessions/<id>/performance.wav
```

Input formats: anything libsndfile reads (wav, flac, aiff, ogg, mp3). Any sample rate;
both streams are resampled identically to 16 kHz for the recognizers.

Every session writes `sessions/<id>/session.json` (and, for live sessions,
`sessions/<id>/performance.wav`, the live vocal at the device rate for replay).

### Console readout

```
recognizer  parakeet (word) mlx-community/parakeet-tdt-0.6b-v2@8ae155301e23 hash=b005e3e9500ba275 ...
         REFERENCE   LIVE               VERDICT             dt   SCORE
     2.05s  ref ▸ the quick brown   live ▸ the quick brown
               the   the                MATCH            +0.00    25.0
             quick   quick              MATCH            +0.00    25.0
     3.01s  drift ADC↔DAC -0.003 ms
               dog   cat                FAIL_LEXICAL     +0.00    88.9
                 —   um                 FAIL_INSERTED             78.6
              away   away               MATCH            +0.64    80.0
```

`ref ▸ … live ▸ …` lines show each stream's tokens as they resolve; verdict rows follow
once both sides are decidable. `dt` is `live.start − reference.start` (positive = late).

### Config knobs

All exposed on `run`/`replay`; defaults in `offbook/config.py`.

| flag | default | meaning |
|---|---|---|
| `--tolerance` | 0.75 s (phoneme: 0.4) | \|dt\| within which a lexical match is also a timing match |
| `--max-lag` | 1.5 s | extra lateness a live token may have and still be paired |
| `--window-tokens` | 12 (phoneme: 48) | reference tokens held in the edit-distance window |
| `--timing-weight` | 0.5 | weight of a `FAIL_TIMING` relative to a lexical failure (0–1) |
| `--window-s` / `--hop-s` / `--margin-s` | 30 / 2 / 2 s | recognizer window, hop, and how much of the window's tail stays tentative |
| `--agree-s` | 0.3 s | two consecutive decodes must agree on a token (same text, start within this) before it is emitted |
| `--edge-guard-s` | 1.0 s | tokens starting this close to a window's left edge are ignored (cut-off phrases decode badly) |
| `--dtype` | float32 | MLX weight dtype (`bfloat16` halves memory) |
| `--share-weights` | off | one weight set shared by the two recognizer instances |

## How it works

### Audio graph

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

### Recognizers

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
concurrently. See *Tuning transcription* below for why the window is 30 s.

### Comparison and score

Tokens are held until *decidable* — a reference token once the live stream has resolved
past `start + tolerance + max_lag`, a live token once the reference stream has resolved
past `start + tolerance`. The pending window is re-aligned with Levenshtein DP whose
substitution cost carries lexical distance (0/1 for words, normalized edit distance over
IPA characters for phonemes) and a timing penalty; unpairable pairs (outside
`[−tolerance, tolerance + max_lag]`) cost ∞. A beat-late singer is still paired.

| verdict | meaning |
|---|---|
| `MATCH` | same token, \|dt\| ≤ tolerance |
| `FAIL_TIMING` | same token, tolerance < \|dt\| ≤ tolerance + max_lag; worth `1 − timing_weight` |
| `FAIL_LEXICAL` | paired in time, different token |
| `FAIL_MISSED` | reference token with no live counterpart |
| `FAIL_INSERTED` | live token with no reference counterpart |

```
match_rate = Σ points / Σ verdicts
score      = 100 · match_rate · min(1, transport_position / reference_duration)
```

The score climbs from zero through the song and lands on `100 · match_rate`.

## Tuning transcription

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
| **30 / 2 / 2** (default) | **0.065** | **0.136** |

Two things did the work: long windows (a 6 s window cuts phrases and starves the model of
context) and *agreement* (recognizer timestamps jitter by 80–160 ms between overlapping
windows; the original chunker's hard frontier dropped any token that re-decoded a hair
earlier than the previous cut). parakeet-mlx's own cached streaming mode was tried and
rejected: error rate above 0.8 on the same audio. With the defaults, replaying that vocal
against itself scores 100.0 over 197 tokens. The phoneme recognizer through the same
chunker sits at 0.10–0.17 against 30–60 s one-shot decodes (its one-shot decode of a
5-minute file is itself degraded — wav2vec2 does not like long inputs — so compare it on
`--seconds 60`).

Cost of the 30 s window: about 0.35 s of inference per 2 s hop per stream on the M4, so
roughly a third of real time for both streams. Emission latency is `margin + hop` ≈ 4 s
after a word is sung; the comparison aligns on transport timestamps, so this only delays
the readout, not the verdicts.

## Session record

`sessions/<id>/session.json`, schema in [`schema/session.schema.json`](schema/session.schema.json)
(generated by `offbook schema`; CI checks it is current). Top level:

| field | |
|---|---|
| `session_id`, `started_at`, `mode` | `live` or `replay` |
| `reference_path`, `backing_path`, `performance_wav_path` | inputs; the performance wav is what `replay` takes |
| `transport_rate`, `reference_duration_s` | |
| `recognizer` | `impl`, `unit`, `model_id`, `revision`, **`model_hash`**, `dtype` |
| `config` | the full `SessionConfig` used |
| `pairs[]` | every token pair: `reference` / `live` (`text`, `start_s`, `end_s`, `confidence`, `resolved_wall_s`) or null, `verdict`, `dt_s`, `score_after` |
| `counts`, `match_rate`, `final_score` | |
| `drift` | `duplex`, rates, 1 Hz `samples[]` of `{transport_s, drift_s}`, `max_abs_s`, `final_s` |
| `session_duration_s`, `error` | |

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

## Development

```sh
uv run ruff check src tests && uv run mypy && uv run pytest
OFFBOOK_MODEL_TESTS=1 uv run pytest tests/test_replay_models.py   # real models, ~20 s
```

`tests/test_roles_typing.py` runs mypy over a fixture that tries every way of swapping
reference and live and asserts each one is rejected — the invariant is a CI failure, not a
convention.

Layout:

```
src/offbook/
  roles.py            Reference / Live phantom types, the constrained Role TypeVar
  clock.py            TransportTime, ANALYSIS_RATE
  config.py           SessionConfig (chunking, alignment, recognizer)
  audio/              sources · decode · guard · graph · pacer · capture · drift
  asr/                base (Frames[Role], Token[Role], Backend, RecognizerSpec) · chunker ·
                      parakeet · w2v2_phoneme · mlx_wav2vec2/ · hashing · registry
  compare/            normalize · align · verdict · score
  session/            runner · record · console · transcribe (diagnostic)
  web/                server (FastAPI, SSE) · events · static/index.html
  cli.py
models.lock.json      pinned model ids, revisions, weight hashes
schema/               session.schema.json
```
