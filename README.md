# Off Book — dual-stream scoring engine

[![ci](https://github.com/waveformeng/off-book/actions/workflows/ci.yml/badge.svg)](https://github.com/waveformeng/off-book/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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
   recognizer that transcribes the singer, and the tokens are consumed in memory. Session
   records hold timing, confidence, verdicts and scores — not the words — unless
   `record_transcripts` is switched on for tuning work. `sessions/` is gitignored.
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

[DESIGN.md](DESIGN.md) states each of these with the code and the test that enforces it.

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
   that recognizer's defaults; every alignment and chunking knob (all of them in
   [docs/how-it-works.md](docs/how-it-works.md#config-knobs)) is editable, **Reset to
   defaults** puts them back.
5. **Start session**. The status pill goes `loading` (models coming resident, ~5 s) →
   `running` → `done`. **Stop** ends a session early; the record is still written with
   `"error": "stopped"`.

While it runs, the right-hand column shows the score climbing, match rate, transport
position against the reference duration, ADC↔DAC drift, the model id/revision/hash, both
streams appending as tokens resolve (with what the recognizer currently sees but has not
confirmed shown dimmed at the end of each stream), and the verdict table (colour-coded, `dt` positive =
late). When it finishes you get the counts and a link to the session JSON.

**Past sessions** lists every record in the output directory with its JSON, and a
*replay* link on live sessions that flips the form to replay mode with the reference and
performance paths filled in — the fastest way to try a different tolerance or the other
recognizer on the same take.

One session at a time: a second Start while one is loading or running is refused. A
browser refresh mid-song reconnects and replays the readout so far.

### Stage view

<http://127.0.0.1:8765/stage> is the other face of the same server: a full-screen
performance display for the room, with nothing from the control panel on it. Put it on the
venue's screen (press `F` for fullscreen; the cursor hides itself) and drive sessions from
the control panel on the laptop. It shows exactly four things:

- **The singer's waveform** — an oscilloscope trace of the mic, and only the mic. The
  reference vocal has no route to it (`SessionConsole.live_audio` receives the live block
  alone), it is tapped on the capture thread so it follows the mic's clock rather than the
  decoder's, and it goes out on its own stream, `/api/stage/waveform`, rather than into
  the event history.
- **The singer's words** — the LIVE stream's runtime transcript, confirmed words only.
  Nothing appears until it has actually been sung and resolved; the recognizer's tentative
  guesses are not shown, and the reference transcript is never sent to this page. Words
  land ≈ 5 s after they are sung (see [docs/tuning.md](docs/tuning.md)); the waveform is live.
- **The score, as colour** — the background sweeps red → blue with the running *match
  rate* (the percentage), not the progress-scaled score, so it reflects how the singer is
  doing now rather than how far into the song they are. Neutral until the first verdict.
- **Credits, lower left** — *Song title* and *Singer* from the control panel's **Credits**
  section, set in Kabel Black the way MTV ran its lower-third in the '80s. Not written to
  the session record.

When the session ends the final match percentage is revealed; the page then waits for the
next session and picks it up without a reload.

### Endpoints, if you want to script it

| | |
|---|---|
| `GET /api/devices` | audio devices |
| `GET /api/browse?path=` | folders + audio files |
| `GET /api/config/defaults` | word / phoneme default configs |
| `POST /api/session/start` | body: `{mode, reference, backing?, performance?, input_device?, output_device?, out_dir, config, title?, singer?}` |
| `POST /api/session/stop` | |
| `GET /api/session/status` | `status`, `error`, `result` |
| `GET /api/events` | server-sent events: `status`, `header`, `resolved`, `tentative`, `tick`, `verdict`, `drift`, `final` |
| `GET /stage` · `GET /api/stage/waveform` | the stage view, and its server-sent mic frames `{frames: [{transport_s, pcm[]}]}`, each block decimated to 256 samples (no history) |
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

Every alignment and chunking knob is a flag on `run`/`replay`; the table is in
[docs/how-it-works.md](docs/how-it-works.md#config-knobs). `--record-transcripts` writes
token text into the session record, which is off by default (see
[docs/session-record.md](docs/session-record.md)).

## Documentation

| | |
|---|---|
| [DESIGN.md](DESIGN.md) | the four invariants — no lyric text, no output route for the reference, typed roles, local and deterministic — and where each is enforced |
| [docs/how-it-works.md](docs/how-it-works.md) | audio graph, recognizers, alignment, verdicts, the score, every config knob |
| [docs/tuning.md](docs/tuning.md) | the transcription diagnostic, how the chunker defaults were chosen, measurements |
| [docs/session-record.md](docs/session-record.md) | the JSON written after every session |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how to work on it |

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

## License

Apache-2.0 — see [LICENSE](LICENSE). The models are downloaded at first run and are not
part of this repository: Parakeet-TDT is © NVIDIA under CC-BY-4.0, the wav2vec2-espeak
checkpoint is © Meta under Apache-2.0. See [NOTICE](NOTICE).
