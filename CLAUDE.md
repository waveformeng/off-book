# CLAUDE.md

Off Book: a dual-stream lyric-accuracy scoring engine for karaoke. Python 3.12, MLX on
Apple Silicon, FastAPI web UI. Read [DESIGN.md](DESIGN.md) before changing anything that
touches audio, tokens, or the session record — it lists four invariants and the tests
that enforce them, and a change that weakens one will not be merged.

If `.claude/CONTEXT.md` exists, read it: it holds private working context (decisions,
backlog) that is not part of the public repo.

## Commands

```sh
uv sync                                   # deps, locked
uv run offbook check-models               # ~3.7 GB, once; verifies pinned hashes
uv run offbook web                        # control panel :8765, stage view at /stage
uv run offbook replay REF.wav PERF.wav    # deterministic re-score, no audio devices
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy && uv run pytest
OFFBOOK_MODEL_TESTS=1 uv run pytest tests/test_replay_models.py   # real models, ~20 s
uv run offbook schema                     # regenerate schema/session.schema.json
```

CI (`.github/workflows/ci.yml`, macOS runner) runs exactly the check line above plus a
diff of the regenerated schema against the committed one. **Any change to `config.py` or
`session/record.py` must be followed by `uv run offbook schema`** or CI fails.

## Invariants (summary — DESIGN.md is authoritative)

1. **No lyric text**, in, out, or on disk by default. Session records omit token text
   unless `config.record_transcripts` is on (schema v2). The stage view shows only the
   live stream's *confirmed* words — never tentative guesses, never the reference.
2. **The reference vocal has no play path.** `ReferencePCM` has no array interface;
   `_BackingPlayer` is the only thing that feeds an output device; `audio/guard.py`
   wraps every `sounddevice` output entry point.
3. **`Reference` and `Live` are phantom types.** Anything generic over a stream is
   generic over `Role`. `tests/test_roles_typing.py` asserts swaps are type errors.
4. **Local and deterministic.** Greedy decoding, sample-count boundaries, pinned model
   revisions + hashes in `models.lock.json`, fixed seeds.

## Layout

```
src/offbook/
  roles.py clock.py config.py     phantom types · transport time · SessionConfig
  audio/     sources decode guard graph pacer capture drift
  asr/       base chunker parakeet w2v2_phoneme mlx_wav2vec2/ hashing registry
  compare/   normalize sound align verdict score
  session/   runner record console transcribe
  web/       server.py (FastAPI, SSE) · events.py · static/index.html · static/stage.html
docs/        how-it-works · tuning · session-record
```

Two web pages, one server:
- `/` control panel — the operator's diagnostic view; shows both streams incl. tentative.
- `/stage` performance display — singer's oscilloscope waveform (from
  `/api/stage/waveform`, a separate SSE stream that is deliberately *not* in the event
  history), confirmed live words, background hue red→blue by *match rate* (not the
  progress-scaled score), Kabel Black lower-third credits (`title`/`singer` on
  `StartRequest`). Kabel is the MTV easter egg; only the credits use it.
- `/api/events` replays history to late joiners and tracks `state.generation` so a page
  left open across sessions picks up the next one.

## Conventions

- Strict mypy, ruff, line length 100. Docstrings explain *why*; match the density of
  the surrounding code. No comments that restate the line.
- Product-tunable behaviour goes in `SessionConfig` with a description and a default
  that preserves current scores.
- Scoring changes come with before/after replay numbers (see `docs/tuning.md`).
- Never commit audio, `sessions/`, or weights. Test fixtures are generated with `say`.
- The event console (`web/events.py`) subclasses `session/console.py`; add a hook to
  the base class as a no-op, override in the event console.

## Verifying the stage view

Real audio through the whole pipeline: start `offbook web` on a spare port (your own
instance may already hold 8765), POST `/api/session/start` with `mode: replay`, a
reference vocal and a `sessions/<id>/performance.wav` from a past live take, plus
`title`/`singer`, then open `/stage`. Replay is inference-bound (~3× real time, bursty);
live sessions deliver a block every ~21 ms, so motion is smoother than replay shows.
Past takes with known scores live in `sessions/` (gitignored, local only).

## Style references (sibling repos, read-only)

- `../waveform-karaoke-website` — the 80s-MTV neon look: `src/styles/variables.scss`
  (cyan `#00FFE5`, magenta `#FF006B`, yellow, Raleway + IBM Plex Mono), scanline and
  VHS-tracking overlays in `globals.scss`, perspective grid in `Hero.module.scss`.
- `../waveform-karaoke` — the app; `ResponsiveWaveform` (analyser oscilloscope,
  `smoothing: 0.8`) is what the stage waveform imitates; Kabel Black font files in
  `public/fonts/kabel_black/` (copied into `web/static/fonts/`).
