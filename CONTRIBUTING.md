# Contributing

Thanks for looking. Off Book is small and opinionated; the fastest way to a merged change
is to keep it that way.

## Setup

macOS on Apple Silicon (the recognizers run on MLX) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run offbook check-models      # ~3.7 GB, once
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy && uv run pytest
OFFBOOK_MODEL_TESTS=1 uv run pytest tests/test_replay_models.py   # real models, ~20 s
```

CI runs exactly those commands plus a check that `schema/session.schema.json` is current.
If you touch `config.py` or `session/record.py`, regenerate it:

```sh
uv run offbook schema
```

## The invariants are not up for negotiation

[DESIGN.md](DESIGN.md) lists four guarantees and the tests that enforce them. A pull
request that weakens one — a lyric input, a way to hear the reference, a path that lets a
reference token be treated as a live one, a nondeterministic decode — will not be merged
regardless of what it enables. If a feature seems to need one of those, open an issue
first and describe the feature; there is usually another way.

Concretely:

- No lyric text in, out, or on disk by default. The stage view shows the singer's own
  confirmed words and nothing from the reference stream.
- `ReferencePCM` stays without an array interface or a play path; `_BackingPlayer` stays
  the only object that can feed an output device.
- Anything generic over a stream is generic over `Role` and typed `Reference` or `Live`,
  never both.
- Greedy decoding, sample-count boundaries, pinned revisions with verified hashes.

## What a good change looks like

- **One thing per PR.** A recognizer, a scoring policy, a UI change and a tuning
  experiment are four PRs.
- **Scoring changes come with numbers.** If a change can move a score, replay a real
  performance before and after (`uv run offbook replay …`) and put the two records' final
  scores and counts in the PR. The tuning history in [docs/tuning.md](docs/tuning.md) is
  the model.
- **Config knobs, not code paths.** New behaviour that a product decision might want to
  turn on or off goes in `SessionConfig`, with a description and a default that preserves
  today's scores.
- **Match the code around you.** Strict mypy, ruff, line length 100, docstrings that say
  why. Comments explain the non-obvious; the README and `docs/` explain the system.
- **A new recognizer** implements `Backend` (`asr/base.py`), registers in
  `asr/registry.py`, pins its model in `models.lock.json`, and passes
  `test_replay_models.py` twice in a row with identical output. Both streams must run
  through it; `RecognizerSpec` equality is asserted at session start.

## Audio and session data

Never commit audio, session records or model weights. `.gitignore` covers the obvious
extensions; if you add a format, add it there too. Reference vocals in particular are
licensed material — test fixtures are generated (`say`, on macOS) rather than recorded.

## Reporting a problem

Use the issue templates. For a scoring complaint, the session record (with
`record_transcripts` off, which is the default — it contains no words) and the config
that produced it are what make it reproducible.
