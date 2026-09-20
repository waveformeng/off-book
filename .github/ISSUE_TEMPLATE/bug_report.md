---
name: Bug report
about: Something crashed, hung, or did the wrong thing
labels: bug
---

**What happened**

**What you expected**

**How to reproduce**

```sh
uv run offbook …
```

**Environment**

- macOS version / chip / memory:
- `uv run offbook check-models` output (model ids, revisions, hashes):
- recognizer (word / phoneme), dtype:
- live or replay:

**Session record**

Attach `sessions/<id>/session.json` if there is one. With the default config it contains
timing, verdicts and scores but no words. Do not attach audio.
