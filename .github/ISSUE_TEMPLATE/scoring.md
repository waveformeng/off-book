---
name: Scoring disagreement
about: The score, a verdict, or a transcript looks wrong for a performance
labels: scoring
---

**What the score / verdict was, and what you think it should have been**

**Config**

Paste the `config` object from the session record, or the flags you ran with.

**Is the transcript the problem, or the comparison?**

Run the diagnostic on the reference vocal and say which case this is
(see [docs/tuning.md](../../docs/tuning.md)):

```sh
uv run offbook transcribe path/to/reference_vocal.wav --full
```

- [ ] streaming error rate is high but the one-shot decode is good → chunker tuning
- [ ] the one-shot decode is itself poor → the model can't read this vocal
- [ ] the transcripts are fine; the alignment / verdict / score is what's wrong

**Session record**

Attach `sessions/<id>/session.json`. With the default config it contains no words; if
you re-ran with `--record-transcripts` to investigate, please strip the `text` fields
before attaching unless you hold the rights to the vocal.
