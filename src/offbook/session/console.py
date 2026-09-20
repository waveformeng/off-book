"""Console readout: both streams side by side as they resolve, verdicts, running score."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from offbook.asr.base import RecognizerSpec, Token
from offbook.audio.drift import DriftSample
from offbook.compare.verdict import TokenPair, Verdict
from offbook.roles import Live, Reference

_STYLE = {
    Verdict.MATCH: "bold green",
    Verdict.FAIL_LEXICAL: "bold red",
    Verdict.FAIL_TIMING: "bold yellow",
    Verdict.FAIL_MISSED: "red",
    Verdict.FAIL_INSERTED: "magenta",
}


class SessionConsole:
    def __init__(self, quiet: bool = False) -> None:
        self.console = Console(highlight=False)
        self.quiet = quiet

    def header(
        self, spec: RecognizerSpec, mode: str, duplex: bool, rate: int, reference_duration_s: float
    ) -> None:
        self.console.print(
            f"[bold]off book[/bold]  {mode}  transport={rate} Hz  "
            f"{'duplex' if duplex else 'split in/out'}  reference {reference_duration_s:.1f}s"
        )
        self.console.print(f"recognizer  {spec.describe()}")
        self.console.print(
            f"{'REFERENCE':>18}   {'LIVE':<18} {'VERDICT':<14} {'dt':>7}  {'SCORE':>6}"
        )

    def resolved(self, ref: list[Token[Reference]], live: list[Token[Live]], t_s: float) -> None:
        if self.quiet or (not ref and not live):
            return
        line = Text(f"  {t_s:7.2f}s  ", style="dim")
        line.append("ref ▸ ", style="dim cyan")
        line.append(" ".join(t.text for t in ref) or "·", style="cyan")
        line.append("   live ▸ ", style="dim white")
        line.append(" ".join(t.text for t in live) or "·", style="white")
        self.console.print(line)

    def tentative(self, ref: list[Token[Reference]], live: list[Token[Live]], t_s: float) -> None:
        """Tokens the recognizers currently see but have not confirmed. Called after every
        block; the terminal readout ignores it (it would repeat every hop), the web UI
        shows it dimmed ahead of the resolved tokens."""

    def verdict(self, pair: TokenPair, score: float) -> None:
        if self.quiet:
            return
        ref = pair.reference.text if pair.reference is not None else "—"
        live = pair.live.text if pair.live is not None else "—"
        dt = f"{pair.dt_s:+7.2f}" if pair.dt_s is not None else "       "
        line = Text(f"{ref:>18}   ")
        line.append(f"{live:<18} ")
        line.append(f"{pair.verdict.value:<14}", style=_STYLE[pair.verdict])
        line.append(f" {dt}  {score:6.1f}")
        self.console.print(line)

    def drift(self, s: DriftSample) -> None:
        if self.quiet:
            return
        self.console.print(
            f"  {s.transport_s:7.2f}s  drift ADC↔DAC {s.drift_s * 1000:+.2f} ms", style="dim"
        )

    def final(self, score: float, match_rate: float, counts: dict[Verdict, int], path: str) -> None:
        self.console.print()
        self.console.print(f"[bold]final score {score:.1f}[/bold]  match rate {match_rate:.3f}")
        self.console.print("  " + "  ".join(f"{v.value}={n}" for v, n in counts.items()))
        self.console.print(f"session record → {path}")
