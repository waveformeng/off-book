"""The reference/live invariant is enforced by mypy: swapping must fail to type-check."""

import re
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "typing_fixtures" / "swap_roles.py"


def test_every_swap_is_a_type_error() -> None:
    expected = {
        i + 1
        for i, line in enumerate(FIXTURE.read_text().splitlines())
        if line.rstrip().split("  # ")[-1].startswith("ERR ") and not line.startswith('"""')
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(FIXTURE.parent / "mypy.ini"),
            "--no-error-summary",
            str(FIXTURE),
        ],
        capture_output=True,
        text=True,
        cwd=FIXTURE.parents[2],
    )
    reported = {int(m.group(1)) for m in re.finditer(r"swap_roles\.py:(\d+): error", proc.stdout)}
    assert proc.returncode != 0, proc.stdout
    assert expected <= reported, f"unflagged swaps: {sorted(expected - reported)}\n{proc.stdout}"


def test_phantom_roles_cannot_be_instantiated() -> None:
    import pytest

    from offbook.roles import Live, Reference

    with pytest.raises(TypeError):
        Reference()
    with pytest.raises(TypeError):
        Live()
