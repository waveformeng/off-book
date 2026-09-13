from offbook.audio.drift import DriftMonitor


def test_drift_is_zero_when_clocks_agree() -> None:
    m = DriftMonitor(48_000, 48_000, 1.0, duplex=False)
    for k in range(100):
        m.observe(k * 1024, 100.0 + k * 1024 / 48_000, k * 1024, 100.0 + k * 1024 / 48_000)
    assert m.max_abs_s == 0.0


def test_drift_tracks_a_fast_output_clock() -> None:
    m = DriftMonitor(48_000, 48_000, 1.0, duplex=False)
    for k in range(200):
        t = k * 1024 / 48_000
        # DAC delivers 1.001× as many frames per wall second as the ADC.
        m.observe(k * 1024, t, int(k * 1024 * 1.001), t)
    assert m.samples[-1].transport_s > 4.0
    assert 0.9e-3 * m.samples[-1].transport_s < m.final_s < 1.1e-3 * m.samples[-1].transport_s


def test_offset_at_first_block_is_removed() -> None:
    m = DriftMonitor(48_000, 44_100, 1.0, duplex=False)
    m.observe(0, 5.0, 4410, 4.9)
    assert m.latest is not None and m.latest.drift_s == 0.0
