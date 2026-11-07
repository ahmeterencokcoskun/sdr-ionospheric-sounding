"""
Adversarial Challenge & Stress-Test Suite for Milestone 2.
Empirically stress-tests:
1. iono_sense.physics.inversion (Jackson, Tikhonov, Martyn's equivalence)
2. iono_sense.simulation.fmcw_sounder (Dechirping, Range-FFT, Beat-frequency mapping, Doppler coupling)
"""

import numpy as np
import pytest
import warnings
from iono_sense.physics.inversion import IonogramTrueHeightInverter, InversionResult
from iono_sense.simulation.fmcw_sounder import FMCWIonosondeSimulator


# ============================================================================
# PART 1: ADVERSARIAL STRESS TESTS FOR TRUE-HEIGHT INVERSION
# ============================================================================

class TestInversionAdversarial:
    """Stress tests and edge case exploration for IonogramTrueHeightInverter."""

    @pytest.fixture
    def inverter(self):
        return IonogramTrueHeightInverter(default_h0_km=60.0)

    # 1.1 Empty, Zero, and Single-Frequency Inputs
    def test_empty_inputs(self, inverter):
        res_j = inverter.invert_jackson(np.array([]), np.array([]))
        assert isinstance(res_j, InversionResult)
        assert len(res_j.true_heights_km) == 0
        assert res_j.foF2_mhz == 0.0
        assert res_j.residual_rms_km == 0.0

        res_t = inverter.invert_tikhonov(np.array([]), np.array([]))
        assert isinstance(res_t, InversionResult)
        assert len(res_t.true_heights_km) == 0
        assert res_t.foF2_mhz == 0.0

    def test_single_frequency(self, inverter):
        freqs = np.array([5.0])
        h_prime = np.array([250.0])

        res_j = inverter.invert_jackson(freqs, h_prime)
        assert len(res_j.true_heights_km) == 1
        assert res_j.foF2_mhz == 5.0
        assert np.isfinite(res_j.true_heights_km[0])
        assert res_j.true_heights_km[0] >= 60.0

        res_t = inverter.invert_tikhonov(freqs, h_prime)
        assert len(res_t.true_heights_km) == 1
        assert res_t.foF2_mhz == 5.0

    def test_two_frequencies_fallback(self, inverter):
        freqs = np.array([4.0, 5.0])
        h_prime = np.array([200.0, 250.0])

        res_t = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=0.01)
        assert len(res_t.true_heights_km) == 2
        assert res_t.true_heights_km[1] > res_t.true_heights_km[0]

    # 1.2 Non-finite and Out-of-Bounds Filtering
    def test_negative_zero_nan_inf_frequencies(self, inverter):
        freqs = np.array([-5.0, 0.0, np.nan, 3.0, np.inf, 4.0, 5.0])
        h_prime = np.array([100.0, 120.0, 150.0, 200.0, 220.0, 240.0, 260.0])

        res_j = inverter.invert_jackson(freqs, h_prime)
        # Only 3.0, 4.0, 5.0 are valid
        assert len(res_j.plasma_freqs_mhz) == 3
        np.testing.assert_allclose(res_j.plasma_freqs_mhz, [3.0, 4.0, 5.0])
        assert np.all(np.isfinite(res_j.true_heights_km))

    def test_virtual_heights_below_base(self, inverter):
        # All virtual heights below h0 = 60 km
        freqs = np.array([2.0, 3.0, 4.0])
        h_prime = np.array([30.0, 45.0, 55.0])

        res_j = inverter.invert_jackson(freqs, h_prime, h0_km=60.0)
        assert len(res_j.true_heights_km) == 0
        assert res_j.foF2_mhz == 0.0

    def test_duplicate_and_unsorted_frequencies(self, inverter):
        freqs = np.array([6.0, 2.0, 4.0, 2.0, 5.0])
        h_prime = np.array([300.0, 150.0, 220.0, 150.0, 270.0])

        res = inverter.invert_jackson(freqs, h_prime)
        # Should be strictly sorted: 2, 4, 5, 6
        assert len(res.plasma_freqs_mhz) == 4
        np.testing.assert_allclose(res.plasma_freqs_mhz, [2.0, 4.0, 5.0, 6.0])
        assert np.all(np.diff(res.plasma_freqs_mhz) > 0)

    # 1.3 High Condition Numbers & Closely Spaced Frequencies
    def test_closely_spaced_frequencies_condition_number(self, inverter):
        # Dense frequencies: spacing 100 Hz (1e-4 MHz)
        f0 = 5.0
        delta_f = 1e-4
        freqs = np.linspace(f0, f0 + 20 * delta_f, 21)
        h_prime = 200.0 + 10.0 * np.sqrt(np.arange(len(freqs)))

        M = inverter.build_jackson_kernel(freqs)
        cond_M = np.linalg.cond(M)
        assert np.isfinite(cond_M)

        res_j = inverter.invert_jackson(freqs, h_prime)
        assert np.all(np.isfinite(res_j.true_heights_km))

        res_t = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=1e-2)
        assert np.all(np.isfinite(res_t.true_heights_km))

    # 1.4 Non-Monotonic Virtual Heights (Simulated Valleys & Cusps)
    def test_non_monotonic_virtual_heights(self, inverter):
        freqs = np.linspace(2.0, 10.0, 17)
        h_prime = 250.0 + 50.0 * np.sin(np.pi * (freqs - 2.0) / 4.0)
        h_prime = np.maximum(h_prime, 70.0)

        res_j = inverter.invert_jackson(freqs, h_prime)
        assert np.all(np.isfinite(res_j.true_heights_km))
        assert res_j.residual_rms_km < 1e-6  # Forward model matches exactly

        res_t = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=0.1)
        assert np.all(np.isfinite(res_t.true_heights_km))
        assert res_t.subpeak_tec_tecu >= 0.0

    # 1.5 Severe Noise (Gaussian and Impulsive Spikes)
    def test_severe_gaussian_noise(self, inverter):
        np.random.seed(42)
        freqs = np.linspace(2.0, 10.0, 25)
        h_prime_clean = 100.0 + 150.0 * (freqs / 10.0) ** 2
        noise = np.random.normal(0, 30.0, size=len(freqs))
        h_prime_noisy = np.maximum(h_prime_clean + noise, 65.0)

        res_j = inverter.invert_jackson(freqs, h_prime_noisy)
        res_t = inverter.invert_tikhonov(freqs, h_prime_noisy, reg_lambda=1.0)

        assert np.all(np.isfinite(res_j.true_heights_km))
        assert np.all(np.isfinite(res_t.true_heights_km))

        # Tikhonov curvature should be lower (smoother) than Jackson under noise
        curv_j = np.sum(np.diff(res_j.true_heights_km, n=2) ** 2)
        curv_t = np.sum(np.diff(res_t.true_heights_km, n=2) ** 2)
        assert curv_t < curv_j

    def test_impulsive_spike_noise(self, inverter):
        freqs = np.linspace(2.0, 10.0, 20)
        h_prime = 150.0 + 10.0 * freqs
        h_prime[10] += 300.0

        res_j = inverter.invert_jackson(freqs, h_prime)
        res_t = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=0.5)

        assert np.all(np.isfinite(res_j.true_heights_km))
        assert np.all(np.isfinite(res_t.true_heights_km))

    # 1.6 Extreme Regularization Lambda (1e-9 to 1e6)
    def test_extreme_tikhonov_lambda(self, inverter):
        freqs = np.linspace(2.0, 10.0, 20)
        h_prime = 120.0 + 80.0 * (freqs / 10.0) ** 2

        lambdas = [1e-9, 1e-6, 1e-3, 1.0, 1e2, 1e4, 1e6]
        for lam in lambdas:
            res = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=lam)
            assert np.all(np.isfinite(res.true_heights_km)), f"Failed at lambda={lam}"
            assert res.foF2_mhz == 10.0
            assert res.subpeak_tec_tecu >= 0.0

    # 1.7 Extreme Frequency Ranges (0.1 MHz to 30 MHz)
    def test_extreme_frequency_span(self, inverter):
        freqs = np.geomspace(0.1, 30.0, 40)
        h_prime = 100.0 + 200.0 * (freqs / 30.0)

        res_j = inverter.invert_jackson(freqs, h_prime)
        assert np.all(np.isfinite(res_j.true_heights_km))
        assert res_j.foF2_mhz == 30.0

        res_t = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=0.01)
        assert np.all(np.isfinite(res_t.true_heights_km))

    # 1.8 Martyn's Equivalent Path Oblique Conversion
    def test_martyns_equivalence_bounds(self, inverter):
        # P' <= D: should guard and return (0.0, 0.0)
        fv, hv = inverter.convert_oblique_to_vertical(10.0, delay_ms=3.0, distance_km=1500.0)
        assert (fv, hv) == (0.0, 0.0)

        # Exact boundary P' == D
        delay_exact = 1000.0 / inverter.c_km_s * 1e3
        fv_b, hv_b = inverter.convert_oblique_to_vertical(10.0, delay_ms=delay_exact, distance_km=1000.0)
        assert (fv_b, hv_b) == (0.0, 0.0)

        # D = 0: pure vertical sounding
        fv_0, hv_0 = inverter.convert_oblique_to_vertical(14.0, delay_ms=2.0, distance_km=0.0)
        expected_hv = 0.5 * inverter.c_km_s * 0.002
        assert fv_0 == 14.0
        assert pytest.approx(hv_0, rel=1e-5) == expected_hv

        # Transcontinental large distance D = 4000 km
        fv_l, hv_l = inverter.convert_oblique_to_vertical(15.0, delay_ms=16.0, distance_km=4000.0)
        assert 0.0 < fv_l < 15.0
        assert hv_l > 0.0
        assert np.isfinite(fv_l) and np.isfinite(hv_l)

        # Negative and zero inputs
        assert inverter.convert_oblique_to_vertical(-5.0, 5.0, 100.0) == (0.0, 0.0)
        assert inverter.convert_oblique_to_vertical(5.0, -1.0, 100.0) == (0.0, 0.0)
        assert inverter.convert_oblique_to_vertical(5.0, 5.0, -100.0) == (0.0, 0.0)

    # 1.9 Adversarial Bug Spotting: Mismatched Input Lengths Guard
    def test_mismatched_array_lengths(self, inverter):
        """CHALLENGE: Verify whether _clean_and_sort guards against mismatched array lengths."""
        freqs = np.array([2.0, 3.0, 4.0])
        h_prime = np.array([100.0, 150.0])
        # If unguarded, np.isfinite(f) & np.isfinite(hp) raises ValueError: operands could not be broadcast
        try:
            res = inverter.invert_jackson(freqs, h_prime)
            assert len(res.true_heights_km) == 0
        except ValueError as e:
            pytest.fail(f"BUG CONFIRMED: invert_jackson crashes with unhandled ValueError on mismatched lengths: {e}")

    # 1.10 Adversarial Bug Spotting: Martyn's Equivalence NaN/Inf Leakage
    def test_martyns_nan_inf_guards(self, inverter):
        """CHALLENGE: Verify whether convert_oblique_to_vertical handles NaN / Inf parameters."""
        fv_nan_f, hv_nan_f = inverter.convert_oblique_to_vertical(float("nan"), 5.0, 1000.0)
        assert (fv_nan_f, hv_nan_f) == (0.0, 0.0), f"BUG: Leaked NaN in frequency conversion: ({fv_nan_f}, {hv_nan_f})"

        fv_nan_d, hv_nan_d = inverter.convert_oblique_to_vertical(10.0, float("nan"), 1000.0)
        assert (fv_nan_d, hv_nan_d) == (0.0, 0.0), f"BUG: Leaked NaN in delay conversion: ({fv_nan_d}, {hv_nan_d})"

        fv_nan_dist, hv_nan_dist = inverter.convert_oblique_to_vertical(10.0, 5.0, float("nan"))
        assert (fv_nan_dist, hv_nan_dist) == (0.0, 0.0), f"BUG: Leaked NaN in distance conversion: ({fv_nan_dist}, {hv_nan_dist})"


# ============================================================================
# PART 2: ADVERSARIAL STRESS TESTS FOR FMCW SOUNDER
# ============================================================================

class TestFMCWSounderAdversarial:
    """Stress tests and edge case exploration for FMCWIonosondeSimulator."""

    @pytest.fixture
    def sounder(self):
        return FMCWIonosondeSimulator(
            f_start_mhz=2.0,
            f_stop_mhz=12.0,
            chirp_rate_khz_per_sec=100.0,
        )

    # 2.1 Beat Frequency Mapping & Edge Cases
    def test_beat_frequency_mapping_invertibility(self, sounder):
        for h in [60.0, 150.0, 300.0, 600.0]:
            fb = sounder.compute_beat_frequency(h)
            h_rec = sounder.compute_virtual_height_from_beat(fb)
            assert pytest.approx(h_rec, rel=1e-6) == h

    def test_beat_frequency_pathological_inputs(self, sounder):
        assert sounder.compute_beat_frequency(0.0) == 0.0
        assert sounder.compute_beat_frequency(-100.0) == 0.0
        assert sounder.compute_beat_frequency(np.nan) == 0.0

        assert sounder.compute_virtual_height_from_beat(0.0) == 0.0
        assert sounder.compute_virtual_height_from_beat(-50.0) == 0.0
        assert sounder.compute_virtual_height_from_beat(np.nan) == 0.0

    def test_zero_and_negative_chirp_rate(self):
        sounder_zero = FMCWIonosondeSimulator(chirp_rate_khz_per_sec=0.0)
        assert sounder_zero.compute_beat_frequency(300.0) == 0.0
        assert sounder_zero.compute_virtual_height_from_beat(100.0) == 0.0

        sounder_neg = FMCWIonosondeSimulator(chirp_rate_khz_per_sec=-50.0)
        assert sounder_neg.compute_virtual_height_from_beat(100.0) == 0.0

    # 2.2 Dechirping Mixer
    def test_dechirp_signal_shapes_and_lengths(self, sounder):
        s_tx = np.exp(1j * np.linspace(0, 100, 1000))
        s_rx = np.exp(1j * np.linspace(0, 100, 1000))
        s_if = sounder.dechirp_signal(s_rx, s_tx)
        assert len(s_if) == 1000
        np.testing.assert_allclose(np.abs(s_if), 1.0, atol=1e-6)

        s_short = s_rx[:350]
        s_if_short = sounder.dechirp_signal(s_short, s_tx)
        assert len(s_if_short) == 350

        assert len(sounder.dechirp_signal(np.array([]), s_tx)) == 0
        assert len(sounder.dechirp_signal(s_rx, np.array([]))) == 0

    # 2.3 Adversarial Bug Spotting: Range-FFT Dechirped Spectrum Monotonicity Bug
    def test_range_spectrum_peak_and_non_zero(self, sounder):
        """CHALLENGE: Verify that simulate_dechirped_range_spectrum produces valid non-zero power with peak at target height."""
        target_h = 320.0
        max_h = 650.0
        n_bins = 200

        profile = sounder.simulate_dechirped_range_spectrum(
            h_prime_km=target_h,
            att_factor=1.0,
            n_height_bins=n_bins,
            max_height_km=max_h,
        )
        assert len(profile) == n_bins
        # Check if output is not completely silent/zeros
        total_p = np.sum(profile)
        if total_p == 0.0:
            pytest.fail("BUG CONFIRMED: simulate_dechirped_range_spectrum produced all zeros! (np.fft.fftfreq negative Nyquist breaks np.interp monotonicity)")

        heights = np.linspace(60.0, max_h, n_bins)
        peak_idx = int(np.argmax(profile))
        peak_h = heights[peak_idx]
        bin_res = (max_h - 60.0) / n_bins
        assert abs(peak_h - target_h) <= 2.0 * bin_res

    def test_range_spectrum_edge_inputs(self, sounder):
        p0 = sounder.simulate_dechirped_range_spectrum(0.0, 1.0, 100, 600.0)
        assert np.all(p0 == 0.0)

        p_att0 = sounder.simulate_dechirped_range_spectrum(250.0, 0.0, 100, 600.0)
        assert np.all(p_att0 == 0.0)

        p_nan = sounder.simulate_dechirped_range_spectrum(np.nan, 1.0, 100, 600.0)
        assert np.all(p_nan == 0.0)

    # 2.4 Adversarial Bug Spotting: Zero Chirp Rate Division-by-Zero in simulate_dechirped_range_spectrum
    def test_range_spectrum_zero_chirp_rate(self):
        """CHALLENGE: Verify whether simulate_dechirped_range_spectrum divides by zero when chirp_rate=0."""
        sounder_zero = FMCWIonosondeSimulator(chirp_rate_khz_per_sec=0.0)
        with warnings.catch_warnings(record=True) as warn_records:
            warnings.simplefilter("always")
            p = sounder_zero.simulate_dechirped_range_spectrum(250.0, 1.0, 100, 650.0)
            zero_div_warns = [w for w in warn_records if "divide by zero" in str(w.message)]
            if len(zero_div_warns) > 0:
                pytest.fail(f"BUG CONFIRMED: Division by zero encountered when chirp_rate=0 in simulate_dechirped_range_spectrum: {zero_div_warns[0].message}")
        assert np.all(p == 0.0)

    # 2.5 Oracle Range-FFT Peak Detection across SNR (-30 dB to +30 dB)
    def test_range_fft_snr_sweep_oracle(self, sounder):
        """Empirical SNR sweep using correct rfftfreq formulation to determine SNR breakdown limit."""
        target_h = 280.0
        max_h = 600.0
        n_bins = 150
        heights = np.linspace(60.0, max_h, n_bins)
        bin_res = (max_h - 60.0) / n_bins

        fb = sounder.compute_beat_frequency(target_h)
        fb_max = sounder.compute_beat_frequency(max_h)
        fs = 2.5 * fb_max
        n_samples = 512
        t = np.arange(n_samples) / fs

        snr_levels_db = [-30, -15, -5, 0, 10, 20, 30]
        peak_errors = []

        clean_sig = np.exp(1j * 2.0 * np.pi * fb * t)
        sig_pow = np.mean(np.abs(clean_sig) ** 2)

        for snr_db in snr_levels_db:
            np.random.seed(123)
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_pow = sig_pow / snr_linear
            noise = (
                np.random.normal(0, np.sqrt(noise_pow / 2), n_samples)
                + 1j * np.random.normal(0, np.sqrt(noise_pow / 2), n_samples)
            )
            rx_sig = clean_sig + noise

            # Correct Range-FFT spectrum calculation using rfftfreq
            window = np.hanning(n_samples)
            spec_full = np.abs(np.fft.fft(rx_sig * window)) ** 2
            n_half = n_samples // 2 + 1
            freqs_fft = np.fft.rfftfreq(n_samples, d=1.0 / fs)
            heights_fft = (sounder.c * freqs_fft) / (2.0 * sounder.chirp_rate * 1e3)
            power_prof = np.interp(heights, heights_fft, spec_full[:n_half], left=0.0, right=0.0)

            peak_h = heights[np.argmax(power_prof)]
            err = abs(peak_h - target_h)
            peak_errors.append((snr_db, err))

        # Peak detection breakdown threshold:
        # At SNR >= 0 dB, peak detection should be locked within 1-2 bins
        for snr_db, err in peak_errors:
            if snr_db >= 0:
                assert err <= 2.0 * bin_res, f"SNR {snr_db} dB error {err} km exceeds 2 bins ({2.0 * bin_res} km)"

    # 2.6 Doppler Shift Injection into Beat Frequency
    def test_doppler_shift_coupling(self, sounder):
        true_h = 300.0
        fb_clean = sounder.compute_beat_frequency(true_h)

        doppler_shifts = [-10.0, -2.0, 0.0, 5.0, 15.0]  # Hz
        for fd in doppler_shifts:
            fb_doppler = fb_clean + fd
            h_apparent = sounder.compute_virtual_height_from_beat(fb_doppler)
            expected_delta_h = (sounder.c * fd) / (2.0 * sounder.chirp_rate * 1e3)
            assert pytest.approx(h_apparent - true_h, rel=1e-5) == expected_delta_h
