"""
Empirical Verification and Adversarial Stress Harness for iono_sense.physics.inversion.
Authored by Challenger M2-1.
"""

import numpy as np
import pytest
import scipy.linalg as sla
from iono_sense.physics.inversion import IonogramTrueHeightInverter, InversionResult


class TestBuddenParabolicAnalytical:
    """1. Test against Budden (1985) parabolic layer profile."""

    def test_budden_parabolic_layer_accuracy(self):
        """
        Budden (1985) parabolic profile:
        z(f) = z0 + ym * (1 - sqrt(1 - (f/fc)^2))
        h'(f) = z0 + 0.5 * ym * (f/fc) * ln((fc + f)/(fc - f))
        
        Using prompt formulation where zm = 0.5 * ym (or ym = 2 * zm):
        h'(f) = z0 + zm * (f/fc) * ln((fc + f)/(fc - f))
        Reconstructs true heights h(f) with error < 1.0 km across entire sounding band.
        """
        fc = 6.0  # Critical frequency [MHz]
        z0 = 100.0  # Base height [km]
        zm = 50.0  # Scaling parameter [km] -> semi-thickness ym = 100.0 km
        ym = 2.0 * zm

        # Sounding band from 0.5 MHz up to 5.85 MHz (97.5% of fc)
        freqs = np.linspace(0.5, 5.85, 80)
        h_prime = z0 + zm * (freqs / fc) * np.log((fc + freqs) / (fc - freqs))
        h_analytical = z0 + ym * (1.0 - np.sqrt(np.maximum(1.0 - (freqs / fc) ** 2, 0.0)))

        inverter = IonogramTrueHeightInverter(default_h0_km=z0)
        res = inverter.invert_jackson(freqs, h_prime, h0_km=z0)

        assert len(res.true_heights_km) == len(freqs)
        errors = np.abs(res.true_heights_km - h_analytical)
        max_error = np.max(errors)
        mean_error = np.mean(errors)

        print(f"\n[Budden Parabolic] Max Error: {max_error:.4f} km, Mean Error: {mean_error:.4f} km")
        assert max_error < 1.0, f"Budden parabolic error {max_error:.4f} km exceeded 1.0 km threshold!"

    def test_budden_parabolic_direct_semi_thickness(self):
        """
        Direct semi-thickness test with zm = ym = 100 km,
        h'(f) = z0 + 0.5 * ym * (f/fc) * ln((fc + f)/(fc - f))
        """
        fc = 7.5
        z0 = 90.0
        ym = 110.0
        freqs = np.linspace(0.8, 7.3, 70)
        h_prime = z0 + 0.5 * ym * (freqs / fc) * np.log((fc + freqs) / (fc - freqs))
        h_analytical = z0 + ym * (1.0 - np.sqrt(np.maximum(1.0 - (freqs / fc) ** 2, 0.0)))

        inverter = IonogramTrueHeightInverter(default_h0_km=z0)
        res = inverter.invert_jackson(freqs, h_prime, h0_km=z0)

        errors = np.abs(res.true_heights_km - h_analytical)
        max_err = np.max(errors)
        print(f"\n[Budden Parabolic Direct ym] Max Error: {max_err:.4f} km")
        assert max_err < 1.0, f"Error {max_err:.4f} km >= 1.0 km"


class TestLinearProfileAnalytical:
    """2. Test linear profile where h'(f) - h0 = 2(h(f) - h0)."""

    def test_linear_profile_virtual_height_doubling(self):
        """
        For a linear electron density profile:
        f_N^2(z) = beta * (z - h0)
        h(f) - h0 = f^2 / beta
        Virtual height: h'(f) - h0 = 2 * (h(f) - h0) = 2 * f^2 / beta.
        Verify that invert_jackson reconstructs exact analytical h(f).
        """
        h0 = 80.0
        beta = 0.5  # MHz^2 / km
        freqs = np.linspace(1.0, 8.0, 50)

        h_true_analytical = h0 + (freqs ** 2) / beta
        h_prime = h0 + 2.0 * (freqs ** 2) / beta

        inverter = IonogramTrueHeightInverter(default_h0_km=h0)
        res = inverter.invert_jackson(freqs, h_prime, h0_km=h0)

        # Theoretical error for piecewise constant gradient should be machine precision
        errors = np.abs(res.true_heights_km - h_true_analytical)
        max_error = np.max(errors)
        print(f"\n[Linear Profile Doubling] Max Error: {max_error:.4e} km")
        assert max_error < 1e-10, f"Linear profile error {max_error} km exceeded tolerance"

        # Verify doubling relation directly: h'(f) - h0 == 2 * (h_true - h0)
        np.testing.assert_allclose(h_prime - h0, 2.0 * (res.true_heights_km - h0), rtol=1e-12)


class TestMatrixStabilityAndSingularity:
    """3. Matrix stability & singularity stress."""

    def test_high_dimension_stability(self):
        """N >= 150 dense grid."""
        N = 200
        freqs = np.linspace(1.0, 15.0, N)
        h0 = 60.0
        h_prime = h0 + 150.0 * (freqs / 15.0) ** 1.5

        inverter = IonogramTrueHeightInverter(default_h0_km=h0)
        M = inverter.build_jackson_kernel(freqs)

        cond_M = np.linalg.cond(M)
        print(f"\n[High Dimension N={N}] Condition number cond(M) = {cond_M:.2e}")

        res = inverter.invert_jackson(freqs, h_prime, h0_km=h0)
        assert len(res.true_heights_km) == N
        assert np.all(np.isfinite(res.true_heights_km))
        assert res.residual_rms_km < 1e-6

    def test_sparse_dimension(self):
        """N = 2, 3 sparse grids."""
        inverter = IonogramTrueHeightInverter(default_h0_km=60.0)

        for N in [2, 3]:
            freqs = np.linspace(3.0, 6.0, N)
            h_prime = np.linspace(120.0, 240.0, N)

            res_j = inverter.invert_jackson(freqs, h_prime)
            assert len(res_j.true_heights_km) == N
            assert np.all(np.isfinite(res_j.true_heights_km))

            res_t = inverter.invert_tikhonov(freqs, h_prime, reg_lambda=0.01)
            assert len(res_t.true_heights_km) == N
            assert np.all(np.isfinite(res_t.true_heights_km))

    def test_non_monotonic_virtual_heights(self):
        """Non-monotonic virtual heights (cusps, severe noise oscillations)."""
        freqs = np.linspace(2.0, 10.0, 30)
        h0 = 60.0
        # Synthetic profile with dip / cusp
        h_prime = 150.0 + 80.0 * np.sin(2.0 * np.pi * freqs / 5.0)
        h_prime = np.maximum(h_prime, h0 + 5.0)

        inverter = IonogramTrueHeightInverter(default_h0_km=h0)
        res_j = inverter.invert_jackson(freqs, h_prime, h0_km=h0)
        assert np.all(np.isfinite(res_j.true_heights_km))

        res_t = inverter.invert_tikhonov(freqs, h_prime, h0_km=h0, reg_lambda=0.05)
        assert np.all(np.isfinite(res_t.true_heights_km))

    def test_tikhonov_noise_and_lambda_sweep(self):
        """
        Test invert_tikhonov with high noise and varying regularization parameter
        lambda in [10^-6, 10^2]. Ensure solutions remain non-negative, finite,
        and well-behaved without NaN/Inf.
        """
        np.random.seed(1337)
        freqs = np.linspace(2.0, 10.0, 40)
        h0 = 60.0
        h_prime_clean = h0 + 100.0 * (freqs / 10.0) ** 2
        # Severe noise: sigma = 35 km
        noise = np.random.normal(0, 35.0, size=len(freqs))
        h_prime_noisy = np.maximum(h_prime_clean + noise, h0 + 1.0)

        inverter = IonogramTrueHeightInverter(default_h0_km=h0)
        lambdas = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]

        for lam in lambdas:
            res = inverter.invert_tikhonov(freqs, h_prime_noisy, h0_km=h0, reg_lambda=lam)
            assert np.all(np.isfinite(res.true_heights_km)), f"NaN/Inf at lambda={lam}"
            assert np.all(np.isfinite(res.electron_density_m3))
            assert res.subpeak_tec_tecu >= 0.0
            print(f"[Tikhonov Sweep] lambda={lam:1e} -> hmF2={res.hmF2_km:.2f} km, TEC={res.subpeak_tec_tecu:.2f} TECU, min_h={np.min(res.true_heights_km):.2f} km")


class TestMartynsEquivalenceTheorem:
    """4. Martyn's equivalence theorem tests."""

    @pytest.fixture
    def inverter(self):
        return IonogramTrueHeightInverter(default_h0_km=60.0)

    def test_zero_distance_limit(self, inverter):
        """Zero-distance limit D -> 0 must perfectly recover vertical sounding."""
        f_oblique = 10.0  # MHz
        delay_ms = 2.5  # ms
        expected_P_prime = inverter.c_km_s * (delay_ms * 1e-3)  # ~749.48 km
        expected_h_v = 0.5 * expected_P_prime

        fv, hv = inverter.convert_oblique_to_vertical(f_oblique, delay_ms, distance_km=0.0)
        assert fv == f_oblique
        assert pytest.approx(hv, rel=1e-6) == expected_h_v

    def test_long_distance(self, inverter):
        """Long distance D = 1000 km."""
        f_oblique = 15.0  # MHz
        # Choose delay corresponding to P' = 1200 km > 1000 km
        P_prime = 1200.0
        delay_ms = (P_prime / inverter.c_km_s) * 1e3
        D = 1000.0

        fv, hv = inverter.convert_oblique_to_vertical(f_oblique, delay_ms, distance_km=D)

        # Expected geometry:
        # hv = 0.5 * sqrt(1200^2 - 1000^2) = 0.5 * sqrt(440000) = 331.662 km
        expected_hv = 0.5 * np.sqrt(P_prime**2 - D**2)
        expected_fv = f_oblique * (2.0 * expected_hv / P_prime)

        assert pytest.approx(hv, rel=1e-5) == expected_hv
        assert pytest.approx(fv, rel=1e-5) == expected_fv
        print(f"\n[Martyn D=1000km] fv={fv:.4f} MHz (from {f_oblique} MHz), hv={hv:.2f} km")

    def test_edge_case_d_greater_than_p_prime(self, inverter):
        """Edge case D > P': must trigger defensive guard cleanly without crashes."""
        f_oblique = 12.0
        delay_ms = 1.0  # P' ~ 299.79 km
        D = 500.0  # D > P' (physically impossible propagation)

        fv, hv = inverter.convert_oblique_to_vertical(f_oblique, delay_ms, distance_km=D)
        assert (fv, hv) == (0.0, 0.0), f"Expected (0.0, 0.0) guard, got ({fv}, {hv})"

        # Boundary P' == D
        delay_exact = (D / inverter.c_km_s) * 1e3
        fv_b, hv_b = inverter.convert_oblique_to_vertical(f_oblique, delay_exact, distance_km=D)
        assert (fv_b, hv_b) == (0.0, 0.0)


class TestDefensiveEdgeCasesAndAdversarialAttacks:
    """Adversarial stress and edge cases exploring vulnerabilities."""

    @pytest.fixture
    def inverter(self):
        return IonogramTrueHeightInverter(default_h0_km=60.0)

    def test_mismatched_lengths_guard(self, inverter):
        """Verify handling of mismatched array lengths."""
        freqs = np.array([2.0, 3.0, 4.0, 5.0])
        h_prime = np.array([100.0, 150.0])
        # If unguarded, raises ValueError
        try:
            res = inverter.invert_jackson(freqs, h_prime)
            assert len(res.true_heights_km) == 0
        except ValueError as e:
            pytest.fail(f"UNHANDLED_EXCEPTION: invert_jackson crashes on mismatched array lengths: {e}")

    def test_martyns_nan_input_guard(self, inverter):
        """Verify convert_oblique_to_vertical with NaN frequency."""
        fv, hv = inverter.convert_oblique_to_vertical(float("nan"), 5.0, 500.0)
        assert not np.isnan(fv), f"NAN_LEAK: fv leaked NaN: {fv}"
        assert (fv, hv) == (0.0, 0.0)

    def test_martyns_nan_delay_guard(self, inverter):
        """Verify convert_oblique_to_vertical with NaN delay."""
        fv, hv = inverter.convert_oblique_to_vertical(10.0, float("nan"), 500.0)
        assert not np.isnan(fv) and not np.isnan(hv), f"NAN_LEAK: ({fv}, {hv})"
        assert (fv, hv) == (0.0, 0.0)

    def test_martyns_nan_distance_guard(self, inverter):
        """Verify convert_oblique_to_vertical with NaN distance."""
        fv, hv = inverter.convert_oblique_to_vertical(10.0, 5.0, float("nan"))
        assert not np.isnan(fv) and not np.isnan(hv), f"NAN_LEAK: ({fv}, {hv})"
        assert (fv, hv) == (0.0, 0.0)

    def test_tikhonov_non_negative_increment_audit(self, inverter):
        """
        Audit whether invert_tikhonov satisfies worker claim:
        'projected onto non-negative layer increments Delta h_k >= 0'.
        """
        freqs = np.linspace(2.0, 10.0, 20)
        h0 = 60.0
        # Highly oscillating virtual heights
        h_prime = h0 + 50.0 + 40.0 * np.sin(np.arange(20))
        res_t = inverter.invert_tikhonov(freqs, h_prime, h0_km=h0, reg_lambda=1e-5)
        dh = np.diff(np.concatenate([[h0], res_t.true_heights_km]))
        has_negative_dh = np.any(dh < -1e-6)
        print(f"\n[Tikhonov Non-Negative Audit] Min dh: {np.min(dh):.4f} km, has negative dh: {has_negative_dh}")
        # Note: In unconstrained linear least-squares, dh can be negative unless constrained/projected.
