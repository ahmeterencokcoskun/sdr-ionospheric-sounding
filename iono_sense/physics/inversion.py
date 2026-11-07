"""
True-Height Profile Inversion Engine for Vertical and Oblique Ionograms.
Implements Jackson (1969) piecewise linear lamination matrix inversion,
Tikhonov regularized least-squares inversion with 2nd-order discrete curvature penalty,
and Martyn's equivalent path theorem for oblique-to-vertical conversion.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import scipy.linalg as sla


@dataclass
class InversionResult:
    """
    Structured container for true-height ionospheric profile inversion results.

    Attributes:
        true_heights_km: 1D array of reconstructed true heights h(f) [km].
        plasma_freqs_mhz: 1D array of plasma critical frequencies f_N [MHz].
        electron_density_m3: 1D array of electron densities N_e(h) [m^-3].
        foF2_mhz: Critical ordinary frequency of the F2 peak layer [MHz].
        hmF2_km: True height of maximum electron density hmF2 [km].
        NmF2_m3: Peak electron density of the F2 layer [m^-3].
        subpeak_tec_tecu: Sub-peak Total Electron Content up to hmF2 [TECU].
        residual_rms_km: RMS residual between input h' and forward-modeled h' [km].
    """
    true_heights_km: np.ndarray
    plasma_freqs_mhz: np.ndarray
    electron_density_m3: np.ndarray
    foF2_mhz: float
    hmF2_km: float
    NmF2_m3: float
    subpeak_tec_tecu: float
    residual_rms_km: float


class IonogramTrueHeightInverter:
    """
    True-Height Ionogram Inversion Engine.

    Solves the inverse problem h'(f) -> h(f_N) and N_e(h) using:
      1. Jackson (1969) piecewise linear lamination matrix kernel (M_ik).
      2. Tikhonov regularized least-squares with 2nd-order discrete curvature penalty.
      3. Martyn's equivalent path theorem for oblique-to-vertical ionogram transformation.
    """

    def __init__(self, default_h0_km: float = 60.0):
        self.default_h0_km = float(default_h0_km)

        # SI fundamental physical constants matching iono_sense.physics.plasma_engine
        self.e = 1.60217663e-19
        self.m_e = 9.1093837e-31
        self.eps0 = 8.8541878e-12
        self.c_km_s = 299792.458
        self.c_mps = 2.99792458e8
        self.fp_to_ne_si = (4.0 * np.pi**2 * self.eps0 * self.m_e) / (self.e**2)

    def _clean_and_sort(
        self, freqs_mhz: np.ndarray, h_prime_km: np.ndarray, h0_km: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cleans and sorts frequency and virtual height arrays:
        Filters non-finite values, enforces f > 0, enforces h' >= h0, sorts ascending by frequency,
        and eliminates duplicates.
        """
        f = np.asarray(freqs_mhz, dtype=np.float64)
        hp = np.asarray(h_prime_km, dtype=np.float64)

        if f.ndim == 0 or hp.ndim == 0 or len(f) == 0 or len(hp) == 0 or len(f) != len(hp):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        # Clamp virtual height to at least h0 + 1e-3 to satisfy physical positivity
        valid = np.isfinite(f) & np.isfinite(hp) & (f > 0.0) & (hp >= h0_km)
        if not np.any(valid):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        f_v = f[valid]
        hp_v = hp[valid]

        # Strictly sort frequencies ascending
        sort_idx = np.argsort(f_v)
        f_s = f_v[sort_idx]
        hp_s = hp_v[sort_idx]

        # Remove duplicate frequencies if present
        unique_f, unique_idx = np.unique(f_s, return_index=True)
        return unique_f, hp_s[unique_idx]

    def build_jackson_kernel(self, freqs_mhz: np.ndarray) -> np.ndarray:
        """
        Constructs the Jackson (1969) lower-triangular lamination kernel matrix M.

        M[i, k] = (2 * f_i^2 / (f_k^2 - f_{k-1}^2)) * (sqrt(1 - (f_{k-1}/f_i)^2) - sqrt(1 - (f_k/f_i)^2))
        for 0 <= k <= i, and 0 for k > i.

        Args:
            freqs_mhz: Strictly increasing 1D array of frequencies [MHz] (length N).

        Returns:
            M: (N, N) lower-triangular matrix of kernel coefficients.
        """
        N = len(freqs_mhz)
        M = np.zeros((N, N), dtype=np.float64)
        f_steps = np.concatenate([[0.0], freqs_mhz])

        for i in range(N):
            fi = freqs_mhz[i]
            fi_sq = fi**2
            for k in range(i + 1):
                fk_prev = f_steps[k]
                fk = f_steps[k + 1]
                denom = fk**2 - fk_prev**2
                if denom <= 0.0:
                    continue
                term1 = np.sqrt(np.maximum(1.0 - (fk_prev / fi)**2, 0.0))
                term2 = np.sqrt(np.maximum(1.0 - (fk / fi)**2, 0.0))
                M[i, k] = (2.0 * fi_sq / denom) * (term1 - term2)

        return M

    def _extract_parameters(
        self,
        true_h: np.ndarray,
        freqs: np.ndarray,
        h0_km: float,
        hp_orig: np.ndarray,
        hp_modeled: np.ndarray,
    ) -> InversionResult:
        """Extracts plasma parameters and packages into InversionResult."""
        if len(freqs) == 0:
            return InversionResult(
                true_heights_km=np.empty(0, dtype=np.float64),
                plasma_freqs_mhz=np.empty(0, dtype=np.float64),
                electron_density_m3=np.empty(0, dtype=np.float64),
                foF2_mhz=0.0,
                hmF2_km=float(h0_km),
                NmF2_m3=0.0,
                subpeak_tec_tecu=0.0,
                residual_rms_km=0.0,
            )

        ne_profile = self.fp_to_ne_si * ((freqs * 1e6)**2)
        max_idx = int(np.argmax(freqs))
        foF2 = float(freqs[max_idx])
        hmF2 = float(true_h[max_idx])
        NmF2 = float(self.fp_to_ne_si * ((foF2 * 1e6)**2))

        # Sub-peak TEC trapezoidal integration from h0 to hmF2 (in TECU: 1 TECU = 1e16 m^-2)
        h_full = np.concatenate([[h0_km], true_h])
        ne_full = np.concatenate([[0.0], ne_profile])

        # Ensure monotonic height vector for integration
        if hasattr(np, "trapezoid"):
            tec_m2 = float(np.trapezoid(ne_full, h_full * 1e3))
        else:
            tec_m2 = float(np.trapz(ne_full, h_full * 1e3))
        subpeak_tec = max(0.0, tec_m2 / 1e16)

        if len(hp_orig) > 0 and len(hp_modeled) > 0:
            residual_rms = float(np.sqrt(np.mean((hp_orig - hp_modeled)**2)))
        else:
            residual_rms = 0.0

        return InversionResult(
            true_heights_km=true_h,
            plasma_freqs_mhz=freqs,
            electron_density_m3=ne_profile,
            foF2_mhz=foF2,
            hmF2_km=hmF2,
            NmF2_m3=NmF2,
            subpeak_tec_tecu=subpeak_tec,
            residual_rms_km=residual_rms,
        )

    def invert_jackson(
        self,
        freqs_mhz: np.ndarray,
        h_prime_km: np.ndarray,
        h0_km: Optional[float] = None,
    ) -> InversionResult:
        """
        Inverts virtual heights h'(f) using Jackson piecewise linear lamination.

        Args:
            freqs_mhz: Sounding frequencies [MHz].
            h_prime_km: Measured virtual heights [km].
            h0_km: Base ionospheric height [km] (default: 60.0).

        Returns:
            InversionResult containing h(f), N_e(h), foF2, hmF2, NmF2, subpeak TEC, residual RMS.
        """
        h0 = float(h0_km if h0_km is not None else self.default_h0_km)
        f_clean, hp_clean = self._clean_and_sort(freqs_mhz, h_prime_km, h0)
        if len(f_clean) == 0:
            return self._extract_parameters(np.empty(0), np.empty(0), h0, np.empty(0), np.empty(0))

        M = self.build_jackson_kernel(f_clean)
        # Solve M * dh = (hp - h0) via forward substitution (lower-triangular)
        dh = sla.solve_triangular(M, hp_clean - h0, lower=True)
        h_true = h0 + np.cumsum(dh)
        hp_modeled = h0 + M @ dh

        return self._extract_parameters(h_true, f_clean, h0, hp_clean, hp_modeled)

    def invert_tikhonov(
        self,
        freqs_mhz: np.ndarray,
        h_prime_km: np.ndarray,
        h0_km: Optional[float] = None,
        reg_lambda: float = 1e-3,
    ) -> InversionResult:
        """
        Inverts virtual heights h'(f) using two-stage Tikhonov regularized least-squares with
        2nd-order discrete curvature regularization and cumulative monotonic post-projection.

        Objective: min_x ||A x - (h' - h0)||^2 + lambda ||L x||^2
        where x = h - h0, A = M @ D, L is the second-difference curvature matrix.

        Args:
            freqs_mhz: Sounding frequencies [MHz].
            h_prime_km: Measured virtual heights [km].
            h0_km: Base ionospheric height [km] (default: 60.0).
            reg_lambda: Regularization weight parameter lambda >= 0 (default: 1e-3).

        Returns:
            InversionResult containing smoothed h(f), N_e(h), foF2, hmF2, NmF2, subpeak TEC, residual RMS.
        """
        h0 = float(h0_km if h0_km is not None else self.default_h0_km)
        f_clean, hp_clean = self._clean_and_sort(freqs_mhz, h_prime_km, h0)
        N = len(f_clean)
        if N == 0:
            return self._extract_parameters(np.empty(0), np.empty(0), h0, np.empty(0), np.empty(0))
        if N < 3 or reg_lambda <= 0.0:
            return self.invert_jackson(f_clean, hp_clean, h0_km=h0)

        M = self.build_jackson_kernel(f_clean)
        # Discrete difference operator D such that dh = D @ x
        D = np.eye(N, dtype=np.float64)
        for i in range(1, N):
            D[i, i - 1] = -1.0
        A = M @ D

        # Second-order curvature operator L with shape (N-2, N)
        L = np.zeros((N - 2, N), dtype=np.float64)
        for i in range(N - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0

        # Normal equations: (A^T A + lambda * L^T L) x = A^T (hp - h0)
        H_reg = A.T @ A + float(reg_lambda) * (L.T @ L)
        rhs = A.T @ (hp_clean - h0)
        x = sla.solve(H_reg, rhs, assume_a="pos")
        h_true = np.maximum.accumulate(np.maximum(h0 + x, h0))
        hp_modeled = h0 + A @ (h_true - h0)

        return self._extract_parameters(h_true, f_clean, h0, hp_clean, hp_modeled)

    def convert_oblique_to_vertical(
        self,
        f_oblique_mhz: float,
        delay_ms: float,
        distance_km: float,
    ) -> Tuple[float, float]:
        """
        Applies Martyn's Equivalent Path Theorem to convert an oblique sounding measurement
        (f_oblique, tau, D) into equivalent vertical sounding parameters (f_v, h'_v).

        P' = c * delay
        h'_v = 0.5 * sqrt(P'^2 - D^2)
        f_v = f_oblique * cos(phi_0) = f_oblique * (2 * h'_v / P')

        When distance_km == 0: f_v = f_oblique, h'_v = 0.5 * P'.

        Args:
            f_oblique_mhz: Transmitted oblique carrier frequency [MHz].
            delay_ms: Measured group propagation delay [ms].
            distance_km: Great-circle ground distance between TX and RX [km].

        Returns:
            Tuple[float, float]: (f_v_mhz, h_prime_v_km) equivalent vertical frequency and height.
        """
        if (
            not (np.isfinite(f_oblique_mhz) and np.isfinite(delay_ms) and np.isfinite(distance_km))
            or f_oblique_mhz <= 0.0
            or delay_ms <= 0.0
            or distance_km < 0.0
        ):
            return 0.0, 0.0

        P_prime = self.c_km_s * (delay_ms * 1e-3)

        if distance_km == 0.0:
            h_v = 0.5 * P_prime
            return float(f_oblique_mhz), float(h_v)

        if P_prime <= distance_km:
            return 0.0, 0.0

        h_v = 0.5 * np.sqrt(max(P_prime**2 - distance_km**2, 0.0))
        f_v = float(f_oblique_mhz * (2.0 * h_v / P_prime))
        return f_v, float(h_v)
