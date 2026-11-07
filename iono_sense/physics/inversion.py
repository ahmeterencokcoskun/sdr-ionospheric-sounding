"""Gerçek İrtifa İyonogram Tersinim Motoru (Jackson Laminasyon & Tikhonov Regülarizasyonu)."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import scipy.linalg as sla

from iono_sense.physics.plasma_engine import integrate_trapz


@dataclass
class InversionResult:
    true_heights_km: np.ndarray
    plasma_freqs_mhz: np.ndarray
    electron_density_m3: np.ndarray
    foF2_mhz: float
    hmF2_km: float
    NmF2_m3: float
    subpeak_tec_tecu: float
    residual_rms_km: float


class IonogramTrueHeightInverter:
    """Sanal irtifa h'(f) verisinden gerçek irtifa h(f_N) ve elektron yoğunluğu Ne(h) çözer."""

    def __init__(self, default_h0_km: float = 60.0):
        self.default_h0_km = float(default_h0_km)
        self.e = 1.60217663e-19
        self.m_e = 9.1093837e-31
        self.eps0 = 8.8541878e-12
        self.c_km_s = 299792.458
        self.fp_to_ne_si = (4.0 * np.pi**2 * self.eps0 * self.m_e) / (self.e**2)

    @staticmethod
    def _clean_and_sort(
        freqs_mhz: np.ndarray, h_prime_km: np.ndarray, h0_km: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        f = np.asarray(freqs_mhz, dtype=np.float64)
        hp = np.asarray(h_prime_km, dtype=np.float64)
        if f.ndim == 0 or hp.ndim == 0 or len(f) == 0 or len(f) != len(hp):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        valid = np.isfinite(f) & np.isfinite(hp) & (f > 0.0) & (hp >= h0_km)
        if not np.any(valid):
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        f_v, hp_v = f[valid], hp[valid]
        sort_idx = np.argsort(f_v)
        unique_f, unique_idx = np.unique(f_v[sort_idx], return_index=True)
        return unique_f, hp_v[sort_idx][unique_idx]

    @staticmethod
    def build_jackson_kernel(freqs_mhz: np.ndarray) -> np.ndarray:
        """Jackson (1969) alt-üçgensel parçalı doğrusal laminasyon matrisi M[i, k]."""
        N = len(freqs_mhz)
        M = np.zeros((N, N), dtype=np.float64)
        f_steps = np.concatenate([[0.0], freqs_mhz])

        for i in range(N):
            fi = freqs_mhz[i]
            fi_sq = fi**2
            for k in range(i + 1):
                fk_prev, fk = f_steps[k], f_steps[k + 1]
                denom = fk**2 - fk_prev**2
                if denom > 0.0:
                    t1 = np.sqrt(np.maximum(1.0 - (fk_prev / fi) ** 2, 0.0))
                    t2 = np.sqrt(np.maximum(1.0 - (fk / fi) ** 2, 0.0))
                    M[i, k] = (2.0 * fi_sq / denom) * (t1 - t2)
        return M

    def _extract_parameters(
        self,
        true_h: np.ndarray,
        freqs: np.ndarray,
        h0_km: float,
        hp_orig: np.ndarray,
        hp_modeled: np.ndarray,
    ) -> InversionResult:
        if len(freqs) == 0:
            empty = np.empty(0, dtype=np.float64)
            return InversionResult(empty, empty, empty, 0.0, float(h0_km), 0.0, 0.0, 0.0)

        ne_profile = self.fp_to_ne_si * ((freqs * 1e6) ** 2)
        max_idx = int(np.argmax(freqs))
        foF2 = float(freqs[max_idx])
        hmF2 = float(true_h[max_idx])
        NmF2 = float(self.fp_to_ne_si * ((foF2 * 1e6) ** 2))

        h_full = np.concatenate([[h0_km], true_h]) * 1e3
        ne_full = np.concatenate([[0.0], ne_profile])
        subpeak_tec = max(0.0, integrate_trapz(ne_full, h_full) / 1e16)
        rms = float(np.sqrt(np.mean((hp_orig - hp_modeled) ** 2))) if len(hp_orig) > 0 else 0.0

        return InversionResult(true_h, freqs, ne_profile, foF2, hmF2, NmF2, subpeak_tec, rms)

    def invert_jackson(
        self,
        freqs_mhz: np.ndarray,
        h_prime_km: np.ndarray,
        h0_km: Optional[float] = None,
    ) -> InversionResult:
        h0 = float(h0_km if h0_km is not None else self.default_h0_km)
        f_clean, hp_clean = self._clean_and_sort(freqs_mhz, h_prime_km, h0)
        if len(f_clean) == 0:
            return self._extract_parameters(np.empty(0), np.empty(0), h0, np.empty(0), np.empty(0))

        M = self.build_jackson_kernel(f_clean)
        dh = sla.solve_triangular(M, hp_clean - h0, lower=True)
        return self._extract_parameters(h0 + np.cumsum(dh), f_clean, h0, hp_clean, h0 + M @ dh)

    def invert_tikhonov(
        self,
        freqs_mhz: np.ndarray,
        h_prime_km: np.ndarray,
        h0_km: Optional[float] = None,
        reg_lambda: float = 1e-3,
    ) -> InversionResult:
        h0 = float(h0_km if h0_km is not None else self.default_h0_km)
        f_clean, hp_clean = self._clean_and_sort(freqs_mhz, h_prime_km, h0)
        N = len(f_clean)
        if N == 0:
            return self._extract_parameters(np.empty(0), np.empty(0), h0, np.empty(0), np.empty(0))
        if N < 3 or reg_lambda <= 0.0:
            return self.invert_jackson(f_clean, hp_clean, h0_km=h0)

        M = self.build_jackson_kernel(f_clean)
        D = np.eye(N, dtype=np.float64) - np.eye(N, k=-1, dtype=np.float64)
        A = M @ D

        L = np.zeros((N - 2, N), dtype=np.float64)
        idx = np.arange(N - 2)
        L[idx, idx] = 1.0
        L[idx, idx + 1] = -2.0
        L[idx, idx + 2] = 1.0

        H_reg = A.T @ A + float(reg_lambda) * (L.T @ L)
        x = sla.solve(H_reg, A.T @ (hp_clean - h0), assume_a="pos")
        h_true = np.maximum.accumulate(np.maximum(h0 + x, h0))
        return self._extract_parameters(h_true, f_clean, h0, hp_clean, h0 + A @ (h_true - h0))

    def convert_oblique_to_vertical(
        self, f_oblique_mhz: float, delay_ms: float, distance_km: float
    ) -> Tuple[float, float]:
        """Martyn eşdeğer yol teoremi ile eğik yayılımı dikey eşdeğere (f_v, h'_v) dönüştürür."""
        if (
            not (np.isfinite(f_oblique_mhz) and np.isfinite(delay_ms) and np.isfinite(distance_km))
            or f_oblique_mhz <= 0.0
            or delay_ms <= 0.0
            or distance_km < 0.0
        ):
            return 0.0, 0.0

        P_prime = self.c_km_s * (delay_ms * 1e-3)
        if distance_km == 0.0:
            return float(f_oblique_mhz), float(0.5 * P_prime)
        if P_prime <= distance_km:
            return 0.0, 0.0

        h_v = 0.5 * np.sqrt(max(P_prime**2 - distance_km**2, 0.0))
        return float(f_oblique_mhz * (2.0 * h_v / P_prime)), float(h_v)
