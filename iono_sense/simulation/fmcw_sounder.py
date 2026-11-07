#!/usr/bin/env python3
"""Fiziksel Dikey FMCW İyonosond Radarı ve Breit-Tuve İyonogram Çözücüsü."""

from typing import Dict, Optional
import numpy as np

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine, integrate_trapz


class FMCWIonosondeSimulator:
    """Breit-Tuve grup indisi integrasyonu ve Range-FFT spektrumu ile dikey iyonosond simülatörü."""

    def __init__(
        self,
        plasma_engine: Optional[IonosphericPlasmaEngine] = None,
        f_start_mhz: float = 1.0,
        f_stop_mhz: float = 15.0,
        chirp_rate_khz_per_sec: float = 100.0,
        c_mps: float = 2.99792458e8,
    ):
        self.plasma = plasma_engine or IonosphericPlasmaEngine(is_daytime=True)
        self.f_start_mhz = f_start_mhz
        self.f_stop_mhz = f_stop_mhz
        self.chirp_rate = chirp_rate_khz_per_sec * 1e3
        self.c = c_mps

    def compute_beat_frequency(self, virtual_height_km: float) -> float:
        if np.isnan(virtual_height_km) or virtual_height_km <= 0.0:
            return 0.0
        return float((2.0 * (virtual_height_km * 1e3) / self.c) * self.chirp_rate)

    def compute_virtual_height_from_beat(self, fb_hz: float) -> float:
        if np.isnan(fb_hz) or fb_hz <= 0.0 or self.chirp_rate <= 0.0:
            return 0.0
        return float((self.c * fb_hz) / (2.0 * self.chirp_rate * 1e3))

    @staticmethod
    def dechirp_signal(s_rx: np.ndarray, s_tx: np.ndarray) -> np.ndarray:
        n = min(len(s_rx), len(s_tx))
        return s_rx[:n] * np.conj(s_tx[:n]) if n > 0 else np.array([], dtype=np.complex128)

    def simulate_dechirped_range_spectrum(
        self,
        h_prime_km: float,
        att_factor: float,
        n_height_bins: int,
        max_height_km: float,
    ) -> np.ndarray:
        if (
            not np.isfinite(h_prime_km)
            or h_prime_km <= 0.0
            or att_factor <= 0.0
            or self.chirp_rate <= 0.0
            or n_height_bins <= 0
        ):
            return np.zeros(max(0, n_height_bins), dtype=np.float64)

        fb = self.compute_beat_frequency(h_prime_km)
        fs_if = max(2.5 * self.compute_beat_frequency(max_height_km), 1000.0)
        n_fft = max(2 * n_height_bins, 256)

        t = np.arange(n_fft) / fs_if
        s_if = att_factor * np.exp(1j * 2.0 * np.pi * fb * t)
        spec = (np.abs(np.fft.fft(s_if * np.hanning(n_fft))[: n_fft // 2 + 1]) ** 2)
        max_spec = np.max(spec)
        if max_spec > 0:
            spec = (spec / max_spec) * att_factor

        freqs_fft = np.fft.rfftfreq(n_fft, d=1.0 / fs_if)
        heights_fft = (self.c * freqs_fft) / (2.0 * self.chirp_rate * 1e3)
        heights_target = np.linspace(60.0, max_height_km, n_height_bins)
        return np.interp(heights_target, heights_fft, spec, left=0.0, right=0.0)

    @staticmethod
    def compute_group_refractive_index_O(f_hz: float, fp_profile: np.ndarray) -> np.ndarray:
        X = (fp_profile / f_hz) ** 2
        return 1.0 / np.sqrt(np.maximum(1.0 - X, 0.002))

    @staticmethod
    def compute_group_refractive_index_X(
        f_hz: float, fp_profile: np.ndarray, fH_profile: np.ndarray
    ) -> np.ndarray:
        X = (fp_profile / f_hz) ** 2
        Y = fH_profile / f_hz
        one_minus_Y = np.maximum(1.0 - Y, 0.05)
        rad = np.maximum(1.0 - (X / one_minus_Y), 0.002)
        return (1.0 / np.sqrt(rad)) * (1.0 + 0.5 * (X * Y) / (one_minus_Y**2))

    def _deposit_echo(
        self,
        ionogram_power: np.ndarray,
        f_idx: int,
        h_prime: float,
        att: float,
        scale: float,
        heights_km: np.ndarray,
    ) -> None:
        n_bins = len(heights_km)
        if n_bins < 8:
            if heights_km[0] <= h_prime <= heights_km[-1]:
                b = int(np.interp(h_prime, heights_km, np.arange(n_bins)))
                if 0 <= b < n_bins:
                    ionogram_power[b, f_idx] += scale * att
                    if b + 1 < n_bins:
                        ionogram_power[b + 1, f_idx] += 0.5 * scale * att
                    if b - 1 >= 0:
                        ionogram_power[b - 1, f_idx] += 0.5 * scale * att
        else:
            p = self.simulate_dechirped_range_spectrum(h_prime, att, n_bins, float(heights_km[-1]))
            ionogram_power[:, f_idx] += scale * p

    def synthesize_ionogram_profile(
        self,
        n_freq_steps: int = 140,
        n_height_bins: int = 150,
        max_height_km: float = 650.0,
    ) -> Dict:
        freqs_mhz = np.linspace(self.f_start_mhz, self.f_stop_mhz, n_freq_steps)
        heights_km = np.linspace(60.0, max_height_km, n_height_bins)
        ionogram_power = np.zeros((n_height_bins, n_freq_steps), dtype=np.float64)

        h_km, fp_hz, fH_hz = self.plasma.h_km, self.plasma.fp_hz, self.plasma.f_H
        h_prime_O_arr = np.full(n_freq_steps, np.nan, dtype=np.float64)
        h_prime_X_arr = np.full(n_freq_steps, np.nan, dtype=np.float64)
        true_refl_h_arr = np.full(n_freq_steps, np.nan, dtype=np.float64)

        foE = float(np.max(fp_hz[(h_km >= 90.0) & (h_km <= 140.0)])) / 1e6
        foF1 = float(np.max(fp_hz[(h_km > 140.0) & (h_km <= 220.0)])) / 1e6
        mask_F2 = h_km > 220.0
        foF2 = float(np.max(fp_hz[mask_F2])) / 1e6
        fxF2 = foF2 + (float(np.mean(fH_hz[mask_F2])) / 2e6)

        for f_idx, f_mhz in enumerate(freqs_mhz):
            f_hz = f_mhz * 1e6

            refl_O = np.where(fp_hz >= f_hz)[0]
            if len(refl_O) > 0:
                irefl = int(refl_O[0])
                true_refl_h_arr[f_idx] = float(h_km[irefl])
                if irefl >= 2:
                    mu_p_O = self.compute_group_refractive_index_O(f_hz, fp_hz[:irefl])
                    h_prime_O = 60.0 + integrate_trapz(mu_p_O, h_km[:irefl])
                    h_prime_O_arr[f_idx] = h_prime_O
                    att_O = np.exp(-((1.8 / f_mhz) ** 1.8))
                    self._deposit_echo(ionogram_power, f_idx, h_prime_O, att_O, 1.5, heights_km)

            if f_hz > float(np.max(fH_hz)):
                refl_X = np.where((fp_hz**2) >= (f_hz**2 - f_hz * fH_hz))[0]
                if len(refl_X) > 0:
                    irefl_x = int(refl_X[0])
                    if irefl_x >= 2:
                        mu_p_X = self.compute_group_refractive_index_X(f_hz, fp_hz[:irefl_x], fH_hz[:irefl_x])
                        h_prime_X = 60.0 + integrate_trapz(mu_p_X, h_km[:irefl_x])
                        h_prime_X_arr[f_idx] = h_prime_X
                        att_X = 0.85 * np.exp(-((2.2 / f_mhz) ** 1.8))
                        self._deposit_echo(ionogram_power, f_idx, h_prime_X, att_X, 1.2, heights_km)

        time_awgn = (
            np.random.standard_normal((n_height_bins, n_freq_steps))
            + 1j * np.random.standard_normal((n_height_bins, n_freq_steps))
        ) * (0.04 / np.sqrt(2.0))
        ionogram_power += (np.abs(np.fft.fft(time_awgn, axis=0)) ** 2) / n_height_bins

        max_p = np.max(ionogram_power)
        ionogram_db = 10.0 * np.log10(ionogram_power / (max_p if max_p > 0 else 1.0) + 1e-4)

        return {
            "freqs_mhz": freqs_mhz,
            "heights_km": heights_km,
            "ionogram_db": ionogram_db,
            "h_prime_O_km": h_prime_O_arr,
            "h_prime_X_km": h_prime_X_arr,
            "true_refl_h_km": true_refl_h_arr,
            "critical_frequencies": {
                "foE_mhz": foE,
                "foF1_mhz": foF1,
                "foF2_mhz": foF2,
                "fxF2_mhz": fxF2,
            },
        }
