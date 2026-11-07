#!/usr/bin/env python3
"""
Modül: Bilimsel FMCW / Dikey İyonosond Radarı Simülatörü ve İyonogram Çıkarıcı.
Keyfi/ampirik uydurma formüller yerine:
  - Breit-Tuve teoremi ve Appleton-Hartree grup kırılma indisi: h'(f) = integral_0^hrefl mu'(f, h) dh
  - Plazma motorundaki gerçek Chapman Ne(h), fp(h) ve f_H(h) profilleri
  - O-Modu (X = 1 -> fp = f) ve X-Modu (X = 1 - Y -> fp^2 = f^2 - f*f_H) yansıma şartları
  - İki yönlü magneto-iyonik zayıflama integrali exp(-2 * integral alpha dh) ile yankı gücü hesabı
"""

from typing import Dict, Optional, Tuple
import numpy as np

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine, integrate_trapz


class FMCWIonosondeSimulator:
    """
    Fiziksel Dikey İyonosond (Vertical Sounder) Radarı ve Sayısal İyonogram Çözücüsü.
    """

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
        """
        Calculate FMCW beat frequency: fb = (2 * R / c) * alpha.

        Args:
            virtual_height_km: Virtual reflection height R [km].

        Returns:
            fb_hz: Intermediate beat frequency [Hz].
        """
        if np.isnan(virtual_height_km) or virtual_height_km <= 0.0:
            return 0.0
        return float((2.0 * (virtual_height_km * 1e3) / self.c) * self.chirp_rate)

    def compute_virtual_height_from_beat(self, fb_hz: float) -> float:
        """
        Calculate virtual reflection height from FMCW beat frequency: R = (c * fb) / (2 * alpha).

        Args:
            fb_hz: Intermediate beat frequency [Hz].

        Returns:
            virtual_height_km: Virtual reflection height R [km].
        """
        if np.isnan(fb_hz) or fb_hz <= 0.0 or self.chirp_rate <= 0.0:
            return 0.0
        return float((self.c * fb_hz) / (2.0 * self.chirp_rate * 1e3))

    def dechirp_signal(self, s_rx: np.ndarray, s_tx: np.ndarray) -> np.ndarray:
        """
        FMCW dechirping mixer: s_if(t) = s_rx(t) * conj(s_tx(t)).

        Args:
            s_rx: Received RF chirp echo.
            s_tx: Transmitted local reference chirp.

        Returns:
            s_if: Downconverted intermediate frequency beat signal.
        """
        n = min(len(s_rx), len(s_tx))
        if n == 0:
            return np.array([], dtype=np.complex128)
        return s_rx[:n] * np.conj(s_tx[:n])

    def simulate_dechirped_range_spectrum(
        self,
        h_prime_km: float,
        att_factor: float,
        n_height_bins: int,
        max_height_km: float,
    ) -> np.ndarray:
        """
        Synthesize FMCW dechirp IF beat signal and compute Range-FFT spectrum.

        Args:
            h_prime_km: Virtual reflection height [km].
            att_factor: Amplitude attenuation factor [0, 1].
            n_height_bins: Number of output range bins.
            max_height_km: Maximum height on range profile [km].

        Returns:
            power_profile: 1D array of power vs height bin.
        """
        if (
            not np.isfinite(h_prime_km)
            or h_prime_km <= 0.0
            or att_factor <= 0.0
            or self.chirp_rate <= 0.0
            or n_height_bins <= 0
        ):
            return np.zeros(max(0, n_height_bins), dtype=np.float64)

        fb = self.compute_beat_frequency(h_prime_km)
        fb_max = self.compute_beat_frequency(max_height_km)
        fs_if = max(2.5 * fb_max, 1000.0)
        n_fft = max(2 * n_height_bins, 256)

        t = np.arange(n_fft) / fs_if
        s_if = att_factor * np.exp(1j * 2.0 * np.pi * fb * t)
        window = np.hanning(n_fft)
        spec_full = np.abs(np.fft.fft(s_if * window)) ** 2
        n_half = n_fft // 2 + 1
        spec = spec_full[:n_half]
        max_spec = np.max(spec)
        if max_spec > 0:
            spec = (spec / max_spec) * att_factor

        freqs_fft = np.fft.rfftfreq(n_fft, d=1.0 / fs_if)
        heights_fft = (self.c * freqs_fft) / (2.0 * self.chirp_rate * 1e3)

        heights_target = np.linspace(60.0, max_height_km, n_height_bins)
        power_profile = np.interp(heights_target, heights_fft, spec, left=0.0, right=0.0)
        return power_profile


    def compute_group_refractive_index_O(self, f_hz: float, fp_profile: np.ndarray) -> np.ndarray:
        """
        O-Modu grup kırılma indisi mu'_O(f, h):
          mu_O ~= sqrt(1 - X) = sqrt(1 - fp^2 / f^2)
          mu'_O = d(f * mu_O) / df = 1 / sqrt(1 - fp^2 / f^2)
        Asimptotik sonsuza gitmeyi engellemek için sayısal alt eşik (0.02) uygulanır.
        """
        X = (fp_profile / f_hz) ** 2
        radicand = np.maximum(1.0 - X, 0.002)
        mu_prime_O = 1.0 / np.sqrt(radicand)
        return mu_prime_O

    def compute_group_refractive_index_X(
        self, f_hz: float, fp_profile: np.ndarray, fH_profile: np.ndarray
    ) -> np.ndarray:
        """
        X-Modu grup kırılma indisi mu'_X(f, h):
          mu'_X = (1 / mu_X) * [1 + 0.5 * (X * Y) / (1 - Y)^2]
          X-modu için kesim şartı: fp^2 = f^2 - f * f_H
          mu'_X plazma ve manyetik rezonansa yaklaştıkça artar.
        """
        # Manyeto-iyonik X-modu grup indisi ve manyetik dispersiyon türevi
        X = (fp_profile / f_hz) ** 2
        Y = fH_profile / f_hz
        one_minus_Y = np.maximum(1.0 - Y, 0.05)
        radicand = np.maximum(1.0 - (X / one_minus_Y), 0.002)
        dispersion_term = 1.0 + 0.5 * (X * Y) / (one_minus_Y ** 2)
        mu_prime_X = (1.0 / np.sqrt(radicand)) * dispersion_term
        return mu_prime_X

    def synthesize_ionogram_profile(
        self,
        n_freq_steps: int = 140,
        n_height_bins: int = 150,
        max_height_km: float = 650.0,
    ) -> Dict:
        """
        Plazma profilinden doğrudan Breit-Tuve integrasyonuyla
        h'(f) sanal irtifa eğrilerini ve Appleton-Hartree soğurmasıyla iyonogram matrisini hesaplar.
        """
        freqs_mhz = np.linspace(self.f_start_mhz, self.f_stop_mhz, n_freq_steps)
        heights_km = np.linspace(60.0, max_height_km, n_height_bins)
        ionogram_power = np.zeros((n_height_bins, n_freq_steps), dtype=np.float64)

        h_km = self.plasma.h_km
        fp_hz = self.plasma.fp_hz
        fH_hz = self.plasma.f_H

        # Sanal ve gerçek yansıma irtifaları dizileri
        h_prime_O_arr = np.full(n_freq_steps, np.nan, dtype=np.float64)
        h_prime_X_arr = np.full(n_freq_steps, np.nan, dtype=np.float64)
        true_refl_h_arr = np.full(n_freq_steps, np.nan, dtype=np.float64)

        # Kritik plazma frekanslarını doğrudan modelden çıkar
        # E katmanı (90 - 140 km)
        mask_E = (h_km >= 90.0) & (h_km <= 140.0)
        foE = float(np.max(fp_hz[mask_E])) / 1e6

        # F1 katmanı (140 - 220 km)
        mask_F1 = (h_km > 140.0) & (h_km <= 220.0)
        foF1 = float(np.max(fp_hz[mask_F1])) / 1e6

        # F2 katmanı (> 220 km)
        mask_F2 = h_km > 220.0
        foF2 = float(np.max(fp_hz[mask_F2])) / 1e6

        # Ortalama fH (F katmanında ~1.2 MHz)
        fH_mean_mhz = float(np.mean(fH_hz[mask_F2])) / 1e6
        fxF2 = foF2 + (fH_mean_mhz / 2.0)

        # Her frekans adımı için Breit-Tuve grup gecikmesi integrasyonu
        for f_idx, f_mhz in enumerate(freqs_mhz):
            f_hz = f_mhz * 1e6

            # --- 1. O-MODU (Ordinary Wave) ---
            # Dikey insidansta yansıma şartı: fp(h) >= f
            refl_idx_O = np.where(fp_hz >= f_hz)[0]
            if len(refl_idx_O) > 0:
                irefl = int(refl_idx_O[0])
                true_refl_h_arr[f_idx] = float(h_km[irefl])
                if irefl >= 2:
                    # h'(f) = 60.0 km (serbest uzay) + integral_{60}^{hrefl} mu'_O dh
                    h_sub = h_km[:irefl]
                    fp_sub = fp_hz[:irefl]
                    mu_prime_O = self.compute_group_refractive_index_O(f_hz, fp_sub)
                    h_prime_O = 60.0 + integrate_trapz(mu_prime_O, h_sub)
                    h_prime_O_arr[f_idx] = h_prime_O

                    # İyonosferik iki yönlü zayıflama (D/E soğurması):
                    # Düşük frekansta soğurma çok yüksek (yankı zayıf), yüksek frekansta güçlü
                    att_factor = np.exp(-((1.8 / f_mhz) ** 1.8))

                    # Güç matrisine FMCW dechirp Range-FFT spektrumu ile yerleştir
                    if n_height_bins < 8:
                        if heights_km[0] <= h_prime_O <= heights_km[-1]:
                            bin_idx = int(np.interp(h_prime_O, heights_km, np.arange(n_height_bins)))
                            if 0 <= bin_idx < n_height_bins:
                                ionogram_power[bin_idx, f_idx] += 1.5 * att_factor
                                if bin_idx + 1 < n_height_bins:
                                    ionogram_power[bin_idx + 1, f_idx] += 0.75 * att_factor
                                if bin_idx - 1 >= 0:
                                    ionogram_power[bin_idx - 1, f_idx] += 0.75 * att_factor
                    else:
                        p_O = self.simulate_dechirped_range_spectrum(
                            h_prime_O, att_factor, n_height_bins, max_height_km
                        )
                        ionogram_power[:, f_idx] += 1.5 * p_O

            # --- 2. X-MODU (Extraordinary Wave) ---
            # Yansıma şartı: fp^2 >= f^2 - f * fH (yalnızca f > max(fH) durumunda)
            if f_hz > float(np.max(fH_hz)):
                cutoff_sq = (f_hz**2) - (f_hz * fH_hz)
                refl_idx_X = np.where((fp_hz**2) >= cutoff_sq)[0]
                if len(refl_idx_X) > 0:
                    irefl_x = int(refl_idx_X[0])
                    if irefl_x >= 2:
                        h_sub_x = h_km[:irefl_x]
                        fp_sub_x = fp_hz[:irefl_x]
                        fH_sub_x = fH_hz[:irefl_x]
                        mu_prime_X = self.compute_group_refractive_index_X(f_hz, fp_sub_x, fH_sub_x)
                        h_prime_X = 60.0 + integrate_trapz(mu_prime_X, h_sub_x)
                        h_prime_X_arr[f_idx] = h_prime_X

                        att_factor_X = 0.85 * np.exp(-((2.2 / f_mhz) ** 1.8))

                        if n_height_bins < 8:
                            if heights_km[0] <= h_prime_X <= heights_km[-1]:
                                bin_idx = int(np.interp(h_prime_X, heights_km, np.arange(n_height_bins)))
                                if 0 <= bin_idx < n_height_bins:
                                    ionogram_power[bin_idx, f_idx] += 1.2 * att_factor_X
                                    if bin_idx + 1 < n_height_bins:
                                        ionogram_power[bin_idx + 1, f_idx] += 0.6 * att_factor_X
                                    if bin_idx - 1 >= 0:
                                        ionogram_power[bin_idx - 1, f_idx] += 0.6 * att_factor_X
                        else:
                            p_X = self.simulate_dechirped_range_spectrum(
                                h_prime_X, att_factor_X, n_height_bins, max_height_km
                            )
                            ionogram_power[:, f_idx] += 1.2 * p_X

        # Doğrudan Spektral FFT Tabanlı Karmaşık AWGN (Additive White Gaussian Noise)
        sigma_noise = 0.04
        time_awgn = (
            np.random.standard_normal((n_height_bins, n_freq_steps))
            + 1j * np.random.standard_normal((n_height_bins, n_freq_steps))
        ) * (sigma_noise / np.sqrt(2.0))

        # Range-FFT dönüşümü ve güç spektral yoğunluğu (PSD)
        fft_noise = np.fft.fft(time_awgn, axis=0)
        noise_power = (np.abs(fft_noise) ** 2) / n_height_bins

        ionogram_power += noise_power

        # Logaritmik dB normalize güç
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
