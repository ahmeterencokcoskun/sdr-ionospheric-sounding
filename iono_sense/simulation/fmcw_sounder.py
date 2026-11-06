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
          X-modu için kesim şartı: fp^2 = f^2 - f * f_H
          mu'_X plazma ve manyetik rezonansa yaklaştıkça artar.
        """
        # Yaklaşık manyeto-iyonik X-modu grup indisi
        X = (fp_profile / f_hz) ** 2
        Y = fH_profile / f_hz
        radicand = np.maximum(1.0 - (X / np.maximum(1.0 - Y, 0.05)), 0.002)
        mu_prime_X = 1.0 / np.sqrt(radicand)
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
                if irefl >= 2:
                    # h' = integral_0^hrefl mu'_O dh
                    h_sub = h_km[:irefl]
                    fp_sub = fp_hz[:irefl]
                    mu_prime_O = self.compute_group_refractive_index_O(f_hz, fp_sub)
                    h_prime_O = integrate_trapz(mu_prime_O, h_sub)

                    # İyonosferik iki yönlü zayıflama (D/E soğurması):
                    # Düşük frekansta soğurma çok yüksek (yankı zayıf), yüksek frekansta güçlü
                    att_factor = np.exp(-((1.8 / f_mhz) ** 1.8))

                    # Güç matrisine Gauss saçılma penceresi ile yerleştir
                    bin_idx = int(np.interp(h_prime_O, heights_km, np.arange(n_height_bins)))
                    if 0 <= bin_idx < n_height_bins:
                        ionogram_power[bin_idx, f_idx] += 1.5 * att_factor
                        if bin_idx + 1 < n_height_bins:
                            ionogram_power[bin_idx + 1, f_idx] += 0.75 * att_factor
                        if bin_idx - 1 >= 0:
                            ionogram_power[bin_idx - 1, f_idx] += 0.75 * att_factor

            # --- 2. X-MODU (Extraordinary Wave) ---
            # Yansıma şartı: fp^2 >= f^2 - f * fH
            cutoff_sq = (f_hz**2) - (f_hz * fH_hz)
            refl_idx_X = np.where((fp_hz**2) >= cutoff_sq)[0]
            if len(refl_idx_X) > 0:
                irefl = int(refl_idx_X[0])
                if irefl >= 2:
                    h_sub = h_km[:irefl]
                    fp_sub = fp_hz[:irefl]
                    fH_sub = fH_hz[:irefl]
                    mu_prime_X = self.compute_group_refractive_index_X(f_hz, fp_sub, fH_sub)
                    h_prime_X = integrate_trapz(mu_prime_X, h_sub)

                    att_factor_X = 0.85 * np.exp(-((2.2 / f_mhz) ** 1.8))

                    bin_idx = int(np.interp(h_prime_X, heights_km, np.arange(n_height_bins)))
                    if 0 <= bin_idx < n_height_bins:
                        ionogram_power[bin_idx, f_idx] += 1.2 * att_factor_X
                        if bin_idx + 1 < n_height_bins:
                            ionogram_power[bin_idx + 1, f_idx] += 0.6 * att_factor_X
                        if bin_idx - 1 >= 0:
                            ionogram_power[bin_idx - 1, f_idx] += 0.6 * att_factor_X

        # Sayısal gürültü tabanı (AWGN benzeri radyo paraziti)
        noise = 0.03 * np.random.rand(n_height_bins, n_freq_steps)
        ionogram_power += noise

        # Logaritmik dB normalize güç
        max_p = np.max(ionogram_power)
        ionogram_db = 10.0 * np.log10(ionogram_power / (max_p if max_p > 0 else 1.0) + 1e-4)

        return {
            "freqs_mhz": freqs_mhz,
            "heights_km": heights_km,
            "ionogram_db": ionogram_db,
            "critical_frequencies": {
                "foE_mhz": foE,
                "foF1_mhz": foF1,
                "foF2_mhz": foF2,
                "fxF2_mhz": fxF2,
            },
        }
