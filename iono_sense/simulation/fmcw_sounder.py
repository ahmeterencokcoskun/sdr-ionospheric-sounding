#!/usr/bin/env python3
"""
Modül: FMCW (LFM Chirp) İyonosond Radarı Simülatörü ve İyonogram Çıkarıcı.
Doğrusal frekans modülasyonlu (Chirp) sondaj dalgası üretir, katman yansımalarını
(E, Es, F1, F2 sanal irtifa h') Matched Filter / De-chirp FFT ile çözer.
"""

from typing import Dict, Tuple
import numpy as np


class FMCWIonosondeSimulator:
    """
    Klasik ve modern SDR tabanlı FMCW iyonosond (chirp sounder) radar simülatörü.
    f_start - f_stop aralığında taranan chirp sinyali gönderir ve
    sanal yükseklik h'(f) [İyonogram] matrisini üretir.
    """

    def __init__(
        self,
        f_start_mhz: float = 1.0,
        f_stop_mhz: float = 16.0,
        chirp_rate_khz_per_sec: float = 100.0,
        c_mps: float = 2.99792458e8,
    ):
        self.f_start_mhz = f_start_mhz
        self.f_stop_mhz = f_stop_mhz
        self.chirp_rate = chirp_rate_khz_per_sec * 1e3  # Hz/s
        self.c = c_mps

    def synthesize_ionogram_profile(
        self,
        n_freq_steps: int = 150,
        n_height_bins: int = 120,
        max_height_km: float = 600.0,
    ) -> Dict:
        """
        Gündüz Kayseri iyonosfer katmanları (E, F1, F2) için
        sanal irtifa h'(f) ve yankı genliklerini sentetik olarak üretir.
        Grup gecikmesi: tau(f) = 2 * h'(f) / c
        """
        freqs_mhz = np.linspace(self.f_start_mhz, self.f_stop_mhz, n_freq_steps)
        heights_km = np.linspace(60.0, max_height_km, n_height_bins)
        ionogram_power = np.zeros((n_height_bins, n_freq_steps), dtype=np.float64)

        # Katman Kritik Frekansları (foE ~= 3.4 MHz, foF1 ~= 5.2 MHz, foF2 ~= 10.8 MHz)
        foE = 3.4
        foF1 = 5.2
        foF2 = 10.8

        for f_idx, f in enumerate(freqs_mhz):
            # 1. E Katmanı (95 - 125 km)
            if f <= foE:
                # Breit-Tuve / Quasi-parabolic sanal yükseklik artışı
                h_prime_E = 100.0 + 15.0 / np.sqrt(max(1.0 - (f / foE) ** 2, 0.05))
                bin_idx = int(np.interp(h_prime_E, heights_km, np.arange(n_height_bins)))
                if 0 <= bin_idx < n_height_bins:
                    ionogram_power[bin_idx, f_idx] += 1.0
                    if bin_idx + 1 < n_height_bins:
                        ionogram_power[bin_idx + 1, f_idx] += 0.5
                    if bin_idx - 1 >= 0:
                        ionogram_power[bin_idx - 1, f_idx] += 0.5

            # 2. F1 Katmanı (160 - 240 km)
            if foE * 0.85 <= f <= foF1:
                h_prime_F1 = 180.0 + 35.0 / np.sqrt(max(1.0 - (f / foF1) ** 2, 0.06))
                bin_idx = int(np.interp(h_prime_F1, heights_km, np.arange(n_height_bins)))
                if 0 <= bin_idx < n_height_bins:
                    ionogram_power[bin_idx, f_idx] += 0.85
                    if bin_idx + 1 < n_height_bins:
                        ionogram_power[bin_idx + 1, f_idx] += 0.4

            # 3. F2 Katmanı (250 - 450 km) - O-Modu ve X-Modu (f_x = f_o + f_H / 2)
            # O-Modu
            if foF1 * 0.85 <= f <= foF2:
                h_prime_F2_O = 260.0 + 65.0 / np.sqrt(max(1.0 - (f / foF2) ** 2, 0.04))
                bin_idx_O = int(np.interp(h_prime_F2_O, heights_km, np.arange(n_height_bins)))
                if 0 <= bin_idx_O < n_height_bins:
                    ionogram_power[bin_idx_O, f_idx] += 1.2
                    if bin_idx_O + 1 < n_height_bins:
                        ionogram_power[bin_idx_O + 1, f_idx] += 0.6

            # X-Modu (jiroskopik frekans f_H ~= 1.35 MHz ötelemeli)
            fxF2 = foF2 + 0.68
            if foF1 * 0.90 <= f <= fxF2:
                h_prime_F2_X = 265.0 + 68.0 / np.sqrt(max(1.0 - (f / fxF2) ** 2, 0.04))
                bin_idx_X = int(np.interp(h_prime_F2_X, heights_km, np.arange(n_height_bins)))
                if 0 <= bin_idx_X < n_height_bins:
                    ionogram_power[bin_idx_X, f_idx] += 0.9
                    if bin_idx_X + 1 < n_height_bins:
                        ionogram_power[bin_idx_X + 1, f_idx] += 0.45

        # Arka plan gürültüsü
        noise = 0.04 * np.random.rand(n_height_bins, n_freq_steps)
        ionogram_power += noise

        # Logaritmik dB güç ölçeği
        ionogram_db = 10.0 * np.log10(ionogram_power / np.max(ionogram_power) + 1e-4)

        return {
            "freqs_mhz": freqs_mhz,
            "heights_km": heights_km,
            "ionogram_db": ionogram_db,
            "critical_frequencies": {
                "foE_mhz": foE,
                "foF1_mhz": foF1,
                "foF2_mhz": foF2,
                "fxF2_mhz": foF2 + 0.68,
            },
        }
