#!/usr/bin/env python3
"""
Modül 2: ITU-R F.1487 Watterson Çok-Yollu Sönümleme Kanalı ve
Sentetik FT8 (8-GFSK) / WSPR (4-FSK) Taban Bant I/Q Sinyal Üreticisi.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


COSTAS_ARRAY = np.array([3, 1, 4, 0, 6, 5, 2], dtype=np.int32)


@dataclass
class VirtualStation:
    callsign: str
    grid: str
    distance_km: float
    base_freq_hz: float
    snr_db: float
    doppler_shift_hz: float
    doppler_spread_hz: float
    multipath_delay_ms: float
    tid_amp_hz: float
    tid_freq_hz: float


class WattersonChannelSimulator:
    """
    15 saniyelik (fs = 12000 Hz) pencerede çoklu FT8 istasyonu,
    1F2 + 2F2 çok-yollu Rayleigh sönümlemesi, TID mikro-Doppler yörüngesi,
    RSPdx TCXO ortak-mod kayması ve Middleton Class-A darbeli gürültü üretir.
    """

    def __init__(self, fs: float = 12000.0, rng_seed: int = 23112006):
        self.fs = fs
        self.frame_sec = 15.0
        self.n_samples = int(self.fs * self.frame_sec)
        self.rng = np.random.default_rng(rng_seed)

        # FT8 Fiziksel Katman Sabitleri
        self.n_symbols = 79
        self.sym_dur = 0.16  # 6.25 baud
        self.tone_spacing = 6.25  # Hz
        self.samples_per_sym = int(round(self.fs * self.sym_dur))  # 1920 örnek

    def generate_ldpc_codeword(self, station_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministik 77-bit bilgi + 14-bit CRC -> 174-bit LDPC kod kelimesi
        ve 79 sembollük tam FT8 (Costas + 58 veri sembolü) dizisini üretir.
        """
        local_rng = np.random.default_rng(1000 + station_idx)
        msg_bits = local_rng.integers(0, 2, size=91, dtype=np.int32)
        # Deterministik parite matrisi (LDPC 174,91 temsili üreteç)
        H_gen = local_rng.integers(0, 2, size=(83, 91), dtype=np.int32)
        parity_bits = (H_gen @ msg_bits) % 2
        codeword_174 = np.concatenate([msg_bits, parity_bits])

        # 174 bit -> 58 adet 3-bitlik Gray kodlu 8-GFSK sembolü (0..7)
        bit_triplets = codeword_174.reshape(58, 3)
        data_symbols = (
            bit_triplets[:, 0] * 4 + bit_triplets[:, 1] * 2 + bit_triplets[:, 2]
        )

        # 79 Sembollük FT8 çerçevesi: [Costas(7) | Data(29) | Costas(7) | Data(29) | Costas(7)]
        frame_symbols = np.zeros(self.n_symbols, dtype=np.int32)
        frame_symbols[0:7] = COSTAS_ARRAY
        frame_symbols[7:36] = data_symbols[0:29]
        frame_symbols[36:43] = COSTAS_ARRAY
        frame_symbols[43:72] = data_symbols[29:58]
        frame_symbols[72:79] = COSTAS_ARRAY

        return frame_symbols, codeword_174

    def synthesize_ft8_waveform(
        self, symbols: np.ndarray, f_base_hz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sürekli fazlı 8-GFSK karmaşık taban bant dalga şekli s(t) ve
        saf modülasyon fazı phi_mod(t) üretir.
        """
        total_sig_samples = self.n_symbols * self.samples_per_sym
        freq_trace = np.repeat(symbols.astype(np.float64) * self.tone_spacing, self.samples_per_sym)

        # Gauss darbe şekillendirme (BT = 2.0)
        gauss_len = 65
        t_g = np.linspace(-2.0, 2.0, gauss_len)
        g_win = np.exp(-2.0 * (np.pi**2) * (t_g**2))
        g_win /= np.sum(g_win)
        freq_smooth = np.convolve(freq_trace, g_win, mode="same")

        # Anlık modülasyon fazı
        phi_mod = 2.0 * np.pi * np.cumsum(freq_smooth) / self.fs
        t_vec = np.arange(total_sig_samples) / self.fs
        carrier_phase = 2.0 * np.pi * f_base_hz * t_vec
        s_clean = np.exp(1j * (carrier_phase + phi_mod)).astype(np.complex64)
        return s_clean, phi_mod

    def _rayleigh_doppler_tap(self, n_len: int, doppler_spread_hz: float) -> np.ndarray:
        """Gauss Doppler güç spektrumuna sahip zamanla değişen Rayleigh sönümleme katsayısı G_m(t)."""
        w = (
            self.rng.standard_normal(n_len) + 1j * self.rng.standard_normal(n_len)
        ) / np.sqrt(2.0)
        freqs = np.fft.fftfreq(n_len, d=1.0 / self.fs)
        sigma_f = max(doppler_spread_hz, 0.03)
        H_gauss = np.exp(-((freqs) ** 2) / (2.0 * (sigma_f**2)))
        W_f = np.fft.fft(w) * np.sqrt(H_gauss)
        g_t = np.fft.ifft(W_f)
        g_t /= np.sqrt(np.mean(np.abs(g_t) ** 2) + 1e-12)
        return g_t

    def simulate_band_frame(
        self, band_mhz: float, tcxo_common_drift_hz: float = 1.35
    ) -> Dict:
        """
        Belirli bir HF bandı için çoklu istasyon içeren 15 saniyelik
        alıcı I/Q tamponunu ve yer-doğrusu (ground-truth) verilerini üretir.
        """
        rx_buffer = np.zeros(self.n_samples, dtype=np.complex64)

        stations = [
            VirtualStation("DL1ABC", "JO52", 2180.0, 720.0, +8.5, +0.42, 0.14, 1.25, 0.22, 0.08),
            VirtualStation("G4XYZ", "IO91", 2840.0, 1140.0, +4.0, -0.31, 0.25, 1.85, 0.35, 0.11),
            VirtualStation("I2QRS", "JN45", 1920.0, 1580.0, +11.2, +0.18, 0.09, 0.90, 0.15, 0.06),
            VirtualStation("OH2MNO", "KP20", 2410.0, 1980.0, -2.5, -0.65, 0.48, 2.20, 0.40, 0.14),
            VirtualStation("EA4DEF", "IN80", 3310.0, 2360.0, +1.5, +0.54, 0.31, 1.65, 0.28, 0.09),
        ]

        ground_truth = []
        start_sample = int(0.5 * self.fs)  # 0.5s başlangıç ofseti

        for idx, st in enumerate(stations):
            symbols, cw_174 = self.generate_ldpc_codeword(idx)
            s_clean, phi_mod = self.synthesize_ft8_waveform(symbols, st.base_freq_hz)
            sig_len = len(s_clean)
            t_sig = np.arange(sig_len) / self.fs

            # TID (Gezinen İyonosferik Düzensizlik) kaynaklı anlık mikro-Doppler profili
            inst_doppler_true = (
                st.doppler_shift_hz
                + st.tid_amp_hz * np.sin(2.0 * np.pi * st.tid_freq_hz * t_sig + 0.4 * idx)
            )
            phi_ionosphere = 2.0 * np.pi * np.cumsum(inst_doppler_true) / self.fs
            phi_tcxo = 2.0 * np.pi * tcxo_common_drift_hz * t_sig

            # 1F2 Ana Yol (Baskın bileşen + hafif Rician/Rayleigh dalgalanma)
            g1 = 0.85 + 0.15 * self._rayleigh_doppler_tap(sig_len, st.doppler_spread_hz)
            path_1f2 = g1 * s_clean * np.exp(1j * (phi_ionosphere + phi_tcxo))

            # 2F2 İkincil Çok-Yollu Bileşen (Gecikmeli ve farklı Doppler kaymalı)
            delay_samples = int(round((st.multipath_delay_ms * 1e-3) * self.fs))
            g2 = 0.32 * self._rayleigh_doppler_tap(sig_len, st.doppler_spread_hz * 1.4)
            path_2f2 = np.zeros_like(path_1f2)
            if delay_samples < sig_len:
                path_2f2[delay_samples:] = (
                    g2[:-delay_samples]
                    * s_clean[:-delay_samples]
                    * np.exp(1j * (phi_ionosphere[:-delay_samples] * 1.35 + phi_tcxo[:-delay_samples]))
                )

            # SNR Ölçekleme (2500 Hz referans gürültü bandına göre)
            amp = 10.0 ** (st.snr_db / 20.0)
            combined_sig = amp * (path_1f2 + path_2f2)

            rx_buffer[start_sample : start_sample + sig_len] += combined_sig.astype(np.complex64)

            ground_truth.append({
                "station": st,
                "symbols": symbols,
                "codeword_174": cw_174,
                "phi_mod": phi_mod,
                "t_sig": t_sig,
                "inst_doppler_true": inst_doppler_true,
                "start_sample": start_sample,
            })

        # Arka Plan AWGN + Middleton Class-A Darbeli (Impulsive) Gürültü
        noise_std = 0.35
        awgn = noise_std * (
            self.rng.standard_normal(self.n_samples)
            + 1j * self.rng.standard_normal(self.n_samples)
        ) / np.sqrt(2.0)

        # %1.5 olasılıkla yüksek genlikli atmosferik/korona darbe gürültüsü
        impulse_mask = self.rng.random(self.n_samples) < 0.015
        impulses = impulse_mask * (noise_std * 4.5) * (
            self.rng.standard_normal(self.n_samples)
            + 1j * self.rng.standard_normal(self.n_samples)
        )

        rx_buffer += (awgn + impulses).astype(np.complex64)

        return {
            "band_mhz": band_mhz,
            "fs": self.fs,
            "tcxo_true_hz": tcxo_common_drift_hz,
            "iq_buffer": rx_buffer,
            "ground_truth": ground_truth,
        }
