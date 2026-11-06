#!/usr/bin/env python3
"""
Modül 2: ITU-R F.1487 Watterson Çok-Yollu Sönümleme Kanalı ve
Stokastik (Dinamik / Rastgele Sentetik) FT8 ve WSPR Taban Bant I/Q Sinyal Üreticisi.
Sabit seed veya statik istasyonlar yerine her çalıştırmada benzersiz gökyüzü koşulları üretir.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


COSTAS_ARRAY = np.array([3, 1, 4, 0, 6, 5, 2], dtype=np.int32)

WSPR_SYNC_VECTOR = np.array([
    1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
    0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
    0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
    0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
    0, 0
], dtype=np.int32)


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
    Rastgele sentetik (gerçek dışı ama fizik kurallarına uygun) FT8 ve WSPR istasyonları,
    Rayleigh/Rician çok-yollu kanalı ve Middleton Class-A gürültüsü üreten dinamik motor.
    """

    def __init__(self, fs: float = 12000.0, rng_seed: Optional[int] = None):
        self.fs = fs
        self.frame_sec = 15.0
        self.n_samples = int(self.fs * self.frame_sec)
        # Seed verilmezse işletim sistemi entropisi (None) kullanılarak her seferinde bambaşka veri üretilir
        self.rng = np.random.default_rng(rng_seed)

        # FT8 Parametreleri
        self.n_symbols = 79
        self.sym_dur = 0.16  # 6.25 Baud
        self.tone_spacing = 6.25  # Hz
        self.samples_per_sym = int(round(self.fs * self.sym_dur))  # 1920 örnek

    def _generate_random_callsign(self) -> str:
        """Rastgele ITU standartlarına uygun amatör telsiz çağrı işareti üretir."""
        prefixes = ["TA", "DL", "G", "I", "OH", "EA", "K", "JA", "VK", "PY", "ZL", "F", "SP"]
        pfx = str(self.rng.choice(prefixes))
        digit = str(self.rng.integers(0, 10))
        letters = "".join(chr(c) for c in self.rng.integers(65, 91, size=3))
        return f"{pfx}{digit}{letters}"

    def _generate_random_grid(self) -> str:
        """Rastgele 4 haneli Maidenhead grid lokatörü üretir (örn: KM78, JO52)."""
        field1 = chr(self.rng.integers(65, 83))  # A-R
        field2 = chr(self.rng.integers(65, 83))  # A-R
        square1 = str(self.rng.integers(0, 10))
        square2 = str(self.rng.integers(0, 10))
        return f"{field1}{field2}{square1}{square2}"

    def generate_random_stations(self, count: int = 5, min_freq: float = 400.0, max_freq: float = 2400.0) -> List[VirtualStation]:
        """Tamamen rastgele ve fiziksel olarak çakışmayan sentetik istasyonlar üretir."""
        stations = []
        # Frekans çakışmasını engellemek için frekans aralığını dilimle
        freq_slots = np.linspace(min_freq, max_freq, count + 1)

        for i in range(count):
            call = self._generate_random_callsign()
            grid = self._generate_random_grid()
            dist = float(self.rng.uniform(800.0, 4500.0))
            f_base = float(self.rng.uniform(freq_slots[i] + 30.0, freq_slots[i + 1] - 30.0))
            snr = float(self.rng.uniform(-4.0, 14.0))  # dB
            doppler_shift = float(self.rng.uniform(-0.9, 0.9))  # Hz
            doppler_spread = float(self.rng.uniform(0.08, 0.45))  # Hz
            delay_ms = float(self.rng.uniform(0.8, 3.2))  # ms
            tid_amp = float(self.rng.uniform(0.12, 0.45))  # Hz
            tid_freq = float(self.rng.uniform(0.05, 0.16))  # Hz

            stations.append(
                VirtualStation(
                    callsign=call,
                    grid=grid,
                    distance_km=dist,
                    base_freq_hz=f_base,
                    snr_db=snr,
                    doppler_shift_hz=doppler_shift,
                    doppler_spread_hz=doppler_spread,
                    multipath_delay_ms=delay_ms,
                    tid_amp_hz=tid_amp,
                    tid_freq_hz=tid_freq,
                )
            )
        return stations

    def generate_ldpc_codeword(self) -> Tuple[np.ndarray, np.ndarray]:
        """Her seferinde tamamen rastgele 174-bit LDPC kod kelimesi ve 79 sembol üretir."""
        msg_bits = self.rng.integers(0, 2, size=91, dtype=np.int32)
        H_gen = self.rng.integers(0, 2, size=(83, 91), dtype=np.int32)
        parity_bits = (H_gen @ msg_bits) % 2
        codeword_174 = np.concatenate([msg_bits, parity_bits])

        bit_triplets = codeword_174.reshape(58, 3)
        data_symbols = (
            bit_triplets[:, 0] * 4 + bit_triplets[:, 1] * 2 + bit_triplets[:, 2]
        )

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
        """Sürekli fazlı 8-GFSK karmaşık taban bant dalga şekli s(t)."""
        total_sig_samples = self.n_symbols * self.samples_per_sym
        freq_trace = np.repeat(symbols.astype(np.float64) * self.tone_spacing, self.samples_per_sym)

        # Gauss darbe şekillendirme (BT = 2.0)
        gauss_len = 65
        t_g = np.linspace(-2.0, 2.0, gauss_len)
        g_win = np.exp(-2.0 * (np.pi**2) * (t_g**2))
        g_win /= np.sum(g_win)
        freq_smooth = np.convolve(freq_trace, g_win, mode="same")

        phi_mod = 2.0 * np.pi * np.cumsum(freq_smooth) / self.fs
        t_vec = np.arange(total_sig_samples) / self.fs
        carrier_phase = 2.0 * np.pi * f_base_hz * t_vec
        s_clean = np.exp(1j * (carrier_phase + phi_mod)).astype(np.complex64)
        return s_clean, phi_mod

    def synthesize_wspr_waveform(
        self, symbols: np.ndarray, f_base_hz: float, drift_hz_per_min: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """WSPR 4-CPFSK dalga şekli üretir."""
        wspr_tone_spacing = 1.46484375
        wspr_samples_per_sym = int(round(self.fs * (8192.0 / 12000.0)))
        total_samples = len(symbols) * wspr_samples_per_sym

        freq_trace = np.repeat(symbols.astype(np.float64) * wspr_tone_spacing, wspr_samples_per_sym)
        t_vec = np.arange(total_samples) / self.fs

        drift_slope = (drift_hz_per_min / 60.0)
        drift_freq = drift_slope * t_vec

        phi_mod = 2.0 * np.pi * np.cumsum(freq_trace + drift_freq) / self.fs
        carrier_phase = 2.0 * np.pi * f_base_hz * t_vec
        s_clean = np.exp(1j * (carrier_phase + phi_mod)).astype(np.complex64)
        return s_clean, phi_mod

    def generate_wspr_symbols(self) -> Tuple[np.ndarray, np.ndarray]:
        """Rastgele 162-bit konvolüsyonel kod ve 4-FSK WSPR sembol dizisi üretir."""
        conv_bits = self.rng.integers(0, 2, size=162, dtype=np.int32)
        symbols = 2 * WSPR_SYNC_VECTOR + conv_bits
        return symbols, conv_bits

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
        self,
        band_mhz: float,
        tcxo_common_drift_hz: Optional[float] = None,
        station_count: int = 5,
    ) -> Dict:
        """Her çağrılışında rastgele ve dinamik sentetik istasyonlar ile 15s I/Q üretir."""
        rx_buffer = np.zeros(self.n_samples, dtype=np.complex64)

        # TCXO kayması belirtilmemişse gerçekçi bir termal osilatör kayması üret
        if tcxo_common_drift_hz is None:
            tcxo_common_drift_hz = float(self.rng.uniform(0.8, 1.9))

        stations = self.generate_random_stations(count=station_count)
        ground_truth = []
        start_sample = int(0.5 * self.fs)

        for idx, st in enumerate(stations):
            symbols, cw_174 = self.generate_ldpc_codeword()
            s_clean, phi_mod = self.synthesize_ft8_waveform(symbols, st.base_freq_hz)
            sig_len = len(s_clean)
            t_sig = np.arange(sig_len) / self.fs

            # Anlık TID mikro-Doppler yörüngesi
            inst_doppler_true = (
                st.doppler_shift_hz
                + st.tid_amp_hz * np.sin(2.0 * np.pi * st.tid_freq_hz * t_sig + self.rng.uniform(0, 2 * np.pi))
            )
            phi_ionosphere = 2.0 * np.pi * np.cumsum(inst_doppler_true) / self.fs
            phi_tcxo = 2.0 * np.pi * tcxo_common_drift_hz * t_sig

            # 1F2 Ana Yol
            g1 = 0.85 + 0.15 * self._rayleigh_doppler_tap(sig_len, st.doppler_spread_hz)
            path_1f2 = g1 * s_clean * np.exp(1j * (phi_ionosphere + phi_tcxo))

            # 2F2 İkincil Çok-Yollu Bileşen
            delay_samples = int(round((st.multipath_delay_ms * 1e-3) * self.fs))
            g2 = 0.32 * self._rayleigh_doppler_tap(sig_len, st.doppler_spread_hz * 1.4)
            path_2f2 = np.zeros_like(path_1f2)
            if delay_samples < sig_len:
                path_2f2[delay_samples:] = (
                    g2[:-delay_samples]
                    * s_clean[:-delay_samples]
                    * np.exp(1j * (phi_ionosphere[:-delay_samples] * 1.35 + phi_tcxo[:-delay_samples]))
                )

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

        # Middleton Class-A gürültü modeli
        noise_std = 0.35
        awgn = noise_std * (
            self.rng.standard_normal(self.n_samples)
            + 1j * self.rng.standard_normal(self.n_samples)
        ) / np.sqrt(2.0)

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

    def simulate_wspr_slot(
        self,
        band_mhz: float = 14.0956,
        tcxo_drift_slope_hz_min: Optional[float] = None,
        station_count: int = 4,
    ) -> Dict:
        """Tamamen rastgele çağrı işaretleri, mesafeler ve iyonosferik sürüklenme eğimleri üretir."""
        wspr_samples_per_sym = int(round(self.fs * (8192.0 / 12000.0)))
        total_samples = 162 * wspr_samples_per_sym
        rx_buffer = np.zeros(total_samples, dtype=np.complex64)

        if tcxo_drift_slope_hz_min is None:
            tcxo_drift_slope_hz_min = float(self.rng.uniform(0.2, 0.7))

        wspr_stations = self.generate_random_stations(count=station_count, min_freq=1400.0, max_freq=1700.0)
        # Her istasyon için -1.5 ile +1.5 Hz/dk arasında rastgele iyonosferik drift
        drifts_hz_min = [float(self.rng.uniform(-1.2, 1.2)) for _ in range(station_count)]

        ground_truth = []
        t_full = np.arange(total_samples) / self.fs

        for idx, (st, drift_val) in enumerate(zip(wspr_stations, drifts_hz_min)):
            symbols, code_bits = self.generate_wspr_symbols()
            total_drift = drift_val + tcxo_drift_slope_hz_min
            s_clean, phi_mod = self.synthesize_wspr_waveform(symbols, st.base_freq_hz, drift_hz_per_min=total_drift)

            g_t = 0.90 + 0.10 * self._rayleigh_doppler_tap(total_samples, st.doppler_spread_hz)
            phi_doppler = 2.0 * np.pi * st.doppler_shift_hz * t_full
            sig_rx = g_t * s_clean * np.exp(1j * phi_doppler)

            amp = 10.0 ** (st.snr_db / 20.0)
            rx_buffer += (amp * sig_rx).astype(np.complex64)

            ground_truth.append({
                "station": st,
                "symbols": symbols,
                "code_bits": code_bits,
                "drift_hz_per_min": drift_val,
                "total_drift_hz_min": total_drift,
                "base_freq_hz": st.base_freq_hz,
            })

        noise_std = 0.40
        awgn = noise_std * (
            self.rng.standard_normal(total_samples)
            + 1j * self.rng.standard_normal(total_samples)
        ) / np.sqrt(2.0)
        rx_buffer += awgn.astype(np.complex64)

        return {
            "band_mhz": band_mhz,
            "fs": self.fs,
            "total_duration_sec": total_samples / self.fs,
            "rx_buffer": rx_buffer,
            "ground_truth": ground_truth,
            "tcxo_drift_slope_hz_min": tcxo_drift_slope_hz_min,
        }
