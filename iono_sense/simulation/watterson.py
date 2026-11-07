#!/usr/bin/env python3
"""ITU-R F.1487 Watterson Multipath Fading Channel and Synthetic FT8/WSPR Generator."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from iono_sense.dsp.ldpc_decoder import FT8LDPCDecoder


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
    0, 0,
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
    """Generates synthetic FT8 and WSPR stations, Rayleigh fading, and Middleton noise."""

    def __init__(self, fs: float = 12000.0, rng_seed: Optional[int] = None):
        self.fs = fs
        self.frame_sec = 15.0
        self.n_samples = int(self.fs * self.frame_sec)
        self.rng = np.random.default_rng(abs(int(rng_seed)) if rng_seed is not None else None)

        self.n_symbols = 79
        self.sym_dur = 0.16
        self.tone_spacing = 6.25
        self.samples_per_sym = int(round(self.fs * self.sym_dur))

        self.ldpc_decoder = FT8LDPCDecoder()
        self.H_info = self.ldpc_decoder.H[:, : self.ldpc_decoder.k_bits]

    def _generate_random_callsign(self) -> str:
        prefixes = ["TA", "DL", "G", "I", "OH", "EA", "K", "JA", "VK", "PY", "ZL", "F", "SP"]
        pfx = str(self.rng.choice(prefixes))
        digit = str(self.rng.integers(0, 10))
        letters = "".join(chr(c) for c in self.rng.integers(65, 91, size=3))
        return f"{pfx}{digit}{letters}"

    def _generate_random_grid(self) -> str:
        f1 = chr(self.rng.integers(65, 83))
        f2 = chr(self.rng.integers(65, 83))
        s1 = str(self.rng.integers(0, 10))
        s2 = str(self.rng.integers(0, 10))
        return f"{f1}{f2}{s1}{s2}"

    def generate_random_stations(
        self, count: int = 5, min_freq: float = 400.0, max_freq: float = 2400.0
    ) -> List[VirtualStation]:
        freq_slots = np.linspace(min_freq, max_freq, count + 1)
        return [
            VirtualStation(
                callsign=self._generate_random_callsign(),
                grid=self._generate_random_grid(),
                distance_km=float(self.rng.uniform(800.0, 4500.0)),
                base_freq_hz=float(self.rng.uniform(freq_slots[i] + 30.0, freq_slots[i + 1] - 30.0)),
                snr_db=float(self.rng.uniform(-4.0, 14.0)),
                doppler_shift_hz=float(self.rng.uniform(-0.9, 0.9)),
                doppler_spread_hz=float(self.rng.uniform(0.08, 0.45)),
                multipath_delay_ms=float(self.rng.uniform(0.8, 3.2)),
                tid_amp_hz=float(self.rng.uniform(0.12, 0.45)),
                tid_freq_hz=float(self.rng.uniform(0.05, 0.16)),
            )
            for i in range(count)
        ]

    def generate_ldpc_codeword(self) -> Tuple[np.ndarray, np.ndarray]:
        msg_bits = self.rng.integers(0, 2, size=91, dtype=np.int32)
        parity_bits = (self.H_info @ msg_bits) % 2
        codeword_174 = np.concatenate([msg_bits, parity_bits])

        triplets = codeword_174.reshape(58, 3)
        data_syms = triplets[:, 0] * 4 + triplets[:, 1] * 2 + triplets[:, 2]

        frame_syms = np.zeros(self.n_symbols, dtype=np.int32)
        frame_syms[0:7] = COSTAS_ARRAY
        frame_syms[7:36] = data_syms[0:29]
        frame_syms[36:43] = COSTAS_ARRAY
        frame_syms[43:72] = data_syms[29:58]
        frame_syms[72:79] = COSTAS_ARRAY
        return frame_syms, codeword_174

    def synthesize_ft8_waveform(
        self, symbols: np.ndarray, f_base_hz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        total_samples = self.n_symbols * self.samples_per_sym
        freq_trace = np.repeat(symbols.astype(np.float64) * self.tone_spacing, self.samples_per_sym)

        t_g = np.linspace(-2.0, 2.0, 65)
        g_win = np.exp(-2.0 * (np.pi**2) * (t_g**2))
        g_win /= np.sum(g_win)
        freq_smooth = np.convolve(freq_trace, g_win, mode="same")

        phi_mod = 2.0 * np.pi * np.cumsum(freq_smooth) / self.fs
        t_vec = np.arange(total_samples) / self.fs
        s_clean = np.exp(1j * (2.0 * np.pi * f_base_hz * t_vec + phi_mod)).astype(np.complex64)
        return s_clean, phi_mod

    def synthesize_wspr_waveform(
        self, symbols: np.ndarray, f_base_hz: float, drift_hz_per_min: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        wspr_samples_per_sym = int(round(self.fs * (8192.0 / 12000.0)))
        total_samples = len(symbols) * wspr_samples_per_sym
        freq_trace = np.repeat(symbols.astype(np.float64) * 1.46484375, wspr_samples_per_sym)
        t_vec = np.arange(total_samples) / self.fs

        phi_mod = 2.0 * np.pi * np.cumsum(freq_trace + (drift_hz_per_min / 60.0) * t_vec) / self.fs
        s_clean = np.exp(1j * (2.0 * np.pi * f_base_hz * t_vec + phi_mod)).astype(np.complex64)
        return s_clean, phi_mod

    def generate_wspr_symbols(self) -> Tuple[np.ndarray, np.ndarray]:
        conv_bits = self.rng.integers(0, 2, size=162, dtype=np.int32)
        return 2 * WSPR_SYNC_VECTOR + conv_bits, conv_bits

    def _rayleigh_doppler_tap(self, n_len: int, doppler_spread_hz: float) -> np.ndarray:
        w = (self.rng.standard_normal(n_len) + 1j * self.rng.standard_normal(n_len)) / np.sqrt(2.0)
        freqs = np.fft.fftfreq(n_len, d=1.0 / self.fs)
        sigma_f = max(doppler_spread_hz, 0.03)
        H_gauss = np.exp(-(freqs**2) / (2.0 * (sigma_f**2)))
        g_t = np.fft.ifft(np.fft.fft(w) * np.sqrt(H_gauss))
        return g_t / np.sqrt(np.mean(np.abs(g_t) ** 2) + 1e-12)

    def simulate_band_frame(
        self,
        band_mhz: float,
        tcxo_common_drift_hz: Optional[float] = None,
        station_count: int = 5,
    ) -> Dict:
        rx_buffer = np.zeros(self.n_samples, dtype=np.complex64)
        if tcxo_common_drift_hz is None:
            tcxo_common_drift_hz = float(self.rng.uniform(0.8, 1.9))

        stations = self.generate_random_stations(count=station_count)
        ground_truth = []
        start_sample = int(0.5 * self.fs)

        for st in stations:
            symbols, cw_174 = self.generate_ldpc_codeword()
            s_clean, phi_mod = self.synthesize_ft8_waveform(symbols, st.base_freq_hz)
            sig_len = len(s_clean)
            t_sig = np.arange(sig_len) / self.fs

            inst_doppler = st.doppler_shift_hz + st.tid_amp_hz * np.sin(
                2.0 * np.pi * st.tid_freq_hz * t_sig + self.rng.uniform(0, 2 * np.pi)
            )
            phi_iono = 2.0 * np.pi * np.cumsum(inst_doppler) / self.fs
            phi_tcxo = 2.0 * np.pi * tcxo_common_drift_hz * t_sig

            g1 = 0.85 + 0.15 * self._rayleigh_doppler_tap(sig_len, st.doppler_spread_hz)
            path_1f2 = g1 * s_clean * np.exp(1j * (phi_iono + phi_tcxo))

            delay_samples = int(round((st.multipath_delay_ms * 1e-3) * self.fs))
            g2 = 0.32 * self._rayleigh_doppler_tap(sig_len, st.doppler_spread_hz * 1.4)
            path_2f2 = np.zeros_like(path_1f2)
            if delay_samples == 0:
                path_2f2 = g2 * s_clean * np.exp(1j * (phi_iono * 1.35 + phi_tcxo))
            elif delay_samples < sig_len:
                path_2f2[delay_samples:] = (
                    g2[:-delay_samples]
                    * s_clean[:-delay_samples]
                    * np.exp(1j * (phi_iono[:-delay_samples] * 1.35 + phi_tcxo[:-delay_samples]))
                )

            combined = (10.0 ** (st.snr_db / 20.0)) * (path_1f2 + path_2f2)
            avail = max(0, min(sig_len, len(rx_buffer) - start_sample))
            if avail > 0:
                rx_buffer[start_sample : start_sample + avail] += combined[:avail].astype(np.complex64)

            ground_truth.append({
                "station": st,
                "symbols": symbols,
                "codeword_174": cw_174,
                "phi_mod": phi_mod,
                "t_sig": t_sig,
                "inst_doppler_true": inst_doppler,
                "start_sample": start_sample,
            })

        noise_std = 0.35
        awgn = noise_std * (
            self.rng.standard_normal(self.n_samples) + 1j * self.rng.standard_normal(self.n_samples)
        ) / np.sqrt(2.0)
        impulses = (self.rng.random(self.n_samples) < 0.015) * (noise_std * 4.5) * (
            self.rng.standard_normal(self.n_samples) + 1j * self.rng.standard_normal(self.n_samples)
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
        wspr_samples_per_sym = int(round(self.fs * (8192.0 / 12000.0)))
        total_samples = 162 * wspr_samples_per_sym
        rx_buffer = np.zeros(total_samples, dtype=np.complex64)

        if tcxo_drift_slope_hz_min is None:
            tcxo_drift_slope_hz_min = float(self.rng.uniform(0.2, 0.7))

        wspr_stations = self.generate_random_stations(count=station_count, min_freq=1400.0, max_freq=1700.0)
        drifts_hz_min = [float(self.rng.uniform(-1.2, 1.2)) for _ in range(station_count)]
        ground_truth = []
        t_full = np.arange(total_samples) / self.fs

        for st, drift_val in zip(wspr_stations, drifts_hz_min):
            symbols, code_bits = self.generate_wspr_symbols()
            total_drift = drift_val + tcxo_drift_slope_hz_min
            s_clean, _ = self.synthesize_wspr_waveform(symbols, st.base_freq_hz, drift_hz_per_min=total_drift)

            g_t = 0.90 + 0.10 * self._rayleigh_doppler_tap(total_samples, st.doppler_spread_hz)
            sig_rx = g_t * s_clean * np.exp(1j * 2.0 * np.pi * st.doppler_shift_hz * t_full)
            rx_buffer += ((10.0 ** (st.snr_db / 20.0)) * sig_rx).astype(np.complex64)

            ground_truth.append({
                "station": st,
                "symbols": symbols,
                "code_bits": code_bits,
                "drift_hz_per_min": drift_val,
                "total_drift_hz_min": total_drift,
                "base_freq_hz": st.base_freq_hz,
            })

        awgn = 0.40 * (
            self.rng.standard_normal(total_samples) + 1j * self.rng.standard_normal(total_samples)
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
