#!/usr/bin/env python3
"""Core DSP, Common-Mode TCXO Calibration, Blind Costas Scanner, and Micro-Doppler Engine."""

from typing import Dict, List, Optional, Tuple
from iono_sense.simulation.watterson import WSPR_SYNC_VECTOR
import numpy as np
from scipy import signal
from scipy.special import erfc


class IonosphericDSPCore:
    """Channel isolation, phase unwrapping, Savitzky-Golay micro-Doppler, and delay-Doppler S(tau, fd) solver."""

    def __init__(self, fs: float = 12000.0):
        self.fs = fs
        self.sym_dur = 0.16
        self.tone_spacing = 6.25
        self.samples_per_sym = int(round(self.fs * self.sym_dur))
        self.n_symbols = 79

    def isolate_station_channel(
        self, iq_buffer: np.ndarray, f_center_hz: float, bw_hz: float = 65.0
    ) -> np.ndarray:
        t = np.arange(len(iq_buffer)) / self.fs
        shifted = iq_buffer * np.exp(-1j * 2.0 * np.pi * f_center_hz * t)
        numtaps = 257
        if len(shifted) <= numtaps:
            return shifted
        taps = signal.firwin(
            numtaps, cutoff=bw_hz / 2.0, fs=self.fs, window="hamming"
        )
        if len(shifted) > 3 * numtaps:
            return signal.filtfilt(
                taps,
                [1.0],
                shifted,
                padlen=min(3 * numtaps, len(shifted) - 1),
            )
        return signal.lfilter(taps, [1.0], shifted)

    def estimate_station_parameters(
        self, iq_buffer: np.ndarray, gt_entry: Dict, f_carrier_mhz: float
    ) -> Dict:
        st = gt_entry["station"]
        start_idx = gt_entry["start_sample"]
        sig_len = self.n_symbols * self.samples_per_sym

        iq_chan = self.isolate_station_channel(
            iq_buffer, st.base_freq_hz, bw_hz=75.0
        )
        iq_pkt = iq_chan[start_idx : start_idx + sig_len]
        if len(iq_pkt) < sig_len:
            iq_pkt = np.pad(iq_pkt, (0, sig_len - len(iq_pkt)))

        z_stripped = iq_pkt * np.exp(-1j * gt_entry["phi_mod"])
        numtaps_n = min(
            513, len(z_stripped) - (1 if len(z_stripped) % 2 == 0 else 0)
        )
        if numtaps_n >= 15:
            if numtaps_n % 2 == 0:
                numtaps_n -= 1
            taps_n = signal.firwin(
                numtaps_n, cutoff=8.0, fs=self.fs, window="blackman"
            )
            z_smooth = signal.filtfilt(
                taps_n,
                [1.0],
                z_stripped,
                padlen=min(3 * numtaps_n, len(z_stripped) - 1),
            )
        else:
            z_smooth = z_stripped

        phi_unwrapped = np.unwrap(np.angle(z_smooth))
        t_sig = gt_entry["t_sig"]
        mean_freq_offset_hz = float(
            np.polyfit(t_sig, phi_unwrapped, deg=1)[0] / (2.0 * np.pi)
        )

        raw_symbols = np.zeros(self.n_symbols, dtype=np.int32)
        all_tone_energies = np.zeros((self.n_symbols, 8), dtype=np.float64)
        t_seg = np.arange(self.samples_per_sym) / self.fs
        ref_tones = np.array([
            np.exp(-1j * 2.0 * np.pi * (tone * self.tone_spacing) * t_seg)
            for tone in range(8)
        ])

        for s_idx in range(self.n_symbols):
            seg = iq_pkt[
                s_idx * self.samples_per_sym : (s_idx + 1) * self.samples_per_sym
            ]
            seg_corr = seg * np.exp(
                -1j * 2.0 * np.pi * mean_freq_offset_hz * t_seg
            )
            tone_energies = np.abs(ref_tones @ seg_corr) ** 2
            all_tone_energies[s_idx] = tone_energies
            raw_symbols[s_idx] = int(np.argmax(tone_energies))

        rx_data_syms = np.concatenate([raw_symbols[7:36], raw_symbols[43:72]])
        tone_energies_data = np.concatenate(
            [all_tone_energies[7:36], all_tone_energies[43:72]], axis=0
        )
        rx_bits = (
            np.column_stack([
                (rx_data_syms >> 2) & 1,
                (rx_data_syms >> 1) & 1,
                rx_data_syms & 1,
            ])
            .flatten()
            .astype(np.int32)
        )

        true_bits = gt_entry["codeword_174"]
        bit_errors = int(np.sum(rx_bits != true_bits))

        return {
            "callsign": st.callsign,
            "distance_km": st.distance_km,
            "snr_db": st.snr_db,
            "mean_freq_offset_hz": mean_freq_offset_hz,
            "phi_unwrapped": phi_unwrapped,
            "z_smooth": z_smooth,
            "t_sig": t_sig,
            "inst_doppler_true": gt_entry["inst_doppler_true"],
            "pre_fec_ber": float(bit_errors / 174.0),
            "bit_errors": bit_errors,
            "rx_bits": rx_bits,
            "true_bits": true_bits,
            "tone_energies_data": tone_energies_data,
            "f_carrier_mhz": f_carrier_mhz,
        }

    def blind_scan_and_extract(
        self,
        iq_buffer: np.ndarray,
        f_carrier_mhz: float = 14.074,
        max_stations: int = 6,
    ) -> List[Dict]:
        """2D Blind Costas sync and candidate estimation for active FT8 signals in real .wav / .iq SDR recordings without ground_truth.

        
        """
        costas = np.array([3, 1, 4, 0, 6, 5, 2], dtype=np.int32)
        sig_len = self.n_symbols * self.samples_per_sym
        if len(iq_buffer) < sig_len or np.max(np.abs(iq_buffer)) < 1e-9:
            return []

        step_samples = self.samples_per_sym // 2
        n_steps = (len(iq_buffer) - self.samples_per_sym) // step_samples + 1
        win = np.hanning(self.samples_per_sym)
        spec_grid = np.zeros((n_steps, self.samples_per_sym), dtype=np.float64)

        for idx in range(n_steps):
            seg = (
                iq_buffer[
                    idx * step_samples : idx * step_samples + self.samples_per_sym
                ]
                * win
            )
            spec_grid[idx] = np.abs(np.fft.fft(seg)) ** 2

        freqs = np.fft.fftfreq(self.samples_per_sym, d=1.0 / self.fs)
        valid_bins = np.where((freqs >= 350.0) & (freqs <= 2600.0))[0]
        if len(valid_bins) == 0:
            return []

        noise_floor = float(np.percentile(spec_grid[:, valid_bins], 30)) + 1e-12
        max_start_step = max(1, n_steps - (self.n_symbols * 2))
        candidates = []

        for b_idx in valid_bins[:-8]:
            best_score = -1.0
            best_t_step = 0
            for t_step in range(0, max_start_step):
                sync_e = 0.0
                for blk in (0, 36, 72):
                    for k_sym, c_tone in enumerate(costas):
                        s_step = t_step + 2 * (blk + k_sym)
                        if s_step < n_steps:
                            sync_e += spec_grid[s_step, b_idx + c_tone]
                score = sync_e / (21.0 * noise_floor)
                if score > best_score:
                    best_score = score
                    best_t_step = t_step

            if best_score > 3.5:
                candidates.append(
                    (best_score, float(freqs[b_idx]), best_t_step * step_samples)
                )

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = []
        for score, f0_hz, start_idx in candidates:
            if all(abs(f0_hz - s[1]) > 55.0 for s in selected):
                selected.append((score, f0_hz, start_idx))
            if len(selected) >= max_stations:
                break

        selected.sort(key=lambda x: x[1])
        results = []
        t_seg = np.arange(self.samples_per_sym) / self.fs
        ref_tones = np.array([
            np.exp(-1j * 2.0 * np.pi * (tone * self.tone_spacing) * t_seg)
            for tone in range(8)
        ])

        gauss_len = 65
        t_g = np.linspace(-2.0, 2.0, gauss_len)
        g_win = np.exp(-2.0 * (np.pi**2) * (t_g**2))
        g_win /= np.sum(g_win)

        for _, (score, f0_hz, start_idx) in enumerate(selected):
            iq_chan = self.isolate_station_channel(iq_buffer, f0_hz, bw_hz=75.0)
            iq_pkt = iq_chan[start_idx : start_idx + sig_len]
            if len(iq_pkt) < sig_len:
                iq_pkt = np.pad(iq_pkt, (0, sig_len - len(iq_pkt)))

            raw_symbols = np.zeros(self.n_symbols, dtype=np.int32)
            all_tone_energies = np.zeros((self.n_symbols, 8), dtype=np.float64)
            for s_idx in range(self.n_symbols):
                seg = iq_pkt[
                    s_idx
                    * self.samples_per_sym : (s_idx + 1)
                    * self.samples_per_sym
                ]
                te = np.abs(ref_tones @ seg) ** 2
                all_tone_energies[s_idx] = te
                raw_symbols[s_idx] = int(np.argmax(te))

            freq_trace = np.repeat(
                raw_symbols.astype(np.float64) * self.tone_spacing,
                self.samples_per_sym,
            )
            freq_smooth = np.convolve(freq_trace, g_win, mode="same")
            phi_mod_est = 2.0 * np.pi * np.cumsum(freq_smooth) / self.fs

            z_stripped = iq_pkt * np.exp(-1j * phi_mod_est)
            taps_n = signal.firwin(513, cutoff=8.0, fs=self.fs, window="blackman")
            z_smooth = signal.filtfilt(taps_n, [1.0], z_stripped)

            phi_unwrapped = np.unwrap(np.angle(z_smooth))
            t_sig = np.arange(sig_len) / self.fs
            mean_freq_offset_hz = float(
                np.polyfit(t_sig, phi_unwrapped, deg=1)[0] / (2.0 * np.pi)
            )

            costas_rx = np.concatenate(
                [raw_symbols[0:7], raw_symbols[36:43], raw_symbols[72:79]]
            )
            costas_true = np.tile(costas, 3)
            sync_sym_errors = int(np.sum(costas_rx != costas_true))
            est_ber = float(min(0.49, (sync_sym_errors / 21.0) * 0.5))

            rx_data_syms = np.concatenate([raw_symbols[7:36], raw_symbols[43:72]])
            tone_energies_data = np.concatenate(
                [all_tone_energies[7:36], all_tone_energies[43:72]], axis=0
            )
            rx_bits = (
                np.column_stack([
                    (rx_data_syms >> 2) & 1,
                    (rx_data_syms >> 1) & 1,
                    rx_data_syms & 1,
                ])
                .flatten()
                .astype(np.int32)
            )

            snr_db = float(np.clip(10.0 * np.log10(score) - 14.0, -24.0, 25.0))

            results.append({
                "callsign": f"SIG_{int(round(f0_hz))}Hz",
                "distance_km": 2200.0,
                "snr_db": snr_db,
                "mean_freq_offset_hz": mean_freq_offset_hz,
                "phi_unwrapped": phi_unwrapped,
                "z_smooth": z_smooth,
                "t_sig": t_sig,
                "inst_doppler_true": np.zeros_like(t_sig),
                "pre_fec_ber": est_ber,
                "bit_errors": int(round(est_ber * 174)),
                "rx_bits": rx_bits,
                "true_bits": rx_bits,
                "tone_energies_data": tone_energies_data,
                "f_carrier_mhz": f_carrier_mhz,
            })

        return results

    def calibrate_and_extract_doppler(
        self, station_results: List[Dict]
    ) -> Tuple[float, List[Dict]]:
        if not station_results:
            return 0.0, []

        estimated_tcxo_hz = float(
            np.median([r["mean_freq_offset_hz"] for r in station_results])
        )
        c_mps = 2.99792458e8
        cos_inc = max(abs(float(np.cos(np.radians(72.0)))), 1e-3)

        for r in station_results:
            t_sig = r["t_sig"]
            if len(t_sig) < 2:
                continue
            dt = float(t_sig[1] - t_sig[0])
            phi_cal = r["phi_unwrapped"] - 2.0 * np.pi * estimated_tcxo_hz * t_sig

            n_pts = len(phi_cal)
            win_len = min(1201, n_pts if (n_pts % 2 == 1) else (n_pts - 1))
            if win_len >= 5:
                dphi_dt = signal.savgol_filter(
                    phi_cal,
                    window_length=win_len,
                    polyorder=3,
                    deriv=1,
                    delta=dt,
                )
            else:
                dphi_dt = np.gradient(phi_cal, dt)

            fd_est = dphi_dt / (2.0 * np.pi)
            fc_hz = max(float(r["f_carrier_mhz"]) * 1e6, 1e3)

            r["fd_est"] = fd_est
            r["doppler_spread_hz"] = float(np.std(fd_est))
            r["vz_mps"] = -(c_mps / (2.0 * fc_hz * cos_inc)) * fd_est
            r["calibrated_mean_fd_hz"] = (
                r["mean_freq_offset_hz"] - estimated_tcxo_hz
            )

        return estimated_tcxo_hz, station_results

    def analyze_wspr_signals(
        self, wspr_slot: Dict, f_carrier_mhz: float = 14.0956
    ) -> Tuple[float, List[Dict]]:
        c_mps = 2.99792458e8
        fc_hz = f_carrier_mhz * 1e6
        cos_theta = np.cos(np.radians(72.0))
        rx_buf = np.asarray(
            wspr_slot.get("rx_buffer", np.array([], dtype=np.complex64))
        )
        total_len = len(rx_buf)
        if total_len == 0 or not wspr_slot.get("ground_truth"):
            return 0.0, []

        wspr_tone_spacing = 1.46484375
        wspr_sym_samples = int(round(self.fs * (8192.0 / 12000.0)))
        n_symbols = 162
        req_samples = n_symbols * wspr_sym_samples

        t_matrix = (np.arange(req_samples) / self.fs).reshape(
            n_symbols, wspr_sym_samples
        )
        t_sym = np.arange(wspr_sym_samples) / self.fs
        tones_0 = np.array([
            np.exp(-1j * 2.0 * np.pi * (m * wspr_tone_spacing) * t_sym)
            for m in [0, 1]
        ])
        tones_1 = np.array([
            np.exp(-1j * 2.0 * np.pi * (m * wspr_tone_spacing) * t_sym)
            for m in [2, 3]
        ])
        sync_0, sync_1 = (WSPR_SYNC_VECTOR == 0), (WSPR_SYNC_VECTOR == 1)

        def score_drifts(chan_mat: np.ndarray, drifts: np.ndarray) -> np.ndarray:
            scores = []
            for d in drifts:
                sig_d = chan_mat * np.exp(
                    -1j * np.pi * (d / 60.0) * (t_matrix**2)
                )
                c0 = np.abs(sig_d[sync_0] @ tones_0.T) ** 2
                c1 = np.abs(sig_d[sync_1] @ tones_1.T) ** 2
                scores.append(
                    float(
                        np.sum(np.max(c0, axis=1)) + np.sum(np.max(c1, axis=1))
                    )
                )
            return np.array(scores)

        results = []
        t_vec = np.arange(total_len) / self.fs
        numtaps = 513
        taps = signal.firwin(numtaps, cutoff=12.5, fs=self.fs, window="hamming")

        for gt in wspr_slot["ground_truth"]:
            st = gt["station"]
            shifted = rx_buf * np.exp(-1j * 2.0 * np.pi * st.base_freq_hz * t_vec)
            chan = (
                signal.filtfilt(taps, [1.0], shifted)
                if total_len > 3 * numtaps
                else shifted
            )

            if len(chan) < req_samples:
                chan = np.pad(
                    chan, (0, req_samples - len(chan)), mode="constant"
                )
            else:
                chan = chan[:req_samples]

            X = np.fft.fft(chan)
            psd = np.abs(X) ** 2
            pb = np.abs(np.fft.fftfreq(len(chan), d=1.0 / self.fs)) < 12.5
            med_pb = float(np.median(psd[pb]))
            notched = False
            for _ in range(3):
                max_idx = int(np.argmax(np.where(pb, psd, 0.0)))
                if psd[max_idx] > 50000.0 * med_pb and med_pb > 0:
                    k0, k1 = max(0, max_idx - 7), min(len(X), max_idx + 8)
                    X[k0:k1] = 0.0
                    psd[k0:k1] = 0.0
                    notched = True
                else:
                    break
            if notched:
                chan = np.fft.ifft(X).astype(np.complex64)

            chan_matrix = chan.reshape(n_symbols, wspr_sym_samples)
            coarse_drifts = np.linspace(-2.4, 2.4, 25)
            d_coarse = coarse_drifts[
                int(np.argmax(score_drifts(chan_matrix, coarse_drifts)))
            ]

            fine_drifts = np.linspace(d_coarse - 0.25, d_coarse + 0.25, 11)
            meas_drift = float(
                fine_drifts[
                    int(np.argmax(score_drifts(chan_matrix, fine_drifts)))
                ]
            )

            results.append({
                "callsign": st.callsign,
                "grid": st.grid,
                "distance_km": st.distance_km,
                "snr_db": st.snr_db,
                "true_drift_hz_min": gt["drift_hz_per_min"],
                "total_drift_hz_min": gt["total_drift_hz_min"],
                "meas_drift_hz_min": meas_drift,
            })

        est_tcxo_slope = (
            float(np.median([r["meas_drift_hz_min"] for r in results]))
            if results
            else 0.0
        )
        for r in results:
            net_drift = r["meas_drift_hz_min"] - est_tcxo_slope
            r["net_iono_drift_hz_min"] = net_drift
            r["layer_accel_mps2"] = -(c_mps / (2.0 * fc_hz * cos_theta)) * (
                net_drift / 60.0
            )

        return est_tcxo_slope, results

    def compute_scattering_function(
        self,
        z_smooth: Optional[np.ndarray],
        s_ref: Optional[np.ndarray] = None,
        tau_ms: Optional[np.ndarray] = None,
        fd_hz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tau_ms is None:
            tau_ms = np.linspace(0.0, 3.5, 80)
        if fd_hz is None:
            fd_hz = np.linspace(-1.5, 1.5, 80)

        n_tau, n_fd = len(tau_ms), len(fd_hz)
        if (
            z_smooth is None
            or len(z_smooth) < 20
            or np.all(np.abs(z_smooth) < 1e-10)
        ):
            return tau_ms, fd_hz, np.full((n_fd, n_tau), -40.0, dtype=np.float64)

        z = np.nan_to_num(z_smooth, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.complex128
        )
        N = len(z)
        ds = max(1, int(round(self.fs / 200.0)))
        fs_eff = self.fs / ds
        S_mat = np.zeros((n_fd, n_tau), dtype=np.float64)

        for i, tau in enumerate(tau_ms):
            lag = int(round((tau * 1e-3) * self.fs))
            if lag < 0 or lag >= N:
                continue

            if s_ref is not None and len(s_ref) > 0:
                s = np.nan_to_num(s_ref, nan=0.0).astype(np.complex128)
                len_corr = min(N - lag, len(s))
                if len_corr <= 0:
                    continue
                y_tau = z[lag : lag + len_corr] * np.conj(s[:len_corr])
            else:
                y_tau = z if lag == 0 else z[lag:] * np.conj(z[: N - lag])

            if len(y_tau) >= ds * 4:
                numtaps = min(31, len(y_tau) - 1)
                if numtaps % 2 == 0:
                    numtaps -= 1
                if numtaps >= 5:
                    taps = signal.firwin(
                        numtaps, cutoff=0.8 / ds, window="hamming"
                    )
                    y_ds = signal.filtfilt(taps, [1.0], y_tau)[::ds]
                else:
                    y_ds = y_tau[::ds]
            else:
                y_ds = y_tau

            if len(y_ds) < 4:
                continue

            fft_y = np.fft.fftshift(
                np.fft.fft(y_ds * np.hanning(len(y_ds)), n=max(n_fd, len(y_ds)))
            )
            freqs_y = np.fft.fftshift(
                np.fft.fftfreq(len(fft_y), d=1.0 / fs_eff)
            )
            S_mat[:, i] = np.interp(
                fd_hz, freqs_y, np.abs(fft_y) ** 2, left=0.0, right=0.0
            )

        max_p = np.max(S_mat)
        S_db = (
            10.0 * np.log10(S_mat / max_p + 1e-4)
            if max_p > 0.0
            else np.full((n_fd, n_tau), -40.0)
        )
        return tau_ms, fd_hz, np.clip(S_db, -40.0, 0.0)

    @staticmethod
    def compute_multimode_ber_curves() -> Dict[str, np.ndarray]:
        ebn0_db = np.linspace(-4.0, 18.0, 45)
        ebn0_lin = 10.0 ** (ebn0_db / 10.0)

        ber_psk31 = (
            0.5 * (1.0 - np.sqrt(ebn0_lin / (1.0 + ebn0_lin))) + 4e-3
        )
        ber_stanag = (
            0.35 * erfc(np.sqrt(0.45 * ebn0_lin)) * np.exp(-0.18 * ebn0_lin)
            + 1.5e-4
        )
        ber_ft8 = 0.5 * erfc(np.sqrt(0.65 * (ebn0_lin**1.45)))
        ber_wspr = 0.5 * erfc(np.sqrt(0.95 * (ebn0_lin**1.65)))

        return {
            "ebn0_db": ebn0_db,
            "ber_psk31": np.clip(ber_psk31, 1e-6, 0.5),
            "ber_stanag": np.clip(ber_stanag, 1e-6, 0.5),
            "ber_ft8": np.clip(ber_ft8, 1e-6, 0.5),
            "ber_wspr": np.clip(ber_wspr, 1e-6, 0.5),
        }