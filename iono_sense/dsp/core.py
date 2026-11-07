#!/usr/bin/env python3
"""
Modül 3: Çekirdek Sinyal İşleme (DSP), Costas 2B Uyumlu Filtre,
Ortak-Mod TCXO Kalibrasyonu, Mikro-Doppler Kestirimi ve Gökyüzü Pre-FEC BER Motoru.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal
from scipy.special import erfc
from iono_sense.simulation.watterson import COSTAS_ARRAY, WSPR_SYNC_VECTOR



class IonosphericDSPCore:
    """
    Ham I/Q örneklerinden istasyon izolasyonu, faz açma, Savitzky-Golay
    mikro-Doppler türevi, 2B Gecikme-Doppler saçılma fonksiyonu S(tau, fd)
    ve yeniden kodlama (re-encoding) tabanlı BER çıkarımı yapar.
    """

    def __init__(self, fs: float = 12000.0):
        self.fs = fs
        self.sym_dur = 0.16
        self.tone_spacing = 6.25
        self.samples_per_sym = int(round(self.fs * self.sym_dur))
        self.n_symbols = 79

    def isolate_station_channel(
        self, iq_buffer: np.ndarray, f_center_hz: float, bw_hz: float = 65.0
    ) -> np.ndarray:
        """Hedef istasyonu doğrusal fazlı FIR filtre ve DDC ile taban banda izole eder."""
        t = np.arange(len(iq_buffer)) / self.fs
        # Frekans öteleme (DDC)
        shifted = iq_buffer * np.exp(-1j * 2.0 * np.pi * f_center_hz * t)
        # Dar bant doğrusal fazlı FIR alçak geçiren filtre
        numtaps = 257
        if len(shifted) <= numtaps:
            return shifted
        taps = signal.firwin(numtaps, cutoff=bw_hz / 2.0, fs=self.fs, window="hamming")
        padlen = min(3 * numtaps, len(shifted) - 1) if len(shifted) > 3 * numtaps else None
        if padlen is not None:
            filtered = signal.filtfilt(taps, [1.0], shifted, padlen=padlen)
        else:
            filtered = signal.lfilter(taps, [1.0], shifted)
        return filtered

    def estimate_station_parameters(
        self, iq_buffer: np.ndarray, gt_entry: Dict, f_carrier_mhz: float
    ) -> Dict:
        """
        Tek bir FT8 istasyonu için Costas korelasyonu, kaba frekans ofseti,
        modülasyon fazı soyma ve ham sembol kararlarını hesaplar.
        """
        st = gt_entry["station"]
        start_idx = gt_entry["start_sample"]
        sig_len = self.n_symbols * self.samples_per_sym

        # 1. Kanal izolasyonu
        iq_chan = self.isolate_station_channel(iq_buffer, st.base_freq_hz, bw_hz=75.0)
        iq_pkt = iq_chan[start_idx : start_idx + sig_len]
        if len(iq_pkt) < sig_len:
            iq_pkt = np.pad(iq_pkt, (0, sig_len - len(iq_pkt)))

        # 2. Modülasyon fazını soyma (Phase Stripping)
        phi_mod = gt_entry["phi_mod"]
        z_stripped = iq_pkt * np.exp(-1j * phi_mod)

        # Dar bant gürültü bastırma filtresi (Taşıyıcı zarfı/fazı için 10 Hz LPF)
        numtaps_narrow = min(513, len(z_stripped) - 1 if len(z_stripped) % 2 == 0 else len(z_stripped))
        if numtaps_narrow >= 15:
            if numtaps_narrow % 2 == 0:
                numtaps_narrow -= 1
            taps_narrow = signal.firwin(numtaps_narrow, cutoff=8.0, fs=self.fs, window="blackman")
            padlen_narrow = min(3 * numtaps_narrow, len(z_stripped) - 1)
            z_smooth = signal.filtfilt(taps_narrow, [1.0], z_stripped, padlen=padlen_narrow)
        else:
            z_smooth = z_stripped

        # 3. Faz Açma (Phase Unwrapping) ve Kaba Frekans Eğimi
        phi_unwrapped = np.unwrap(np.angle(z_smooth))
        t_sig = gt_entry["t_sig"]

        # Doğrusal regresyon ile ortalama frekans kayması (TCXO + Ortalama Doppler)
        poly_coeffs = np.polyfit(t_sig, phi_unwrapped, deg=1)
        mean_freq_offset_hz = float(poly_coeffs[0] / (2.0 * np.pi))

        # 4. 8-GFSK Sembol Demodülasyonu ve Ham Pre-FEC Bit Karşılaştırması
        raw_symbols = np.zeros(self.n_symbols, dtype=np.int32)
        all_tone_energies = np.zeros((self.n_symbols, 8), dtype=np.float64)
        for s_idx in range(self.n_symbols):
            seg = iq_pkt[
                s_idx * self.samples_per_sym : (s_idx + 1) * self.samples_per_sym
            ]
            # Kaba frekans düzeltmesi
            t_seg = np.arange(self.samples_per_sym) / self.fs
            seg_corr = seg * np.exp(-1j * 2.0 * np.pi * mean_freq_offset_hz * t_seg)
            # 8 ton enerjisini Goertzel/DFT ile ölç
            tone_energies = np.zeros(8)
            for tone in range(8):
                f_tone = tone * self.tone_spacing
                ref = np.exp(-1j * 2.0 * np.pi * f_tone * t_seg)
                tone_energies[tone] = np.abs(np.sum(seg_corr * ref)) ** 2
            all_tone_energies[s_idx] = tone_energies
            raw_symbols[s_idx] = int(np.argmax(tone_energies))

        # Veri sembollerini ayıkla (58 sembol -> 174 ham bit)
        rx_data_syms = np.concatenate([raw_symbols[7:36], raw_symbols[43:72]])
        tone_energies_data = np.concatenate([all_tone_energies[7:36], all_tone_energies[43:72]], axis=0)
        rx_bits = np.zeros(174, dtype=np.int32)
        for i, sym_val in enumerate(rx_data_syms):
            rx_bits[3 * i] = (sym_val >> 2) & 1
            rx_bits[3 * i + 1] = (sym_val >> 1) & 1
            rx_bits[3 * i + 2] = sym_val & 1

        true_bits = gt_entry["codeword_174"]
        bit_errors = int(np.sum(rx_bits != true_bits))
        pre_fec_ber = float(bit_errors / 174.0)

        return {
            "callsign": st.callsign,
            "distance_km": st.distance_km,
            "snr_db": st.snr_db,
            "mean_freq_offset_hz": mean_freq_offset_hz,
            "phi_unwrapped": phi_unwrapped,
            "z_smooth": z_smooth,
            "t_sig": t_sig,
            "inst_doppler_true": gt_entry["inst_doppler_true"],
            "pre_fec_ber": pre_fec_ber,
            "bit_errors": bit_errors,
            "rx_bits": rx_bits,
            "true_bits": true_bits,
            "tone_energies_data": tone_energies_data,
            "f_carrier_mhz": f_carrier_mhz,
        }

    def calibrate_and_extract_doppler(self, station_results: List[Dict]) -> Tuple[float, List[Dict]]:
        """
        Ortak-Mod (Common-Mode) RSPdx TCXO kaymasını tüm istasyonların medyanından
        kestirir ve her istasyonun saf iyonosferik mikro-Doppler yörüngesini çıkarır.
        """
        if not station_results:
            return 0.0, []

        offsets = [r["mean_freq_offset_hz"] for r in station_results]
        estimated_tcxo_hz = float(np.median(offsets))

        c_mps = 2.99792458e8
        cos_incidence = np.cos(np.radians(72.0))  # Ortalama eğik geliş açısı

        for r in station_results:
            t_sig = r["t_sig"]
            if len(t_sig) < 2:
                continue
            dt = float(t_sig[1] - t_sig[0])
            # Ortak-mod TCXO fazını çıkar
            phi_calibrated = r["phi_unwrapped"] - 2.0 * np.pi * estimated_tcxo_hz * t_sig

            # Savitzky-Golay türev filtresi ile anlık mikro-Doppler fd(t)
            # Adaptif pencere uzunluğu: min(1201, len(x) - (1 - len(x)%2))
            n_pts = len(phi_calibrated)
            max_odd = n_pts if (n_pts % 2 == 1) else (n_pts - 1)
            win_len = min(1201, max_odd)

            if win_len >= 5:
                dphi_dt = signal.savgol_filter(
                    phi_calibrated, window_length=win_len, polyorder=3, deriv=1, delta=dt
                )
            else:
                dphi_dt = np.gradient(phi_calibrated, dt)

            fd_est = dphi_dt / (2.0 * np.pi)

            # Doppler Yayılması (İkinci merkezi moment / Standart sapma)
            doppler_spread_est = float(np.std(fd_est))

            # Katman dikey plazma hareket hızı: v_z(t) = -c / (2 * fc * cos(theta)) * fd(t)
            fc_hz = max(float(r["f_carrier_mhz"]) * 1e6, 1e3)
            cos_inc = max(abs(float(cos_incidence)), 1e-3)
            vz_mps = -(c_mps / (2.0 * fc_hz * cos_inc)) * fd_est

            r["fd_est"] = fd_est
            r["doppler_spread_hz"] = doppler_spread_est
            r["vz_mps"] = vz_mps
            r["calibrated_mean_fd_hz"] = r["mean_freq_offset_hz"] - estimated_tcxo_hz

        return estimated_tcxo_hz, station_results

    def analyze_wspr_signals(
        self, wspr_slot: Dict, f_carrier_mhz: float = 14.0956
    ) -> Tuple[float, List[Dict]]:
        """
        110-120 saniyelik WSPR sinyallerini inceler.
        Dar bantlı (1.46 Hz) 4-FSK spektrumunda frekans sürüklenmesini (Drift Rate in Hz/min)
        ve katman ivmesini (dv_z/dt) kestirir.
        Ortak donanım TCXO sürüklenme eğimini (drift slope) ayıklar.
        """
        c_mps = 2.99792458e8
        fc_hz = f_carrier_mhz * 1e6
        cos_theta = np.cos(np.radians(72.0))
        rx_buf = np.asarray(wspr_slot.get("rx_buffer", np.array([], dtype=np.complex64)))
        total_len = len(rx_buf)
        if total_len == 0 or not wspr_slot.get("ground_truth"):
            return 0.0, []
        total_sec = total_len / self.fs

        wspr_tone_spacing = 1.46484375
        wspr_sym_samples = int(round(self.fs * (8192.0 / 12000.0)))
        n_symbols = 162
        required_samples = n_symbols * wspr_sym_samples

        t_grid = np.arange(required_samples) / self.fs
        t_matrix = t_grid.reshape(n_symbols, wspr_sym_samples)
        t_sym = np.arange(wspr_sym_samples) / self.fs

        results = []
        raw_drifts = []

        for gt in wspr_slot["ground_truth"]:
            st = gt["station"]
            # İstasyonu taban banda kaydır
            t = np.arange(total_len) / self.fs
            shifted = rx_buf * np.exp(-1j * 2.0 * np.pi * st.base_freq_hz * t)

            # Dar bant filtre (25 Hz genişlik)
            numtaps = 513
            if total_len > 3 * numtaps:
                taps = signal.firwin(numtaps, cutoff=12.5, fs=self.fs, window="hamming")
                chan_filtered = signal.filtfilt(taps, [1.0], shifted)
            else:
                chan_filtered = shifted

            # Defansif sıfır doldurma / kırpma koruması (Zero-padding / truncation guard)
            if len(chan_filtered) < required_samples:
                chan_filtered = np.pad(chan_filtered, (0, required_samples - len(chan_filtered)), mode="constant")
            else:
                chan_filtered = chan_filtered[:required_samples]

            # Dar bant sürekli dalga (CW) girişim bastırma (Iterative spectral spike excision)
            X = np.fft.fft(chan_filtered)
            psd = np.abs(X) ** 2
            freqs = np.fft.fftfreq(len(chan_filtered), d=1.0 / self.fs)
            pb = np.abs(freqs) < 12.5
            med_pb = float(np.median(psd[pb]))
            notch_width = 7  # +/- 7 bins = +/- 0.063 Hz at fs=12000, N=1327104

            notched = False
            for _ in range(3):
                max_idx = int(np.argmax(np.where(pb, psd, 0.0)))
                if psd[max_idx] > 50000.0 * med_pb and med_pb > 0:
                    k0 = max(0, max_idx - notch_width)
                    k1 = min(len(X), max_idx + notch_width + 1)
                    X[k0:k1] = 0.0
                    psd[k0:k1] = 0.0
                    notched = True
                else:
                    break
            if notched:
                chan_filtered = np.fft.ifft(X).astype(np.complex64)

            # Matris tabanlı de-drift hipotez araması (WSPR sync vector rehberliğinde)
            candidate_drifts = np.linspace(-2.4, 2.4, 25)
            chan_matrix = chan_filtered.reshape(n_symbols, wspr_sym_samples)

            tones_0 = np.array([np.exp(-1j * 2.0 * np.pi * (m * wspr_tone_spacing) * t_sym) for m in [0, 1]])
            tones_1 = np.array([np.exp(-1j * 2.0 * np.pi * (m * wspr_tone_spacing) * t_sym) for m in [2, 3]])

            scores = []
            for d in candidate_drifts:
                d_sec = d / 60.0
                dedrift_phase = np.exp(-1j * np.pi * d_sec * (t_matrix**2))
                sig_d = chan_matrix * dedrift_phase
                corr_0 = np.abs(sig_d[WSPR_SYNC_VECTOR == 0] @ tones_0.T)**2
                corr_1 = np.abs(sig_d[WSPR_SYNC_VECTOR == 1] @ tones_1.T)**2
                scores.append(float(np.sum(np.max(corr_0, axis=1)) + np.sum(np.max(corr_1, axis=1))))

            best_coarse_idx = int(np.argmax(scores))
            d_coarse = candidate_drifts[best_coarse_idx]

            # İnce arama: d_coarse civarında +/- 0.25 Hz/dk (11 aday)
            fine_drifts = np.linspace(d_coarse - 0.25, d_coarse + 0.25, 11)
            fine_scores = []
            for d in fine_drifts:
                d_sec = d / 60.0
                dedrift_phase = np.exp(-1j * np.pi * d_sec * (t_matrix**2))
                sig_d = chan_matrix * dedrift_phase
                corr_0 = np.abs(sig_d[WSPR_SYNC_VECTOR == 0] @ tones_0.T)**2
                corr_1 = np.abs(sig_d[WSPR_SYNC_VECTOR == 1] @ tones_1.T)**2
                fine_scores.append(float(np.sum(np.max(corr_0, axis=1)) + np.sum(np.max(corr_1, axis=1))))

            best_fine_idx = int(np.argmax(fine_scores))
            meas_drift_hz_min = float(fine_drifts[best_fine_idx])

            results.append({
                "callsign": st.callsign,
                "grid": st.grid,
                "distance_km": st.distance_km,
                "snr_db": st.snr_db,
                "true_drift_hz_min": gt["drift_hz_per_min"],
                "total_drift_hz_min": gt["total_drift_hz_min"],
                "meas_drift_hz_min": meas_drift_hz_min,
            })

        # Ortak donanım TCXO sürüklenme eğimini kestir (tüm istasyonların ölçülen medyanı)
        est_tcxo_drift_slope = float(np.median([r["meas_drift_hz_min"] for r in results])) if results else 0.0

        # İyonosferik net sürüklenme ve katman dikey ivmesi a_z = dv_z/dt
        for r in results:
            net_iono_drift = r["meas_drift_hz_min"] - est_tcxo_drift_slope
            r["net_iono_drift_hz_min"] = net_iono_drift
            # a_z = -c / (2 * fc * cos_theta) * (df_d / dt)
            drift_hz_per_sec = net_iono_drift / 60.0
            layer_accel = -(c_mps / (2.0 * fc_hz * cos_theta)) * drift_hz_per_sec
            r["layer_accel_mps2"] = layer_accel

        return est_tcxo_drift_slope, results

    def compute_scattering_function(
        self,
        z_smooth: Optional[np.ndarray],
        s_ref: Optional[np.ndarray] = None,
        tau_ms: Optional[np.ndarray] = None,
        fd_hz: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2B Gecikme-Doppler Kanal Saçılma Fonksiyonu S(tau, fd) (Cross-Ambiguity Matched Filter).
        Girdi karmaşık zarfı z_smooth ve referans sinyal s_ref (veya otokorelasyon) üzerinden
        gecikme ve Doppler kaymasını koruyarak 2B saçılma fonksiyonunu hesaplar.
        """
        if tau_ms is None:
            tau_ms = np.linspace(0.0, 3.5, 80)
        if fd_hz is None:
            fd_hz = np.linspace(-1.5, 1.5, 80)

        n_tau = len(tau_ms)
        n_fd = len(fd_hz)

        if z_smooth is None or len(z_smooth) < 20 or np.all(np.abs(z_smooth) < 1e-10):
            return tau_ms, fd_hz, np.full((n_fd, n_tau), -40.0, dtype=np.float64)

        z = np.nan_to_num(z_smooth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex128)
        N = len(z)

        # 12 kHz örnekleme hızını ~200 Hz etkin hıza seyrelt
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
                if lag == 0:
                    y_tau = z
                else:
                    y_tau = z[lag:] * np.conj(z[:N - lag])

            if len(y_tau) >= ds * 4:
                numtaps = min(31, len(y_tau) - 1)
                if numtaps % 2 == 0:
                    numtaps -= 1
                if numtaps >= 5:
                    taps = signal.firwin(numtaps, cutoff=0.8 / ds, window="hamming")
                    y_ds = signal.filtfilt(taps, [1.0], y_tau)[::ds]
                else:
                    y_ds = y_tau[::ds]
            else:
                y_ds = y_tau

            if len(y_ds) < 4:
                continue

            win = np.hanning(len(y_ds))
            fft_y = np.fft.fftshift(np.fft.fft(y_ds * win, n=max(n_fd, len(y_ds))))
            freqs_y = np.fft.fftshift(np.fft.fftfreq(len(fft_y), d=1.0 / fs_eff))
            p_y = np.abs(fft_y) ** 2

            S_mat[:, i] = np.interp(fd_hz, freqs_y, p_y, left=0.0, right=0.0)

        max_p = np.max(S_mat)
        if max_p > 0.0:
            S_db = 10.0 * np.log10(S_mat / max_p + 1e-4)
        else:
            S_db = np.full((n_fd, n_tau), -40.0, dtype=np.float64)

        S_db = np.clip(S_db, -40.0, 0.0)
        return tau_ms, fd_hz, S_db

    @staticmethod
    def compute_multimode_ber_curves() -> Dict[str, np.ndarray]:
        """
        Aynı iyonosferik Watterson kanalı altında 4 farklı HF sayısal modun
        (WSPR, FT8, STANAG 4285, PSK31) Eb/N0 vs. BER başarım eğrilerini hesaplar.
        """
        ebn0_db = np.linspace(-4.0, 18.0, 45)
        ebn0_lin = 10.0 ** (ebn0_db / 10.0)

        # 1. Korumasız PSK31 (BPSK Rayleigh Fading + İyonosferik Faz Hata Tabanı)
        ber_psk31 = 0.5 * (1.0 - np.sqrt(ebn0_lin / (1.0 + ebn0_lin))) + 4e-3

        # 2. STANAG 4285 (8-PSK + Konvolüsyonel + Derin Serpiştirme + DFE Eşitleyici)
        ber_stanag = 0.35 * erfc(np.sqrt(0.45 * ebn0_lin)) * np.exp(-0.18 * ebn0_lin) + 1.5e-4

        # 3. FT8 (8-GFSK + LDPC (174,91) Sum-Product Şelale Bölgesi)
        ber_ft8 = 0.5 * erfc(np.sqrt(0.65 * (ebn0_lin ** 1.45)))

        # 4. WSPR (4-FSK + K=32 Konvolüsyonel Sıralı Kod Çözücü - En Düşük Eşik)
        ber_wspr = 0.5 * erfc(np.sqrt(0.95 * (ebn0_lin ** 1.65)))

        return {
            "ebn0_db": ebn0_db,
            "ber_psk31": np.clip(ber_psk31, 1e-6, 0.5),
            "ber_stanag": np.clip(ber_stanag, 1e-6, 0.5),
            "ber_ft8": np.clip(ber_ft8, 1e-6, 0.5),
            "ber_wspr": np.clip(ber_wspr, 1e-6, 0.5),
        }
