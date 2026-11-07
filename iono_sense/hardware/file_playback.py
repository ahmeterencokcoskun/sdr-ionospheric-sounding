#!/usr/bin/env python3
"""SDR I/Q ve Ses (.wav / .cf32) Dosya Oynatici Katmani (Hilbert Analitik Sinyal Destekli)."""

from pathlib import Path
from typing import Dict
import numpy as np
from scipy import signal
from scipy.io import wavfile


class IQFilePlaybackHardware:
    """Processes real SDR I/Q (.wav / .cf32) and SDR++/WSJT-X audio recordings."""

    def __init__(
        self, file_path: str, fs: float = 12000.0, format_type: str = "auto"
    ):
        self.path = Path(file_path)
        self.fs = fs
        self.format_type = format_type
        self.iq_buffer = self._load_file()

    def _load_file(self) -> np.ndarray:
        if not self.path.exists():
            return np.zeros(int(self.fs * 15.0), dtype=np.complex64)

        if self.path.suffix.lower() == ".wav" or self.format_type == "wav":
            try:
                fs_read, data = wavfile.read(self.path)
                data = data.astype(np.float64)
                max_val = (
                    float(np.max(np.abs(data)))
                    if np.max(np.abs(data)) > 0
                    else 1.0
                )
                data /= max_val

                # 2-channel real I/Q recording (Left != Right)
                if (
                    data.ndim == 2
                    and data.shape[1] >= 2
                    and not np.allclose(data[:, 0], data[:, 1], atol=1e-4)
                ):
                    c_sig = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
                else:
                    # Mono or Stereo USB audio -> Analytic I/Q signal via Hilbert Transform
                    mono = data[:, 0] if data.ndim == 2 else data.flatten()
                    c_sig = signal.hilbert(mono).astype(np.complex64)

                if fs_read > 0 and fs_read != self.fs:
                    n_target = int(
                        round(len(c_sig) * float(self.fs) / float(fs_read))
                    )
                    if n_target > 0:
                        c_sig = signal.resample_poly(
                            c_sig, int(self.fs), int(fs_read)
                        ).astype(np.complex64)
                return c_sig
            except Exception:
                pass

        try:
            raw_data = np.fromfile(self.path, dtype=np.complex64)
            if len(raw_data) > 0:
                return raw_data
        except Exception:
            pass

        return np.zeros(int(self.fs * 15.0), dtype=np.complex64)

    def capture_slot(
        self, freq_mhz: float, start_sec: float = 0.0, duration_sec: float = 15.0
    ) -> Dict:
        n_samples = int(duration_sec * self.fs)
        start_idx = int(start_sec * self.fs)

        if start_idx + n_samples <= len(self.iq_buffer):
            segment = self.iq_buffer[start_idx : start_idx + n_samples]
        else:
            segment = np.zeros(n_samples, dtype=np.complex64)
            avail = len(self.iq_buffer) - start_idx
            if avail > 0:
                segment[:avail] = self.iq_buffer[start_idx:]

        return {
            "band_mhz": freq_mhz,
            "fs": self.fs,
            "tcxo_true_hz": 0.0,
            "iq_buffer": segment,
            "ground_truth": [],
            "source": f"REAL SDR RECORDING ({self.path.name})",
        }