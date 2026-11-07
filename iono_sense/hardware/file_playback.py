#!/usr/bin/env python3
"""SDR I/Q Dosya Oynatıcı Katmanı (WAV IEEE Float32 ve Binary Complex64)."""

from pathlib import Path
from typing import Dict
import numpy as np


class IQFilePlaybackHardware:
    """Kayıtlı SDR I/Q dosyalarını taban bant simülatörü gibi oynatır."""

    def __init__(self, file_path: str, fs: float = 12000.0, format_type: str = "auto"):
        self.path = Path(file_path)
        self.fs = fs
        self.format_type = format_type
        self.iq_buffer = self._load_file()

    def _load_file(self) -> np.ndarray:
        if not self.path.exists():
            return np.zeros(int(self.fs * 15.0), dtype=np.complex64)

        if self.path.suffix.lower() == ".wav" or self.format_type == "wav":
            try:
                from scipy import signal
                from scipy.io import wavfile

                fs_read, data = wavfile.read(self.path)
                max_val = float(np.max(np.abs(data))) if np.max(np.abs(data)) > 0 else 1.0
                if data.ndim == 2 and data.shape[1] >= 2:
                    c_sig = ((data[:, 0] + 1j * data[:, 1]) / max_val).astype(np.complex64)
                else:
                    c_sig = (data.flatten() / max_val).astype(np.complex64)

                if fs_read > 0 and fs_read != self.fs:
                    n_target = int(round(len(c_sig) * float(self.fs) / float(fs_read)))
                    if n_target > 0:
                        c_sig = signal.resample(c_sig, n_target).astype(np.complex64)
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

    def capture_slot(self, freq_mhz: float, start_sec: float = 0.0, duration_sec: float = 15.0) -> Dict:
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
            "source": f"FILE_PLAYBACK ({self.path.name})",
        }
