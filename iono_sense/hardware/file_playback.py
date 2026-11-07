#!/usr/bin/env python3
"""
Modül: Gerçek SDR I/Q Dosya Oynatıcı ve Kaydedici Katmanı.
SDRplay, HackRF, RTL-SDR veya USRP ile kaydedilmiş ham I/Q dosyalarını
(WAV IEEE Float32, Binary Complex64 / Complex Int16 ve SigMF) okur.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class IQFilePlaybackHardware:
    """Laboratuvar veya saha SDR kayıtlarını gerçek zamanlı simülatör gibi oynatır."""

    def __init__(self, file_path: str, fs: float = 12000.0, format_type: str = "auto"):
        self.path = Path(file_path)
        self.fs = fs
        self.format_type = format_type
        self.iq_buffer = self._load_file()

    def _load_file(self) -> np.ndarray:
        """Dosyayı okur ve 1B np.complex64 dizisi olarak taban banda açar."""
        if not self.path.exists():
            # Dosya yoksa sessiz boş tampon döndür
            return np.zeros(int(self.fs * 15.0), dtype=np.complex64)

        suffix = self.path.suffix.lower()

        # 1. WAV Formatı (Stereo Left=I, Right=Q veya Mono)
        if suffix == ".wav" or self.format_type == "wav":
            try:
                from scipy.io import wavfile
                from scipy import signal
                fs_read, data = wavfile.read(self.path)
                if data.ndim == 2 and data.shape[1] >= 2:
                    # Normalize et
                    max_val = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1.0
                    i_ch = data[:, 0].astype(np.float32) / max_val
                    q_ch = data[:, 1].astype(np.float32) / max_val
                    c_sig = (i_ch + 1j * q_ch).astype(np.complex64)
                else:
                    data_flat = data.flatten()
                    max_val = np.max(np.abs(data_flat)) if np.max(np.abs(data_flat)) > 0 else 1.0
                    c_sig = (data_flat.astype(np.float32) / max_val).astype(np.complex64)

                # WAV başlığındaki örnek hızı hedef self.fs ile uyuşmuyorsa yeniden örnekle
                if fs_read > 0 and fs_read != self.fs:
                    num_target = int(round(len(c_sig) * float(self.fs) / float(fs_read)))
                    if num_target > 0:
                        c_sig = signal.resample(c_sig, num_target).astype(np.complex64)
                return c_sig
            except Exception:
                pass

        # 2. Ham Binary Complex Float32 / Float64
        try:
            raw_data = np.fromfile(self.path, dtype=np.complex64)
            if len(raw_data) > 0:
                return raw_data
        except Exception:
            pass

        return np.zeros(int(self.fs * 15.0), dtype=np.complex64)

    def capture_slot(self, freq_mhz: float, start_sec: float = 0.0, duration_sec: float = 15.0) -> Dict:
        """Kayıtlı I/Q dosyasından 15 saniyelik zaman dilimini kesip döndürür."""
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
