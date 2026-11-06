#!/usr/bin/env python3
"""
Modül 4: Dummy / Benzetimli SDR Donanım Katmanı (DummyLoadHardware)
ve Gerçek Zamanlı NOAA SWPC (services.swpc.noaa.gov/text/) Uzay Havası İstemcisi.
Eski donanım dosyası yerine bu geçici/benzetim motoru kullanılacaktır.
"""

import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from iono_sense.simulation.watterson import WattersonChannelSimulator


BAND_HOPPING_SCHEDULE = [
    {"slot": 0, "sec": ":00", "band": "80m", "freq_mhz": 3.573, "target_layer": "D & E Katmanı"},
    {"slot": 1, "sec": ":15", "band": "40m", "freq_mhz": 7.074, "target_layer": "F1 & NVIS"},
    {"slot": 2, "sec": ":30", "band": "20m", "freq_mhz": 14.074, "target_layer": "F2 Ana DX & TID"},
    {"slot": 3, "sec": ":45", "band": "10m", "freq_mhz": 28.074, "target_layer": "MUF Sınırı & Es"},
]


class DummyLoadHardware:
    """
    SDR Donanımı Benzetimi (Dummy Hardware Controller) ve NOAA SWPC Canlı Veri Çözücüsü.
    RF I/Q yakalamasını sentetik Watterson kanalıyla simüle ederken,
    uzay havası telemetrisini doğrudan NOAA SWPC API/Text servislerinden gerçek zamanlı çeker.
    """

    def __init__(self, fs: float = 12000.0, db_path: str = "iono_telemetry_ta7cae.db"):
        self.fs = fs
        self.db_path = Path(db_path)
        self.simulator = WattersonChannelSimulator(fs=fs)
        self._init_sqlite_db()

    def _init_sqlite_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sounding_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                utc_timestamp TEXT,
                band_mhz REAL,
                callsign TEXT,
                snr_db REAL,
                calibrated_fd_hz REAL,
                doppler_spread_hz REAL,
                pre_fec_ber REAL,
                sfi REAL,
                kp_index REAL,
                bz_nt REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def capture_slot(self, freq_mhz: float) -> Dict:
        """15 saniyelik FT8 tamponunu Watterson simülatörüyle üretir."""
        return self.simulator.simulate_band_frame(band_mhz=freq_mhz, tcxo_common_drift_hz=1.35)

    def log_results(self, freq_mhz: float, station_results: List[Dict], space_weather: Dict) -> None:
        """Kestirilen sonuçları ve uzay havası telemetrisini yerel SQLite veritabanına kaydeder."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        for r in station_results:
            cur.execute(
                """
                INSERT INTO sounding_logs
                (utc_timestamp, band_mhz, callsign, snr_db, calibrated_fd_hz,
                 doppler_spread_hz, pre_fec_ber, sfi, kp_index, bz_nt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    freq_mhz,
                    r["callsign"],
                    r["snr_db"],
                    r["calibrated_mean_fd_hz"],
                    r["doppler_spread_hz"],
                    r["pre_fec_ber"],
                    space_weather["SFI"],
                    space_weather["Kp"],
                    space_weather["Bz_nT"],
                ),
            )
        conn.commit()
        conn.close()

    @staticmethod
    def get_space_weather_snapshot() -> Dict[str, float]:
        """
        NOAA Space Weather Prediction Center (https://services.swpc.noaa.gov/text/)
        üzerinden anlık Güneş Akısı (SFI), Güneş Lekesi (SSN), A ve Kp indekslerini çeker.
        İnternet bağlantısı kesintilerinde güvenli yedekleme değerlerine (fallback) döner.
        """
        # Varsayılan / Fallback değerler
        telemetry = {
            "SFI": 124.0,
            "SN": 109.0,
            "A_index": 7.0,
            "Kp": 2.0,
            "Bz_nT": 5.8,
            "Solar_Wind_kms": 393.6,
            "X_Ray_Flux_Wm2": 1.0e-6,  # C1.0 seviyesi
        }

        # 1. NOAA Solar Indices (SFI & Sunspot Number)
        try:
            url_solar = "https://services.swpc.noaa.gov/text/daily-solar-indices.txt"
            req = urllib.request.Request(url_solar, headers={"User-Agent": "IONO-SENSE-TA7CAE/1.0"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                lines = resp.read().decode("utf-8", errors="ignore").strip().splitlines()
                data_lines = [l for l in lines if l and not l.startswith("#") and not l.startswith(":")]
                if data_lines:
                    latest = data_lines[-1].split()
                    # Format: YYYY MM DD  Radio_Flux(10.7cm)  Sunspot_Number  SESC_Sunspot_Area ...
                    if len(latest) >= 5:
                        telemetry["SFI"] = float(latest[3])
                        telemetry["SN"] = float(latest[4])
        except Exception:
            pass

        # 2. NOAA Geomagnetic Indices (A-Index & Kp-Index)
        try:
            url_geomag = "https://services.swpc.noaa.gov/text/daily-geomagnetic-indices.txt"
            req = urllib.request.Request(url_geomag, headers={"User-Agent": "IONO-SENSE-TA7CAE/1.0"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                lines = resp.read().decode("utf-8", errors="ignore").strip().splitlines()
                data_lines = [l for l in lines if l and not l.startswith("#") and not l.startswith(":")]
                if data_lines:
                    latest = data_lines[-1].split()
                    # Tipik kolonlarda A indeksi ve tahmini Kp indeksleri bulunur
                    if len(latest) >= 4:
                        telemetry["A_index"] = float(latest[3])
                    if len(latest) >= 5 and latest[4] != "-1":
                        telemetry["Kp"] = float(latest[4])
        except Exception:
            pass

        return telemetry
