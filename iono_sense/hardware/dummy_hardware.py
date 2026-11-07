#!/usr/bin/env python3
"""Simulated SDR Hardware Layer (DummyLoadHardware) and Live NOAA SWPC Space Weather Client."""

import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from iono_sense.simulation.watterson import WattersonChannelSimulator


BAND_HOPPING_SCHEDULE = [
    {"slot": 0, "sec": ":00", "band": "80m", "freq_mhz": 3.573, "target_layer": "D & E Layer"},
    {"slot": 1, "sec": ":15", "band": "40m", "freq_mhz": 7.074, "target_layer": "F1 & NVIS"},
    {"slot": 2, "sec": ":30", "band": "20m", "freq_mhz": 14.074, "target_layer": "F2 Ana DX & TID"},
    {"slot": 3, "sec": ":45", "band": "10m", "freq_mhz": 28.074, "target_layer": "MUF Limit & Es"},
]


def _get_field(obj: Any, name: str, default: float = 0.0) -> float:
    if isinstance(obj, dict):
        return float(obj.get(name, default))
    return float(getattr(obj, name, default))


class DummyLoadHardware:
    """SDR hardware simulation, SQLite telemetry logger, and live NOAA SWPC data parser."""

    def __init__(self, fs: float = 12000.0, db_path: str = "iono_telemetry_ta7cae.db"):
        self.fs = fs
        self.db_path = Path(db_path)
        self.simulator = WattersonChannelSimulator(fs=fs)
        self._init_sqlite_db()

    def _init_sqlite_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ionospheric_soundings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    utc_timestamp TEXT,
                    f_min_mhz REAL,
                    f_max_mhz REAL,
                    foF2_mhz REAL,
                    hmF2_km REAL,
                    NmF2_m3 REAL,
                    subpeak_tec_tecu REAL,
                    inversion_method TEXT,
                    residual_rms_km REAL,
                    status TEXT
                )
                """
            )
            cur.execute("PRAGMA table_info(ionospheric_soundings)")
            existing_cols = {row[1] for row in cur.fetchall()}
            expected_cols = {
                "utc_timestamp": "TEXT",
                "f_min_mhz": "REAL",
                "f_max_mhz": "REAL",
                "foF2_mhz": "REAL",
                "hmF2_km": "REAL",
                "NmF2_m3": "REAL",
                "subpeak_tec_tecu": "REAL",
                "inversion_method": "TEXT",
                "residual_rms_km": "REAL",
                "status": "TEXT",
            }
            for col_name, col_type in expected_cols.items():
                if col_name not in existing_cols:
                    cur.execute(f"ALTER TABLE ionospheric_soundings ADD COLUMN {col_name} {col_type}")

    def capture_slot(self, freq_mhz: float) -> Dict:
        return self.simulator.simulate_band_frame(band_mhz=freq_mhz, tcxo_common_drift_hz=1.35)

    def log_results(self, freq_mhz: float, station_results: List[Dict], space_weather: Dict) -> None:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
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

    def log_inversion(
        self,
        inversion_result: Any,
        f_min_mhz: float = 1.0,
        f_max_mhz: float = 15.0,
        method: str = "jackson",
        status: str = "CONVERGED",
    ) -> None:
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ionospheric_soundings
                (utc_timestamp, f_min_mhz, f_max_mhz, foF2_mhz, hmF2_km,
                 NmF2_m3, subpeak_tec_tecu, inversion_method, residual_rms_km, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    float(f_min_mhz),
                    float(f_max_mhz),
                    _get_field(inversion_result, "foF2_mhz"),
                    _get_field(inversion_result, "hmF2_km"),
                    _get_field(inversion_result, "NmF2_m3"),
                    _get_field(inversion_result, "subpeak_tec_tecu"),
                    str(method),
                    _get_field(inversion_result, "residual_rms_km"),
                    str(status),
                ),
            )

    @staticmethod
    def _fetch_swpc_last_row(url: str) -> List[str]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "IONO-SENSE-TA7CAE/1.0"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                lines = resp.read().decode("utf-8", errors="ignore").strip().splitlines()
                data = [ln for ln in lines if ln and not ln.startswith(("#", ":"))]
                return data[-1].split() if data else []
        except Exception:
            return []

    @classmethod
    def get_space_weather_snapshot(cls) -> Dict[str, float]:
        telemetry = {
            "SFI": 124.0,
            "SN": 109.0,
            "A_index": 7.0,
            "Kp": 2.0,
            "Bz_nT": 5.8,
            "Solar_Wind_kms": 393.6,
            "X_Ray_Flux_Wm2": 1.0e-6,
        }

        solar = cls._fetch_swpc_last_row("https://services.swpc.noaa.gov/text/daily-solar-indices.txt")
        if len(solar) >= 5:
            try:
                sfi, sn = float(solar[3]), float(solar[4])
                if sfi > 0.0:
                    telemetry["SFI"] = sfi
                if sn >= 0.0:
                    telemetry["SN"] = sn
            except ValueError:
                pass

        geomag = cls._fetch_swpc_last_row("https://services.swpc.noaa.gov/text/daily-geomagnetic-indices.txt")
        if len(geomag) >= 4:
            try:
                a_val = float(geomag[3])
                if a_val >= 0.0:
                    telemetry["A_index"] = a_val
                if len(geomag) >= 5 and float(geomag[4]) >= 0.0:
                    telemetry["Kp"] = float(geomag[4])
            except ValueError:
                pass

        return telemetry
