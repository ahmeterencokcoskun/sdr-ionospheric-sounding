#!/usr/bin/env python3
"""
Uçtan Uca İyonosferik Algılama ve DSP Orkestrasyon Hattı (Pipeline).
Desteklenen Modlar:
  - 'live'     : Gerçek zamanlı NOAA SWPC telemetrisi + Dinamik stokastik RF benzetimi
  - 'sim'      : Tamamen rastgele stokastik uzay havası ve gökyüzü koşulları
  - 'benchmark': Belirlenmiş sabit tohum (seed) ile deterministik tekrarlanabilir test modu
"""

from typing import Dict, List, Optional
import numpy as np

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine
from iono_sense.simulation.watterson import WattersonChannelSimulator
from iono_sense.simulation.fmcw_sounder import FMCWIonosondeSimulator
from iono_sense.dsp.core import IonosphericDSPCore
from iono_sense.hardware.dummy_hardware import DummyLoadHardware
from iono_sense.reporting.dashboard import AcademicDashboardPlotter
from iono_sense.reporting.document_exporter import AcademicReportExporter


class IonoSensePipeline:
    """Çok-Bantlı Fırsatçı İyonosferik Sondaj ve Analiz Hattı."""

    def __init__(self, mode: str = "live", fs: float = 12000.0, seed: Optional[int] = None):
        self.mode = mode.lower()
        self.fs = fs
        self.seed = seed if self.mode == "benchmark" else None

        # 1. Alt Sistemlerin Başlatılması
        self.plasma_day = IonosphericPlasmaEngine(is_daytime=True)
        self.plasma_night = IonosphericPlasmaEngine(is_daytime=False)
        self.sdr_controller = DummyLoadHardware(fs=self.fs)
        self.dsp_core = IonosphericDSPCore(fs=self.fs)
        self.watterson_sim = WattersonChannelSimulator(fs=self.fs, rng_seed=self.seed)
        self.fmcw_sim = FMCWIonosondeSimulator(plasma_engine=self.plasma_day)
        self.plotter = AcademicDashboardPlotter()
        self.exporter = AcademicReportExporter()

    def get_execution_metadata(self) -> Dict[str, str]:
        """Konsol ve raporlar için aktif veri kaynaklarını açıklar."""
        if self.mode == "live":
            return {
                "mode_title": "CANLI / LIVE HİBRİT",
                "weather_source": "GERÇEK ZAMANLI (NOAA SWPC API - services.swpc.noaa.gov)",
                "rf_hardware": "DUMMY LOAD / BENZETİMLİ SDR (RSPdx Emülatörü)",
                "signal_data": "DİNAMİK STOKASTİK (Rastgele Çok-Yollu HF Trafiği)",
                "ionogram_source": "FİZİKSEL MODEL (Breit-Tuve Grup İndisi + Range-FFT AWGN)",
            }
        elif self.mode == "sim":
            return {
                "mode_title": "TAM STOKASTİK SİMÜLASYON",
                "weather_source": "STOKASTİK SİMÜLASYON (Dinamik Güneş/Manyetik Alan)",
                "rf_hardware": "DUMMY LOAD / BENZETİMLİ SDR",
                "signal_data": "STOKASTİK SENTETİK (Rastgele İstasyonlar & Fading)",
                "ionogram_source": "FİZİKSEL MODEL (Breit-Tuve Grup İndisi + Range-FFT AWGN)",
            }
        else:  # benchmark
            return {
                "mode_title": "STATİK BENCHMARK TESTİ (TEKRARLANABİLİR)",
                "weather_source": "SABİT REFERANS VERİ (N0NBH Standart Günü)",
                "rf_hardware": "DUMMY LOAD / BENZETİMLİ SDR",
                "signal_data": f"SABİT DETERMINİSTİK KANAL (Seed: {self.seed})",
                "ionogram_source": "FİZİKSEL MODEL (Deterministik Breit-Tuve)",
            }

    def run_plasma_attenuation_analysis(self, freqs: Optional[List[float]] = None) -> List[Dict]:
        """Appleton-Hartree ve Sen-Wyller katman zayıflama hata bütçesini hesaplar."""
        if freqs is None:
            freqs = [1.840e6, 3.573e6, 7.074e6, 10.136e6, 14.074e6, 21.074e6, 28.074e6, 50.313e6]
        return [self.plasma_day.compute_attenuation_models(f, elevation_deg=18.0) for f in freqs]

    def run_ft8_micro_doppler_analysis(self, freq_mhz: float = 14.074) -> Dict:
        """FT8 çerçevesini yakalar, TCXO ortak modunu kalibre eder ve mikro-Doppler çıkarır."""
        if self.mode == "live":
            sw_data = self.sdr_controller.get_space_weather_snapshot()
        elif self.mode == "sim":
            rng = np.random.default_rng()
            sw_data = {
                "SFI": float(rng.uniform(90.0, 180.0)),
                "SN": float(rng.uniform(60.0, 150.0)),
                "A_index": float(rng.uniform(3.0, 15.0)),
                "Kp": float(rng.uniform(1.0, 4.5)),
                "Bz_nT": float(rng.uniform(-4.0, 8.0)),
                "Solar_Wind_kms": float(rng.uniform(350.0, 520.0)),
                "X_Ray_Flux_Wm2": 1.2e-6,
            }
        else:  # benchmark
            sw_data = {
                "SFI": 124.0,
                "SN": 109.0,
                "A_index": 7.0,
                "Kp": 2.0,
                "Bz_nT": 5.8,
                "Solar_Wind_kms": 393.6,
                "X_Ray_Flux_Wm2": 1.0e-6,
            }

        # RF tamponunu Watterson simülatöründen al
        frame_20m = self.watterson_sim.simulate_band_frame(band_mhz=freq_mhz)

        raw_metrics = [
            self.dsp_core.estimate_station_parameters(frame_20m["iq_buffer"], gt, f_carrier_mhz=freq_mhz)
            for gt in frame_20m["ground_truth"]
        ]
        tcxo_est_hz, calibrated_stations = self.dsp_core.calibrate_and_extract_doppler(raw_metrics)
        self.sdr_controller.log_results(freq_mhz, calibrated_stations, sw_data)

        return {
            "sw_data": sw_data,
            "frame_data": frame_20m,
            "tcxo_est_hz": tcxo_est_hz,
            "calibrated_stations": calibrated_stations,
        }

    def run_wspr_drift_analysis(self, freq_mhz: float = 14.0956) -> Dict:
        """2 dakikalık WSPR zaman diliminde frekans sürüklenmesi ve katman ivmesini analiz eder."""
        wspr_slot = self.watterson_sim.simulate_wspr_slot(band_mhz=freq_mhz)
        est_drift_slope, wspr_results = self.dsp_core.analyze_wspr_signals(wspr_slot, f_carrier_mhz=freq_mhz)
        return {
            "wspr_slot": wspr_slot,
            "est_drift_slope": est_drift_slope,
            "wspr_results": wspr_results,
        }

    def run_fmcw_ionosonde_sweep(self) -> Dict:
        """FMCW LFM Chirp iyonosond sanal irtifa h'(f) profilini üretir."""
        return self.fmcw_sim.synthesize_ionogram_profile()

    def generate_all_artifacts(
        self,
        plasma_results: List[Dict],
        ft8_analysis: Dict,
        wspr_analysis: Dict,
        png_path: str = "iono_sense_dashboard.png",
        pdf_path: str = "iono_sense_dashboard.pdf",
        typst_path: str = "tez_bulgulari.typ",
        latex_path: str = "tez_bulgulari.tex",
    ) -> None:
        """Dashboard figürlerini ve Typst/LaTeX belgelerini diske üretir."""
        self.plotter.plot_dashboard(
            plasma_day=self.plasma_day,
            plasma_night=self.plasma_night,
            plasma_results=plasma_results,
            frame_20m=ft8_analysis["frame_data"],
            calibrated_stations=ft8_analysis["calibrated_stations"],
            dsp_core=self.dsp_core,
            sw_data=ft8_analysis["sw_data"],
            save_png=png_path,
            save_pdf=pdf_path,
        )

        self.exporter.export_typst(
            plasma_table=plasma_results,
            station_metrics=ft8_analysis["calibrated_stations"],
            tcxo_est=ft8_analysis["tcxo_est_hz"],
            wspr_metrics=wspr_analysis["wspr_results"],
            filepath=typst_path,
        )

        self.exporter.export_latex(
            plasma_table=plasma_results,
            filepath=latex_path,
        )
