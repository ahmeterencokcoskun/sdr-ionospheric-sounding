#!/usr/bin/env python3
"""
IONO-SENSE TA7CAE — Ana Komuta ve Yürütücü Konsolu (CLI Orchestrator).
Desteklenen Modlar:
  --mode live      : Canlı NOAA SWPC internet telemetrisi + Dinamik stokastik RF benzetimi
  --mode sim       : Tam stokastik bağımsız simülasyon modu
  --mode benchmark : Birim testler ve tekrarlanabilir tez bulguları için deterministik mod
"""

import argparse
import sys
import numpy as np

from iono_sense.pipeline import IonoSensePipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IONO-SENSE TA7CAE: Çok-Bantlı Fırsatçı İyonosferik Sondaj ve DSP Motoru",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "sim", "benchmark"],
        default="live",
        help=(
            "Sistem Çalışma Modu:\n"
            "  live      : Gerçek NOAA SWPC uzay havası + Dinamik stokastik RF (Varsayılan)\n"
            "  sim       : Tam stokastik simülasyon (Rastgele iyonosfer ve istasyonlar)\n"
            "  benchmark : Deterministik sabit tohumlu test (Tekrarlanabilir akademik sonuç)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Benchmark modunda kullanılacak deterministik rastgele tohum (Varsayılan: 42)",
    )
    parser.add_argument(
        "--iq-file",
        type=str,
        default=None,
        help="Laboratuvar/saha SDR ham I/Q kayıt dosyası (.wav veya binary cf32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    pipeline = IonoSensePipeline(mode=args.mode, fs=12000.0, seed=args.seed, iq_file=args.iq_file)

    meta = pipeline.get_execution_metadata()

    print("=" * 96)
    print(" IONO-SENSE TA7CAE // ÇOK-BANTLI FIRSATÇI İYONOSFERİK SONDAJ VE DSP MOTORU")
    print(" QTH: Kayseri / Talas (38.6875°N, 35.5555°E - KM78sq) | Operatör: TA7CAE")
    print("-" * 96)
    print(f" ÇALIŞMA MODU         : [{meta['mode_title']}]")
    print(f"  • Uzay Havası Verisi: {meta['weather_source']}")
    print(f"  • Alıcı Donanımı     : {meta['rf_hardware']}")
    print(f"  • RF Sinyal Trafiği : {meta['signal_data']}")
    print(f"  • İyonogram Çözücü   : {meta['ionogram_source']}")
    print("=" * 96)

    # 1. Plazma Katman Zayıflama Analizi
    plasma_results = pipeline.run_plasma_attenuation_analysis()
    print("\n[1] PLAZMA KATMAN ETKİLEŞİMİ VE APPLETON-HARTREE ZAYIFLAMA (ALPHA) HATA BÜTÇESİ:")
    print(
        f"{'Taşıyıcı':<10} | {'Katman':<11} | {'Yansıma':<9} | {'Basit 1/f^2':<12} | "
        f"{'Tam O-Modu':<11} | {'Tam X-Modu':<11} | {'Model Hatası'}"
    )
    print("-" * 96)
    for r in plasma_results:
        h_str = f"{r['refl_height_km']:>5.1f} km" if not np.isnan(r.get("refl_height_km", np.nan)) else "  ---   "
        print(
            f"{r['freq_mhz']:>6.3f} MHz | {r['layer']:<11} | {h_str:<9} | "
            f"{r['loss_approx_db']:>9.2f} dB | {r['loss_O_db']:>8.2f} dB | "
            f"{r['loss_X_db']:>8.2f} dB | {r['discrepancy_X_db']:>8.2f} dB"
        )

    # 2. FT8 Mikro-Doppler ve TCXO Analizi (14.074 MHz)
    ft8_res = pipeline.run_ft8_micro_doppler_analysis(freq_mhz=14.074)
    sw = ft8_res["sw_data"]
    print(
        f"\n[2] 14.074 MHz (20m F2 KATMANI) FT8 ÇÖZÜMLEME VE ORTAK-MOD TCXO KALİBRASYONU:\n"
        f"    Uzay Havası: SFI={sw['SFI']:.1f}, Kp={sw['Kp']:.1f}, A-Index={sw['A_index']:.1f}, Bz={sw['Bz_nT']:+.1f} nT\n"
        f"    Gerçek Alıcı TCXO Kayması: +{ft8_res['frame_data']['tcxo_true_hz']:.3f} Hz  -->  "
        f"Kestirilen Ortak-Mod Kayması: +{ft8_res['tcxo_est_hz']:.3f} Hz"
    )
    print(
        f"{'Çağrı İş.':<10} | {'Menzil':<9} | {'SNR':<8} | {'Kalibre fd':<12} | "
        f"{'Doppler Yay.':<13} | {'Katman Hızı vz':<15} | {'Pre-FEC BER'}"
    )
    print("-" * 96)
    for st_m in ft8_res["calibrated_stations"]:
        mean_vz = float(np.mean(st_m["vz_mps"]))
        print(
            f"{st_m['callsign']:<10} | {st_m['distance_km']:>6.0f} km | "
            f"{st_m['snr_db']:>+5.1f} dB | {st_m['calibrated_mean_fd_hz']:>+9.3f} Hz | "
            f"{st_m['doppler_spread_hz']:>10.3f} Hz | {mean_vz:>+11.2f} m/s | "
            f"{st_m['pre_fec_ber']:.4f} ({st_m['bit_errors']}/174)"
        )

    # 3. WSPR İyonosferik Sürüklenme (Drift Rate) ve İvme Analizi (14.0956 MHz)
    wspr_res = pipeline.run_wspr_drift_analysis(freq_mhz=14.0956)
    print(
        f"\n[3] 14.0956 MHz WSPR (4-FSK) ÇÖZÜMLEME VE İYONOSFERİK SÜRÜKLENME (DRIFT) ANALİZİ:\n"
        f"    Gerçek TCXO Sürüklenme Eğimi: +{wspr_res['wspr_slot']['tcxo_drift_slope_hz_min']:.3f} Hz/dk  -->  "
        f"Kestirilen TCXO Eğimi: +{wspr_res['est_drift_slope']:.3f} Hz/dk"
    )
    print(
        f"{'Çağrı İş.':<10} | {'Menzil':<9} | {'SNR':<8} | {'Ölçülen Drift':<15} | "
        f"{'Net İono Drift':<16} | {'Katman İvmesi az'}"
    )
    print("-" * 96)
    for wm in wspr_res["wspr_results"]:
        print(
            f"{wm['callsign']:<10} | {wm['distance_km']:>6.0f} km | "
            f"{wm['snr_db']:>+5.1f} dB | {wm['meas_drift_hz_min']:>+11.2f} Hz/dk | "
            f"{wm['net_iono_drift_hz_min']:>+12.2f} Hz/dk | {wm['layer_accel_mps2']:>+13.4f} m/s²"
        )

    # 4. FMCW İyonosond Taraması (Breit-Tuve Grup İndisi İntegrasyonu)
    fmcw_res = pipeline.run_fmcw_ionosonde_sweep()
    crit = fmcw_res["critical_frequencies"]
    print(
        f"\n[4] DİKEY FMCW İYONOSOND FİZİKSEL İYONOGRAM TARAMASI (1 - 15 MHz):\n"
        f"    Analitik Kritik Frekanslar: foE = {crit['foE_mhz']:.1f} MHz | foF1 = {crit['foF1_mhz']:.1f} MHz | "
        f"foF2 = {crit['foF2_mhz']:.1f} MHz | fxF2 = {crit['fxF2_mhz']:.1f} MHz"
    )

    # 5. Haselgrove 2B Işın İzleme (Ray-Tracing) Yelpazesi
    ray_fan = pipeline.run_ray_tracing_fan(freq_mhz=14.074)
    print(
        f"\n[5] 2B HASELGROVE IŞIN İZLEME (RAY-TRACING) HESABI (14.074 MHz):\n"
        f"{'Kalkış Açısı Δ':<16} | {'Durum':<10} | {'Apex İrtifa':<13} | {'Yer Menzili'}"
    )
    print("-" * 96)
    for ray in ray_fan:
        status_str = "YANSIDI" if ray.reflected else "UZAYA KAÇTI"
        dist_str = f"{ray.ground_hit_km:>6.1f} km" if ray.reflected else "  ---   "
        print(f"Δ = {ray.elevation_deg:>4.1f}°        | {status_str:<10} | {ray.apex_height_km:>7.1f} km   | {dist_str}")

    # 6. Çıktıları Üretme
    pipeline.generate_all_artifacts(
        plasma_results=plasma_results,
        ft8_analysis=ft8_res,
        wspr_analysis=wspr_res,
        ionogram_data=fmcw_res,
        ray_fan=ray_fan,
    )

    print("\n" + "=" * 96)
    print(" [OK] ÇIKTILAR BAŞARIYLA ÜRETİLDİ:")
    print("  -> Grafik Paneli (PNG/PDF) : iono_sense_dashboard.png & iono_sense_dashboard.pdf")
    print("  -> SQLite Telemetri DB     : iono_telemetry_ta7cae.db")
    print("  -> Typst Tez Raporu        : tez_bulgulari.typ")
    print("  -> LaTeX IEEE SİU Tablosu  : tez_bulgulari.tex")
    print("=" * 96)


if __name__ == "__main__":
    main()
