#!/usr/bin/env python3
"""
IONO-SENSE TA7CAE — Ana Yürütücü (Orchestrator).
Nesne yönelimli mimaride tek bir pipeline nesnesi üzerinden sistem simülasyonunu başlatır.
"""

import numpy as np
from iono_sense.pipeline import IonoSensePipeline


def main() -> None:
    print("=" * 92)
    print(" IONO-SENSE TA7CAE // ÇOK-BANTLI FIRSATÇI İYONOSFERİK SONDAJ VE DSP MOTORU")
    print(" Konum: Talas, Kayseri (38.72°N, 35.48°E - KM78) | Alıcı: SDRplay RSPdx (14-bit)")
    print("=" * 92)

    # 1. Pipeline Başlatma
    pipeline = IonoSensePipeline(fs=12000.0)

    # 2. Plazma Katman Zayıflama Analizi
    plasma_results = pipeline.run_plasma_attenuation_analysis()
    print("\n[1] PLAZMA KATMAN ETKİLEŞİMİ VE APPLETON-HARTREE ZAYIFLAMA (ALPHA) HATA BÜTÇESİ:")
    print(
        f"{'Taşıyıcı':<10} | {'Katman':<11} | {'Yansıma':<9} | {'Basit 1/f^2':<12} | "
        f"{'Tam O-Modu':<11} | {'Tam X-Modu':<11} | {'Model Hatası'}"
    )
    print("-" * 92)
    for r in plasma_results:
        h_str = f"{r['refl_height_km']:>5.1f} km" if not np.isnan(r.get("refl_height_km", np.nan)) else "  ---   "
        print(
            f"{r['freq_mhz']:>6.3f} MHz | {r['layer']:<11} | {h_str:<9} | "
            f"{r['loss_approx_db']:>9.2f} dB | {r['loss_O_db']:>8.2f} dB | "
            f"{r['loss_X_db']:>8.2f} dB | {r['discrepancy_X_db']:>8.2f} dB"
        )

    # 3. FT8 Mikro-Doppler ve TCXO Analizi (14.074 MHz)
    ft8_res = pipeline.run_ft8_micro_doppler_analysis(freq_mhz=14.074)
    print(
        f"\n[2] 14.074 MHz (20m F2 KATMANI) FT8 ÇÖZÜMLEME VE ORTAK-MOD TCXO KALİBRASYONU:\n"
        f"    Gerçek RSPdx TCXO Kayması: +{ft8_res['frame_data']['tcxo_true_hz']:.3f} Hz  -->  "
        f"Kestirilen Ortak-Mod Kayması: +{ft8_res['tcxo_est_hz']:.3f} Hz"
    )
    print(
        f"{'Çağrı İş.':<10} | {'Menzil':<9} | {'SNR':<8} | {'Kalibre fd':<12} | "
        f"{'Doppler Yay.':<13} | {'Katman Hızı vz':<15} | {'Pre-FEC BER'}"
    )
    print("-" * 92)
    for st_m in ft8_res["calibrated_stations"]:
        mean_vz = float(np.mean(st_m["vz_mps"]))
        print(
            f"{st_m['callsign']:<10} | {st_m['distance_km']:>6.0f} km | "
            f"{st_m['snr_db']:>+5.1f} dB | {st_m['calibrated_mean_fd_hz']:>+9.3f} Hz | "
            f"{st_m['doppler_spread_hz']:>10.3f} Hz | {mean_vz:>+11.2f} m/s | "
            f"{st_m['pre_fec_ber']:.4f} ({st_m['bit_errors']}/174)"
        )

    # 4. WSPR İyonosferik Sürüklenme (Drift Rate) ve İvme Analizi (14.0956 MHz)
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
    print("-" * 92)
    for wm in wspr_res["wspr_results"]:
        print(
            f"{wm['callsign']:<10} | {wm['distance_km']:>6.0f} km | "
            f"{wm['snr_db']:>+5.1f} dB | {wm['meas_drift_hz_min']:>+11.2f} Hz/dk | "
            f"{wm['net_iono_drift_hz_min']:>+12.2f} Hz/dk | {wm['layer_accel_mps2']:>+13.4f} m/s²"
        )

    # 5. FMCW İyonosond Taraması
    fmcw_res = pipeline.run_fmcw_ionosonde_sweep()
    crit = fmcw_res["critical_frequencies"]
    print(
        f"\n[4] FMCW İYONOSOND SENTETİK İYONOGRAM TARAMASI (1 - 16 MHz):\n"
        f"    Kritik Frekanslar: foE = {crit['foE_mhz']:.1f} MHz | foF1 = {crit['foF1_mhz']:.1f} MHz | "
        f"foF2 = {crit['foF2_mhz']:.1f} MHz | fxF2 = {crit['fxF2_mhz']:.1f} MHz"
    )

    # 6. Akademik Çıktıları Üretme (Görsel ve Dokümanlar)
    pipeline.generate_all_artifacts(
        plasma_results=plasma_results,
        ft8_analysis=ft8_res,
        wspr_analysis=wspr_res,
    )

    print("\n" + "=" * 92)
    print(" [OK] ÇIKTILAR BAŞARIYLA ÜRETİLDİ:")
    print("  -> Grafik Paneli (PNG/PDF) : iono_sense_dashboard.png & iono_sense_dashboard.pdf")
    print("  -> SQLite Telemetri DB     : iono_telemetry_ta7cae.db")
    print("  -> Typst Tez Raporu        : tez_bulgulari.typ")
    print("  -> LaTeX IEEE SİU Tablosu  : tez_bulgulari.tex")
    print("=" * 92)


if __name__ == "__main__":
    main()
