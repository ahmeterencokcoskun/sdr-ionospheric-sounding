#!/usr/bin/env python3
"""
IONO-SENSE TA7CAE — Ana Mühendislik Çalıştırıcısı, Görselleştirici ve
Otomatik Typst (.typ) / LaTeX (.tex) Akademik Rapor Üreticisi.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasma_physics import IonosphericPlasmaEngine
from dsp_core import IonosphericDSPCore
from sdr_hardware import RSPdxReceiverController


def export_academic_reports(plasma_table: list, station_metrics: list, tcxo_est: float) -> None:
    """Elde edilen bulguları Typst (.typ) ve LaTeX (.tex) formatında diske yazar."""
    # 1. Typst Raporu
    typst_rows = "\n".join(
        f"  [{r['freq_mhz']:.3f} MHz], [{r['layer']}], "
        f"[{r['refl_height_km']:.1f} km], [{r['loss_approx_db']:.2f} dB], "
        f"[{r['loss_O_db']:.2f} dB], [{r['loss_X_db']:.2f} dB], [{r['discrepancy_X_db']:.2f} dB],"
        for r in plasma_table
        if not r["penetrates"]
    )

    typst_content = (
        "// Otomatik Üretilen Akademik Bulgular Raporu (Typst)\n"
        "// Yazar: Ahmet Eren Çokcoşkun (TA7CAE) - Nuh Naci Yazgan Üniversitesi EEE\n\n"
        '#set page(paper: "a4", margin: 2cm)\n'
        '#set text(font: "New Computer Modern", size: 10.5pt, lang: "tr")\n\n'
        "= Çok-Bantlı Fırsatçı İyonosferik Algılama ve Mikro-Doppler Kestirim Bulguları\n"
        '*Operatör:* Ahmet Eren Çokcoşkun (`TA7CAE`) | *Konum:* Kayseri (`KM78`, $38.72^circle "N", 35.48^circle "E"$)\n\n'
        "== 1. Appleton-Hartree ve Sapmasız Yaklaşım Zayıflama ($alpha$) Hata Bütçesi\n"
        'Kayseri jeomanyetik alanı ($B_0 = 48.2 space mu"T"$, $f_H approx 1.35 "MHz"$) altında elde edilen iki yönlü sönümleme karşılaştırması:\n\n'
        "#table(\n"
        "  columns: (auto, auto, auto, auto, auto, auto, auto),\n"
        "  align: center,\n"
        "  [*Frekans*], [*Katman*], [*İrtifa*], [*Basit $1/f^2$*], [*Tam O-Modu*], [*Tam X-Modu*], [*X-Modu Farkı*],\n"
        f"{typst_rows}\n"
        ")\n\n"
        "== 2. Ortak-Mod TCXO Kalibrasyonu ve Alt-Paketsel Mikro-Doppler\n"
        f"- *Kestirilen RSPdx TCXO Ortak-Mod Kayması:* `{tcxo_est:+.3f} Hz` (Gerçek: `+1.350 Hz`)\n"
    )
    with open("tez_bulgulari.typ", "w", encoding="utf-8") as f:
        f.write(typst_content)

    # 2. LaTeX Raporu
    latex_rows = "\n".join(
        f"{r['freq_mhz']:.3f} & {r['layer']} & {r['refl_height_km']:.1f} & "
        f"{r['loss_approx_db']:.2f} & {r['loss_O_db']:.2f} & {r['loss_X_db']:.2f} & "
        f"{r['discrepancy_X_db']:.2f} \\\\"
        for r in plasma_table
        if not r["penetrates"]
    )

    latex_lines = [
        "% Otomatik Üretilen IEEE SİU / Bitirme Tezi Tablosu (LaTeX)",
        "% Yazar: Ahmet Eren Çokcoşkun (TA7CAE)",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Kayseri Orta Enlem İyonosferinde Yaklaşım vs. Tam Magneto-İyonik Zayıflama Analizi}",
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        r"\textbf{Frekans (MHz)} & \textbf{Katman} & \textbf{$h_{\mathrm{refl}}$ (km)} & \textbf{Basit $1/f^2$ (dB)} & \textbf{O-Modu (dB)} & \textbf{X-Modu (dB)} & \textbf{Hata Payı (dB)} \\",
        r"\hline",
        latex_rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open("tez_bulgulari.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")


def main() -> None:
    print("=" * 92)
    print(" IONO-SENSE TA7CAE // ÇOK-BANTLI FIRSATÇI İYONOSFERİK SONDAJ VE DSP MOTORU")
    print(" Konum: Talas, Kayseri (38.72°N, 35.48°E - KM78) | Alıcı: SDRplay RSPdx (14-bit)")
    print("=" * 92)

    # 1. Plazma Fiziği ve Katman Analizi
    plasma_day = IonosphericPlasmaEngine(is_daytime=True)
    plasma_night = IonosphericPlasmaEngine(is_daytime=False)

    test_freqs = [1.840e6, 3.573e6, 7.074e6, 10.136e6, 14.074e6, 21.074e6, 28.074e6, 50.313e6]
    plasma_results = [plasma_day.compute_attenuation_models(f, elevation_deg=18.0) for f in test_freqs]

    print("\n[1] PLAZMA KATMAN ETKİLEŞİMİ VE APPLETON-HARTREE ZAYIFLAMA (ALPHA) HATA BÜTÇESİ:")
    print(
        f"{'Taşıyıcı':<10} | {'Katman':<11} | {'Yansıma':<9} | {'Basit 1/f^2':<12} | "
        f"{'Tam O-Modu':<11} | {'Tam X-Modu':<11} | {'Model Hatası'}"
    )
    print("-" * 92)
    for r in plasma_results:
        h_str = f"{r['refl_height_km']:>5.1f} km" if not np.isnan(r["refl_height_km"]) else "  ---   "
        print(
            f"{r['freq_mhz']:>6.3f} MHz | {r['layer']:<11} | {h_str:<9} | "
            f"{r['loss_approx_db']:>9.2f} dB | {r['loss_O_db']:>8.2f} dB | "
            f"{r['loss_X_db']:>8.2f} dB | {r['discrepancy_X_db']:>8.2f} dB"
        )

    # 2. SDR / Watterson I/Q Yakalama ve DSP Mikro-Doppler Çözümleme (20m Bandı - 14.074 MHz)
    sdr = RSPdxReceiverController(fs=12000.0)
    dsp = IonosphericDSPCore(fs=12000.0)
    sw_data = sdr.get_space_weather_snapshot()

    frame_20m = sdr.capture_slot(freq_mhz=14.074)
    raw_station_metrics = [
        dsp.estimate_station_parameters(frame_20m["iq_buffer"], gt, f_carrier_mhz=14.074)
        for gt in frame_20m["ground_truth"]
    ]
    tcxo_est_hz, calibrated_stations = dsp.calibrate_and_extract_doppler(raw_station_metrics)
    sdr.log_results(14.074, calibrated_stations, sw_data)

    print(
        f"\n[2] 14.074 MHz (20m F2 KATMANI) FT8 ÇÖZÜMLEME VE ORTAK-MOD TCXO KALİBRASYONU:\n"
        f"    Gerçek RSPdx TCXO Kayması: +{frame_20m['tcxo_true_hz']:.3f} Hz  -->  "
        f"Kestirilen Ortak-Mod Kayması: +{tcxo_est_hz:.3f} Hz"
    )
    print(
        f"{'Çağrı İş.':<10} | {'Menzil':<9} | {'SNR':<8} | {'Kalibre fd':<12} | "
        f"{'Doppler Yay.':<13} | {'Katman Hızı vz':<15} | {'Pre-FEC BER'}"
    )
    print("-" * 92)
    for st_m in calibrated_stations:
        mean_vz = float(np.mean(st_m["vz_mps"]))
        print(
            f"{st_m['callsign']:<10} | {st_m['distance_km']:>6.0f} km | "
            f"{st_m['snr_db']:>+5.1f} dB | {st_m['calibrated_mean_fd_hz']:>+9.3f} Hz | "
            f"{st_m['doppler_spread_hz']:>10.3f} Hz | {mean_vz:>+11.2f} m/s | "
            f"{st_m['pre_fec_ber']:.4f} ({st_m['bit_errors']}/174)"
        )

    # 3. 6 Panelli Akademik Dashboard Çizimi
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=160)
    fig.suptitle(
        "IONO-SENSE TA7CAE: Çok-Bantlı Fırsatçı İyonosferik Sondaj ve Mikro-Doppler Analiz Paneli\n"
        f"QTH: Kayseri (38.72°N, 35.48°E) | SFI: {sw_data['SFI']:.0f} | Kp: {sw_data['Kp']:.0f} | "
        f"IMF Bz: +{sw_data['Bz_nT']} nT | X-Ray: C1.0",
        fontsize=13,
        fontweight="bold",
        color="#00e5ff",
    )

    # Panel 1: Gündüz vs Gece Chapman Elektron Yoğunluğu ve Plazma Frekansı
    ax1 = axes[0, 0]
    ax1.plot(plasma_day.fp_hz / 1e6, plasma_day.h_km, color="#00ff9d", lw=2.2, label="Gündüz (D, E, F1, F2)")
    ax1.plot(plasma_night.fp_hz / 1e6, plasma_night.h_km, color="#ff9100", lw=2.0, ls="--", label="Gece (Birleşik F)")
    ax1.set_title("1. İyonosferik Plazma Frekansı Profili fp(h)", fontsize=10.5, color="#e0e0e0")
    ax1.set_xlabel("Plazma Kritik Frekansı fp (MHz)")
    ax1.set_ylabel("İrtifa h (km)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel 2: Appleton-Hartree (O/X Modu) vs Basit Yaklaşım Hata Bütçesi
    ax2 = axes[0, 1]
    valid_r = [r for r in plasma_results if not r["penetrates"]]
    f_labels = [f"{r['freq_mhz']:.1f}" for r in valid_r]
    x_pos = np.arange(len(valid_r))
    w = 0.25
    ax2.bar(x_pos - w, [r["loss_approx_db"] for r in valid_r], width=w, color="#2979ff", label="Basit 1/f² Yaklaşımı")
    ax2.bar(x_pos, [r["loss_sen_wyller_db"] for r in valid_r], width=w, color="#00e676", label="Tam O-Modu (Sen-Wyller)")
    ax2.bar(x_pos + w, [r["loss_X_db"] for r in valid_r], width=w, color="#ff1744", label="Tam X-Modu (Jiro-Rezonans)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(f_labels)
    ax2.set_yscale("log")
    ax2.set_title("2. Zayıflama (Alpha) Model Karşılaştırması (dB)", fontsize=10.5, color="#e0e0e0")
    ax2.set_xlabel("Taşıyıcı Frekans (MHz)")
    ax2.set_ylabel("İki Yönlü Emilim Kaybı (dB - Log)")
    ax2.grid(True, alpha=0.25, which="both")
    ax2.legend(fontsize=8)

    # Panel 3: 14.074 MHz FT8 Spektrogramı (Şelale)
    ax3 = axes[0, 2]
    Sxx, f_spec, t_spec = matplotlib.mlab.specgram(
        frame_20m["iq_buffer"], NFFT=2048, Fs=12000.0, noverlap=1536
    )
    mask_f = (f_spec >= 400.0) & (f_spec <= 2700.0)
    im3 = ax3.pcolormesh(
        f_spec[mask_f], t_spec, 10.0 * np.log10(np.abs(Sxx[mask_f, :]).T + 1e-12),
        shading="auto", cmap="inferno"
    )
    ax3.set_title("3. 14.074 MHz FT8 Taban Bant Spektrogramı", fontsize=10.5, color="#e0e0e0")
    ax3.set_xlabel("Taban Bant Frekansı (Hz)")
    ax3.set_ylabel("Zaman (s)")
    fig.colorbar(im3, ax=ax3, label="Güç (dB)")

    # Panel 4: Alt-Paketsel Anlık Mikro-Doppler İzleyici (TID İmzası)
    ax4 = axes[1, 0]
    target_st = calibrated_stations[0]
    t_sub = target_st["t_sig"][8000:-8000]
    ax4.plot(
        t_sub, target_st["inst_doppler_true"][8000:-8000],
        color="#ffffff", lw=2.0, ls="--", label="Gerçek İyonosferik TID Doppler"
    )
    ax4.plot(
        t_sub, target_st["fd_est"][8000:-8000],
        color="#00e5ff", lw=1.8, label=f"Kestirilen fd(t) [{target_st['callsign']}]"
    )
    ax4.set_title("4. Alt-Paketsel Anlık Mikro-Doppler Yörüngesi fd(t)", fontsize=10.5, color="#e0e0e0")
    ax4.set_xlabel("Paket İçi Zaman (s)")
    ax4.set_ylabel("Anlık Doppler Kayması (Hz)")
    ax4.grid(True, alpha=0.25)
    ax4.legend(fontsize=8.5)

    # Panel 5: 2B Gecikme-Doppler Kanal Saçılma Fonksiyonu S(tau, fd)
    ax5 = axes[1, 1]
    tau_ms, fd_grid, S_db = dsp.compute_scattering_function(target_st["z_smooth"])
    im5 = ax5.contourf(tau_ms, fd_grid, S_db, levels=24, cmap="viridis")
    ax5.set_title("5. Kanal Saçılma Fonksiyonu S(τ, fd) [1F2 & 2F2]", fontsize=10.5, color="#e0e0e0")
    ax5.set_xlabel("Çok-Yollu Grup Gecikmesi τ (ms)")
    ax5.set_ylabel("Doppler Frekansı fd (Hz)")
    fig.colorbar(im5, ax=ax5, label="Normalize Güç (dB)")

    # Panel 6: Çoklu HF Sayısal Mod BER vs Eb/N0 Başarım Eğrileri
    ax6 = axes[1, 2]
    ber_data = dsp.compute_multimode_ber_curves()
    ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_wspr"], color="#00e676", lw=2.2, label="WSPR (4-FSK + K=32 Conv)")
    ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_ft8"], color="#00e5ff", lw=2.2, label="FT8 (8-GFSK + LDPC 174,91)")
    ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_stanag"], color="#ffab00", lw=2.0, ls="-.", label="STANAG 4285 (8-PSK + DFE)")
    ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_psk31"], color="#ff1744", lw=2.0, ls=":", label="PSK31 (Korumasız BPSK)")
    emp_snrs = [s["snr_db"] + 6.0 for s in calibrated_stations]
    emp_bers = [max(s["pre_fec_ber"], 1e-4) for s in calibrated_stations]
    ax6.scatter(emp_snrs, emp_bers, color="#ffffff", edgecolor="#ff1744", s=55, zorder=5, label="Ölçülen Gökyüzü Pre-FEC BER")
    ax6.set_title("6. İyonosferik Kanalda Sayısal Modların BER Başarımı", fontsize=10.5, color="#e0e0e0")
    ax6.set_xlabel("Eb/N0 (dB)")
    ax6.set_ylabel("Bit Hata Oranı (BER)")
    ax6.set_ylim(1e-5, 0.5)
    ax6.grid(True, alpha=0.25, which="both")
    ax6.legend(fontsize=8, loc="lower left")

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig("iono_sense_dashboard.png", dpi=180)
    plt.savefig("iono_sense_dashboard.pdf")
    export_academic_reports(plasma_results, calibrated_stations, tcxo_est_hz)

    print("\n" + "=" * 92)
    print(" [OK] ÇIKTILAR BAŞARIYLA ÜRETİLDİ:")
    print("  -> Grafik Paneli (PNG/PDF) : iono_sense_dashboard.png & iono_sense_dashboard.pdf")
    print("  -> SQLite Telemetri DB     : iono_telemetry_ta7cae.db")
    print("  -> Typst Tez Raporu        : tez_bulgulari.typ")
    print("  -> LaTeX IEEE SİU Tablosu  : tez_bulgulari.tex")
    print("=" * 92)


if __name__ == "__main__":
    main()
