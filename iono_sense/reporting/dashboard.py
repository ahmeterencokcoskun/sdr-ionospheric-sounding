#!/usr/bin/env python3
"""
Akademik Gösterge Paneli Çizicisi:
Plazma profili, zayıflama karşılaştırması, FT8 spektrogramı, mikro-Doppler yörüngesi,
saçılma fonksiyonu ve çok modlu BER eğrilerini içeren 6 panelli figürü üretir.
"""

from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class AcademicDashboardPlotter:
    """6 Panelli Akademik Dashboard Çizen OOP Sınıfı."""

    def __init__(
        self,
        figsize: tuple = (18, 10.5),
        dpi: int = 160,
        dark_mode: bool = True,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.dark_mode = dark_mode

    def plot_dashboard(
        self,
        plasma_day,
        plasma_night,
        plasma_results: List[Dict],
        frame_20m: Dict,
        calibrated_stations: List[Dict],
        dsp_core,
        sw_data: Dict,
        save_png: str = "iono_sense_dashboard.png",
        save_pdf: str = "iono_sense_dashboard.pdf",
    ) -> None:
        """Dashboard figürünü oluşturur ve PNG/PDF olarak kaydeder."""
        if self.dark_mode:
            plt.style.use("dark_background")

        fig, axes = plt.subplots(2, 3, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(
            "IONO-SENSE TA7CAE: Çok-Bantlı Fırsatçı İyonosferik Sondaj ve Mikro-Doppler Analiz Paneli\n"
            f"QTH: Kayseri (38.72°N, 35.48°E) | SFI: {sw_data['SFI']:.0f} | Kp: {sw_data['Kp']:.0f} | "
            f"IMF Bz: +{sw_data['Bz_nT']} nT | X-Ray: C1.0",
            fontsize=13,
            fontweight="bold",
            color="#00e5ff",
        )

        # Panel 1: Plazma Frekansı Profili fp(h)
        ax1 = axes[0, 0]
        ax1.plot(plasma_day.fp_hz / 1e6, plasma_day.h_km, color="#00ff9d", lw=2.2, label="Gündüz (D, E, F1, F2)")
        ax1.plot(plasma_night.fp_hz / 1e6, plasma_night.h_km, color="#ff9100", lw=2.0, ls="--", label="Gece (Birleşik F)")
        ax1.set_title("1. İyonosferik Plazma Frekansı Profili fp(h)", fontsize=10.5, color="#e0e0e0")
        ax1.set_xlabel("Plazma Kritik Frekansı fp (MHz)")
        ax1.set_ylabel("İrtifa h (km)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="lower right", fontsize=8.5)

        # Panel 2: Zayıflama (Alpha) Model Karşılaştırması
        ax2 = axes[0, 1]
        valid_r = [r for r in plasma_results if not r.get("penetrates", False)]
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

        # Panel 3: FT8 Spektrogramı
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

        # Panel 4: Mikro-Doppler İzleyici
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

        # Panel 5: Kanal Saçılma Fonksiyonu S(tau, fd)
        ax5 = axes[1, 1]
        tau_ms, fd_grid, S_db = dsp_core.compute_scattering_function(target_st["z_smooth"])
        im5 = ax5.contourf(tau_ms, fd_grid, S_db, levels=24, cmap="viridis")
        ax5.set_title("5. Kanal Saçılma Fonksiyonu S(τ, fd) [1F2 & 2F2]", fontsize=10.5, color="#e0e0e0")
        ax5.set_xlabel("Çok-Yollu Grup Gecikmesi τ (ms)")
        ax5.set_ylabel("Doppler Frekansı fd (Hz)")
        fig.colorbar(im5, ax=ax5, label="Normalize Güç (dB)")

        # Panel 6: BER Başarım Eğrileri
        ax6 = axes[1, 2]
        ber_data = dsp_core.compute_multimode_ber_curves()
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
        if save_png:
            plt.savefig(save_png, dpi=180)
        if save_pdf:
            plt.savefig(save_pdf)
        plt.close(fig)
