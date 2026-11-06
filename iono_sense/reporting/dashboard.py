#!/usr/bin/env python3
"""
Akademik Gösterge Paneli Çizicisi (Genişletilmiş 6 Panelli Vektörel Dashboard).
Panel 3 içine dikey FMCW İyonogram ($h'-f$) renk haritasını ve Haselgrove Işın İzleme fanını entegre eder.
"""

from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class AcademicDashboardPlotter:
    """Akademik Araştırma Dashboard Çizicisi."""

    def __init__(
        self,
        figsize: tuple = (18, 11.5),
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
        ionogram_data: Optional[Dict] = None,
        ray_fan: Optional[List] = None,
        save_png: str = "iono_sense_dashboard.png",
        save_pdf: str = "iono_sense_dashboard.pdf",
    ) -> None:
        """Dashboard figürünü oluşturur ve PNG/PDF olarak kaydeder."""
        if self.dark_mode:
            plt.style.use("dark_background")

        fig, axes = plt.subplots(2, 3, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(
            "IONO-SENSE TA7CAE: Çok-Bantlı Fırsatçı İyonosferik Sondaj ve Plazma Analiz Paneli\n"
            f"QTH: Kayseri / Talas (38.6875°N, 35.5555°E - KM78sq) | SFI: {sw_data['SFI']:.0f} | Kp: {sw_data['Kp']:.0f} | "
            f"IMF Bz: {sw_data['Bz_nT']:+.1f} nT",
            fontsize=13,
            fontweight="bold",
            color="#00e5ff",
        )

        # Panel 1: Plazma Frekansı Profili fp(h)
        ax1 = axes[0, 0]
        ax1.plot(plasma_day.fp_hz / 1e6, plasma_day.h_km, color="#00ff9d", lw=2.2, label="Gündüz (D, E, F1, F2)")
        ax1.plot(plasma_night.fp_hz / 1e6, plasma_night.h_km, color="#ff9100", lw=2.0, ls="--", label="Gece (Birleşik F)")
        ax1.set_title("1. İyonosferik Plazma Frekansı Profili fp(h)", fontsize=10.5, color="#e0e0e0")
        ax1.set_xlabel("Plazma Frekansı fp (MHz)")
        ax1.set_ylabel("İrtifa h (km)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="lower right", fontsize=8.5)

        # Panel 2: Zayıflama (Alpha) Model Karşılaştırması
        ax2 = axes[0, 1]
        valid_r = [r for r in plasma_results if not r.get("penetrates", False)]
        f_labels = [f"{r['freq_mhz']:.1f}" for r in valid_r]
        x_pos = np.arange(len(valid_r))
        w = 0.25
        ax2.bar(x_pos - w, [r["loss_approx_db"] for r in valid_r], width=w, color="#2979ff", label="Basit 1/f²")
        ax2.bar(x_pos, [r["loss_sen_wyller_db"] for r in valid_r], width=w, color="#00e676", label="Tam O-Modu (Sen-Wyller)")
        ax2.bar(x_pos + w, [r["loss_X_db"] for r in valid_r], width=w, color="#ff1744", label="Tam X-Modu")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(f_labels)
        ax2.set_yscale("log")
        ax2.set_title("2. Magneto-İyonik Zayıflama Hata Bütçesi (dB)", fontsize=10.5, color="#e0e0e0")
        ax2.set_xlabel("Taşıyıcı Frekans (MHz)")
        ax2.set_ylabel("İki Yönlü Emilim Kaybı (dB)")
        ax2.grid(True, alpha=0.25, which="both")
        ax2.legend(fontsize=8)

        # Panel 3: Fiziksel FMCW İyonogramı (h' - f)
        ax3 = axes[0, 2]
        if ionogram_data is not None:
            freqs = ionogram_data["freqs_mhz"]
            heights = ionogram_data["heights_km"]
            iono_db = ionogram_data["ionogram_db"]
            im3 = ax3.pcolormesh(freqs, heights, iono_db, shading="auto", cmap="inferno")
            crit = ionogram_data.get("critical_frequencies", {})
            ax3.axvline(crit.get("foE_mhz", 3.4), color="#00e5ff", ls=":", lw=1.2, label=f"foE: {crit.get('foE_mhz', 3.4):.1f}")
            ax3.axvline(crit.get("foF2_mhz", 10.8), color="#ff1744", ls="--", lw=1.2, label=f"foF2: {crit.get('foF2_mhz', 10.8):.1f}")
            ax3.set_title("3. Dikey FMCW İyonogramı [Breit-Tuve h'(f)]", fontsize=10.5, color="#e0e0e0")
            ax3.set_xlabel("Sondaj Frekansı (MHz)")
            ax3.set_ylabel("Sanal Yükseklik h' (km)")
            fig.colorbar(im3, ax=ax3, label="Güç (dB)")
            ax3.legend(loc="upper left", fontsize=8)
        else:
            Sxx, f_spec, t_spec = matplotlib.mlab.specgram(
                frame_20m["iq_buffer"], NFFT=2048, Fs=12000.0, noverlap=1536
            )
            mask_f = (f_spec >= 400.0) & (f_spec <= 2700.0)
            im3 = ax3.pcolormesh(f_spec[mask_f], t_spec, 10.0 * np.log10(np.abs(Sxx[mask_f, :]).T + 1e-12), shading="auto", cmap="inferno")
            ax3.set_title("3. 14.074 MHz FT8 Taban Bant Spektrogramı", fontsize=10.5, color="#e0e0e0")
            ax3.set_xlabel("Frekans (Hz)")
            ax3.set_ylabel("Zaman (s)")

        # Panel 4: Mikro-Doppler İzleyici
        ax4 = axes[1, 0]
        if calibrated_stations:
            target_st = calibrated_stations[0]
            t_sub = target_st["t_sig"][8000:-8000]
            ax4.plot(t_sub, target_st["inst_doppler_true"][8000:-8000], color="#ffffff", lw=2.0, ls="--", label="Gerçek TID Doppler")
            ax4.plot(t_sub, target_st["fd_est"][8000:-8000], color="#00e5ff", lw=1.8, label=f"Kestirilen [{target_st['callsign']}]")
            ax4.set_title("4. Alt-Paketsel Mikro-Doppler Yörüngesi fd(t)", fontsize=10.5, color="#e0e0e0")
            ax4.set_xlabel("Paket İçi Zaman (s)")
            ax4.set_ylabel("Doppler Kayması (Hz)")
            ax4.grid(True, alpha=0.25)
            ax4.legend(fontsize=8.5)

        # Panel 5: Haselgrove 2B Işın İzleme (Ray-Tracing) Yörüngeleri
        ax5 = axes[1, 1]
        if ray_fan:
            for ray in ray_fan:
                ax5.plot(ray.ground_dist_km, ray.h_km, lw=1.8, label=f"Δ={ray.elevation_deg:.0f}° ({'Yansıdı' if ray.reflected else 'Kaçtı'})")
            ax5.axhline(110.0, color="#76ff03", ls=":", alpha=0.6, label="E Katmanı")
            ax5.axhline(290.0, color="#ff3d00", ls=":", alpha=0.6, label="F2 Katmanı")
            ax5.set_title("5. 2B Haselgrove Işın İzleme (14 MHz Ray-Fan)", fontsize=10.5, color="#e0e0e0")
            ax5.set_xlabel("Yer Mesafesi (km)")
            ax5.set_ylabel("İrtifa (km)")
            ax5.set_ylim(0, 420)
            ax5.grid(True, alpha=0.25)
            ax5.legend(fontsize=7.5, loc="upper right")
        else:
            tau_ms, fd_grid, S_db = dsp_core.compute_scattering_function(target_st["z_smooth"])
            im5 = ax5.contourf(tau_ms, fd_grid, S_db, levels=24, cmap="viridis")
            ax5.set_title("5. Kanal Saçılma Fonksiyonu S(τ, fd)", fontsize=10.5, color="#e0e0e0")
            ax5.set_xlabel("Grup Gecikmesi τ (ms)")
            ax5.set_ylabel("Doppler Frekansı fd (Hz)")

        # Panel 6: BER Başarım Eğrileri
        ax6 = axes[1, 2]
        ber_data = dsp_core.compute_multimode_ber_curves()
        ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_wspr"], color="#00e676", lw=2.2, label="WSPR (4-FSK + K=32)")
        ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_ft8"], color="#00e5ff", lw=2.2, label="FT8 (8-GFSK + LDPC)")
        ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_stanag"], color="#ffab00", lw=2.0, ls="-.", label="STANAG 4285")
        ax6.semilogy(ber_data["ebn0_db"], ber_data["ber_psk31"], color="#ff1744", lw=2.0, ls=":", label="PSK31")
        emp_snrs = [s["snr_db"] + 6.0 for s in calibrated_stations]
        emp_bers = [max(s["pre_fec_ber"], 1e-4) for s in calibrated_stations]
        ax6.scatter(emp_snrs, emp_bers, color="#ffffff", edgecolor="#ff1744", s=55, zorder=5, label="Ölçülen Pre-FEC BER")
        ax6.set_title("6. İyonosferik Modların BER Karşılaştırması", fontsize=10.5, color="#e0e0e0")
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
