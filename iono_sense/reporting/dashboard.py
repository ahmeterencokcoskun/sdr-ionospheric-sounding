#!/usr/bin/env python3
"""6 Panelli Akademik Arastirma Gosterge Paneli Cizicisi."""

from typing import Any, Dict, List, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extract_inv_attr(inv: Any, key: str, default: Any = 0.0) -> Any:
    if isinstance(inv, dict):
        return inv.get(key, default)
    return getattr(inv, key, default)


class AcademicDashboardPlotter:
    """6 panelli vektorel analiz gosterge paneli ureticisi."""

    def __init__(self, figsize: tuple = (18, 11.5), dpi: int = 160, dark_mode: bool = True):
        self.figsize = figsize
        self.dpi = dpi
        self.dark_mode = dark_mode

    def plot_sounding_ionogram(
        self,
        ax,
        ionogram_data: Dict,
        inversion_result: Optional[Any] = None,
        fig=None,
    ) -> None:
        freqs = ionogram_data["freqs_mhz"]
        heights = ionogram_data["heights_km"]
        im = ax.pcolormesh(freqs, heights, ionogram_data["ionogram_db"], shading="auto", cmap="inferno", zorder=1)
        if fig is not None:
            fig.colorbar(im, ax=ax, label="Echo Power (dB)")

        crit = ionogram_data.get("critical_frequencies", {})
        foE = crit.get("foE_mhz", 3.4)
        foF2 = crit.get("foF2_mhz", 10.8)
        ax.axvline(foE, color="#00e5ff", ls=":", lw=1.2, label=f"foE: {foE:.1f} MHz", zorder=2)
        ax.axvline(foF2, color="#ff1744", ls="--", lw=1.2, label=f"foF2: {foF2:.1f} MHz", zorder=2)

        inv = inversion_result if inversion_result is not None else ionogram_data.get("inversion_result")
        if inv is not None:
            inv_f = np.asarray(_extract_inv_attr(inv, "plasma_freqs_mhz", []))
            inv_h = np.asarray(_extract_inv_attr(inv, "true_heights_km", []))
            if len(inv_f) > 0 and len(inv_h) > 0 and np.all(np.isfinite(inv_f)) and np.all(np.isfinite(inv_h)):
                inv_foF2 = float(_extract_inv_attr(inv, "foF2_mhz"))
                inv_hmF2 = float(_extract_inv_attr(inv, "hmF2_km"))
                inv_NmF2 = float(_extract_inv_attr(inv, "NmF2_m3"))
                inv_tec = float(_extract_inv_attr(inv, "subpeak_tec_tecu"))
                inv_rms = float(_extract_inv_attr(inv, "residual_rms_km"))

                ax.plot(inv_f, inv_h, color="#00ff9d", lw=2.2, ls="-", label="True Height h(f)", zorder=5)
                ax.plot(
                    inv_foF2, inv_hmF2, marker="*", markersize=9, color="#ffff00",
                    ls="", label=f"hmF2: {inv_hmF2:.1f} km", zorder=6,
                )
                anno_text = (
                    f"foF2 = {inv_foF2:.2f} MHz\n"
                    f"hmF2 = {inv_hmF2:.1f} km\n"
                    f"NmF2 = {inv_NmF2:.2e} m⁻³\n"
                    f"TEC  = {inv_tec:.2f} TECU\n"
                    f"RMS  = {inv_rms:.2f} km"
                )
                ax.text(
                    0.96, 0.05, anno_text, transform=ax.transAxes, fontsize=7.5, fontfamily="monospace",
                    va="bottom", ha="right", color="#ffffff", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#101426", edgecolor="#00e5ff", alpha=0.88, lw=1.0),
                )

        title_str = (
            "3. Vertical FMCW Ionogram & True-Height Inversion"
            if inv is not None
            else "3. Vertical FMCW Ionogram [Breit-Tuve h'(f)]"
        )
        ax.set_title(title_str, fontsize=10.5, color="#e0e0e0")
        ax.set_xlabel("Sounding Frequency (MHz)")
        ax.set_ylabel("Height / Virtual Height h, h' (km)")
        ax.set_ylim(heights[0], heights[-1])
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=7.5)

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
        inversion_result: Optional[Any] = None,
    ) -> None:
        if self.dark_mode:
            plt.style.use("dark_background")

        fig, axes = plt.subplots(2, 3, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(
            "IONO-SENSE TA7CAE: Multi-Band Opportunistic Ionospheric Sounding & Plasma Analysis Dashboard\n"
            f"QTH: Kayseri / Talas (38.6875°N, 35.5555°E - KM78sq) | SFI: {sw_data['SFI']:.0f} | "
            f"Kp: {sw_data['Kp']:.0f} | IMF Bz: {sw_data['Bz_nT']:+.1f} nT",
            fontsize=13,
            fontweight="bold",
            color="#00e5ff",
        )

        ax1 = axes[0, 0]
        ax1.plot(plasma_day.fp_hz / 1e6, plasma_day.h_km, color="#00ff9d", lw=2.2, label="Daytime (D, E, F1, F2)")
        ax1.plot(plasma_night.fp_hz / 1e6, plasma_night.h_km, color="#ff9100", lw=2.0, ls="--", label="Nighttime (Merged F)")
        ax1.set_title("1. Ionospheric Plasma Frequency Profile fp(h)", fontsize=10.5, color="#e0e0e0")
        ax1.set_xlabel("Plasma Frequency fp (MHz)")
        ax1.set_ylabel("Height h (km)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="lower right", fontsize=8.5)

        ax2 = axes[0, 1]
        valid_r = [r for r in plasma_results if not r.get("penetrates", False)]
        x_pos = np.arange(len(valid_r))
        w = 0.25
        ax2.bar(x_pos - w, [r["loss_approx_db"] for r in valid_r], width=w, color="#2979ff", label="Simple 1/f²")
        ax2.bar(x_pos, [r["loss_sen_wyller_db"] for r in valid_r], width=w, color="#00e676", label="Full O-Mode (Sen-Wyller)")
        ax2.bar(x_pos + w, [r["loss_X_db"] for r in valid_r], width=w, color="#ff1744", label="Full X-Mode")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{r['freq_mhz']:.1f}" for r in valid_r])
        ax2.set_yscale("log")
        ax2.set_title("2. Magneto-Ionic Attenuation Error Budget (dB)", fontsize=10.5, color="#e0e0e0")
        ax2.set_xlabel("Carrier Frequency (MHz)")
        ax2.set_ylabel("Two-Way Absorption Loss (dB)")
        ax2.grid(True, alpha=0.25, which="both")
        ax2.legend(fontsize=8)

        ax3 = axes[0, 2]
        if ionogram_data is not None:
            self.plot_sounding_ionogram(ax=ax3, ionogram_data=ionogram_data, inversion_result=inversion_result, fig=fig)
        else:
            Sxx, f_spec, t_spec = matplotlib.mlab.specgram(frame_20m["iq_buffer"], NFFT=2048, Fs=12000.0, noverlap=1536)
            mask_f = (f_spec >= 400.0) & (f_spec <= 2700.0)
            ax3.pcolormesh(
                f_spec[mask_f], t_spec, 10.0 * np.log10(np.abs(Sxx[mask_f, :]).T + 1e-12), shading="auto", cmap="inferno"
            )
            ax3.set_title("3. 14.074 MHz FT8 Baseband Spectrogram", fontsize=10.5, color="#e0e0e0")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Time (s)")

        ax4 = axes[1, 0]
        if calibrated_stations:
            st = calibrated_stations[0]
            sl = slice(8000, -8000) if len(st["t_sig"]) > 16000 else slice(None)
            ax4.plot(st["t_sig"][sl], st["inst_doppler_true"][sl], color="#ffffff", lw=2.0, ls="--", label="True TID Doppler")
            ax4.plot(st["t_sig"][sl], st["fd_est"][sl], color="#00e5ff", lw=1.8, label=f"Estimated [{st['callsign']}]")
            ax4.legend(fontsize=8.5)
        else:
            ax4.text(0.5, 0.5, "No Station Data Available", transform=ax4.transAxes, ha="center", va="center", color="#888888")
        ax4.set_title("4. Sub-Packet Micro-Doppler Trajectory fd(t)", fontsize=10.5, color="#e0e0e0")
        ax4.set_xlabel("Paket Ici Time (s)")
        ax4.set_ylabel("Doppler Shift (Hz)")
        ax4.grid(True, alpha=0.25)

        ax5 = axes[1, 1]
        if ray_fan:
            for ray in ray_fan:
                ax5.plot(
                    ray.ground_dist_km, ray.h_km, lw=1.8,
                    label=f"Δ={ray.elevation_deg:.0f}° ({'Reflected' if ray.reflected else 'Escaped'})",
                )
            ax5.axhline(110.0, color="#76ff03", ls=":", alpha=0.6, label="E Layer")
            ax5.axhline(290.0, color="#ff3d00", ls=":", alpha=0.6, label="F2 Layer")
            ax5.set_title("5. 2D Haselgrove Ray Tracing (14 MHz Ray-Fan)", fontsize=10.5, color="#e0e0e0")
            ax5.set_xlabel("Ground Distance (km)")
            ax5.set_ylabel("Height (km)")
            ax5.set_ylim(0, 420)
            ax5.grid(True, alpha=0.25)
            ax5.legend(fontsize=7.5, loc="upper right")
        else:
            z_in = calibrated_stations[0]["z_smooth"] if calibrated_stations else None
            tau_ms, fd_grid, S_db = dsp_core.compute_scattering_function(z_in)
            ax5.contourf(tau_ms, fd_grid, S_db, levels=24, cmap="viridis")
            ax5.set_title("5. Channel Scattering Function S(tau, fd)", fontsize=10.5, color="#e0e0e0")
            ax5.set_xlabel("Group Delay tau (ms)")
            ax5.set_ylabel("Doppler Frequency fd (Hz)")

        ax6 = axes[1, 2]
        ber = dsp_core.compute_multimode_ber_curves()
        ax6.semilogy(ber["ebn0_db"], ber["ber_wspr"], color="#00e676", lw=2.2, label="WSPR (4-FSK + K=32)")
        ax6.semilogy(ber["ebn0_db"], ber["ber_ft8"], color="#00e5ff", lw=2.2, label="FT8 (8-GFSK + LDPC)")
        ax6.semilogy(ber["ebn0_db"], ber["ber_stanag"], color="#ffab00", lw=2.0, ls="-.", label="STANAG 4285")
        ax6.semilogy(ber["ebn0_db"], ber["ber_psk31"], color="#ff1744", lw=2.0, ls=":", label="PSK31")
        ax6.scatter(
            [s["snr_db"] + 6.0 for s in calibrated_stations],
            [max(s["pre_fec_ber"], 1e-4) for s in calibrated_stations],
            color="#ffffff", edgecolor="#ff1744", s=55, zorder=5, label="Measured Pre-FEC BER",
        )
        ax6.set_title("6. Ionospheric Modes BER Comparison", fontsize=10.5, color="#e0e0e0")
        ax6.set_xlabel("Eb/N0 (dB)")
        ax6.set_ylabel("Bit Error Rate (BER)")
        ax6.set_ylim(1e-5, 0.5)
        ax6.grid(True, alpha=0.25, which="both")
        ax6.legend(fontsize=8, loc="lower left")

        plt.tight_layout(rect=[0, 0.01, 1, 0.94])
        if save_png:
            plt.savefig(save_png, dpi=180)
        if save_pdf:
            plt.savefig(save_pdf)
        plt.close(fig)
