#!/usr/bin/env python3
"""
Akademik Rapor Dışa Aktarıcı:
Bulguları Typst (.typ) ve LaTeX (.tex) formatında standart IEEE/SİU formatında üretir.
"""

from typing import List, Dict, Optional, Any


class AcademicReportExporter:
    """Typst ve LaTeX formatında akademik raporlar ve tablolar üreten sınıf."""

    def __init__(
        self,
        operator_callsign: str = "TA7CAE",
        operator_name: str = "Ahmet Eren Çokcoşkun",
        qth_name: str = "Kayseri / Talas",
        grid: str = "KM78",
        coordinates_str: str = "38.72°N, 35.48°E",
    ):
        self.callsign = operator_callsign
        self.name = operator_name
        self.qth = qth_name
        self.grid = grid
        self.coords = coordinates_str

    def export_typst(
        self,
        plasma_table: List[Dict],
        station_metrics: List[Dict],
        tcxo_est: float,
        wspr_metrics: Optional[List[Dict]] = None,
        filepath: str = "tez_bulgulari.typ",
        inversion_result: Optional[Any] = None,
    ) -> None:
        """Typst akademik raporunu diske yazar."""
        typst_rows = "\n".join(
            f"  [{r['freq_mhz']:.3f} MHz], [{r['layer']}], "
            f"[{r['refl_height_km']:.1f} km], [{r['loss_approx_db']:.2f} dB], "
            f"[{r['loss_O_db']:.2f} dB], [{r['loss_X_db']:.2f} dB], [{r['discrepancy_X_db']:.2f} dB],"
            for r in plasma_table
            if not r.get("penetrates", False)
        )

        content = (
            "// Otomatik Üretilen Akademik Bulgular Raporu (Typst)\n"
            f"// Yazar: {self.name} ({self.callsign}) - Nuh Naci Yazgan Üniversitesi EEE\n\n"
            '#set page(paper: "a4", margin: 2cm)\n'
            '#set text(font: "New Computer Modern", size: 10.5pt, lang: "tr")\n\n'
            "= Çok-Bantlı Fırsatçı İyonosferik Algılama ve Mikro-Doppler Kestirim Bulguları\n"
            f"*Operatör:* {self.name} (`{self.callsign}`) | *Konum:* {self.qth} (`{self.grid}`, ${self.coords}$)\n\n"
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

        if wspr_metrics:
            content += (
                "\n== 3. WSPR (4-FSK) İyonosferik Frekans Sürüklenmesi (Drift Rate) ve Katman İvmesi\n"
                "#table(\n"
                "  columns: (auto, auto, auto, auto, auto, auto),\n"
                "  align: center,\n"
                "  [*Çağrı İşareti*], [*Grid*], [*Menzil*], [*SNR*], [*İyonosferik Sürüklenme*], [*Katman İvmesi $a_z$*],\n"
            )
            for wm in wspr_metrics:
                content += (
                    f"  [{wm['callsign']}], [{wm['grid']}], [{wm['distance_km']:.0f} km], "
                    f"[{wm['snr_db']:+.1f} dB], [{wm['net_iono_drift_hz_min']:+.2f} Hz/dk], [{wm['layer_accel_mps2']:+.3f} m/s²],\n"
                )
            content += ")\n"

        if inversion_result is not None:
            foF2 = float(getattr(inversion_result, "foF2_mhz", inversion_result.get("foF2_mhz", 0.0) if isinstance(inversion_result, dict) else 0.0))
            hmF2 = float(getattr(inversion_result, "hmF2_km", inversion_result.get("hmF2_km", 0.0) if isinstance(inversion_result, dict) else 0.0))
            NmF2 = float(getattr(inversion_result, "NmF2_m3", inversion_result.get("NmF2_m3", 0.0) if isinstance(inversion_result, dict) else 0.0))
            tec = float(getattr(inversion_result, "subpeak_tec_tecu", inversion_result.get("subpeak_tec_tecu", 0.0) if isinstance(inversion_result, dict) else 0.0))
            rms = float(getattr(inversion_result, "residual_rms_km", inversion_result.get("residual_rms_km", 0.0) if isinstance(inversion_result, dict) else 0.0))
            true_h = getattr(inversion_result, "true_heights_km", inversion_result.get("true_heights_km", None) if isinstance(inversion_result, dict) else None)
            pf = getattr(inversion_result, "plasma_freqs_mhz", inversion_result.get("plasma_freqs_mhz", None) if isinstance(inversion_result, dict) else None)
            ne = getattr(inversion_result, "electron_density_m3", inversion_result.get("electron_density_m3", None) if isinstance(inversion_result, dict) else None)

            content += (
                "\n== 4. Breit-Tuve Sanal İrtifadan Gerçek İrtifa Tersinimi (True-Height Inversion)\n"
                "FMCW dikey iyonosond sanal irtifa verileri $h'(f)$, Breit-Tuve integral denklemine dayanmaktadır:\n"
                "$ h'(f) = h_0 + integral_0^{h_r(f)} mu'(f, f_p(h)) dif h $\n\n"
                "Bu Abel-tipi integral denklemi, Jackson (1969) parçalı doğrusal laminasyon matris çekirdeği "
                "$M_(i k)$ ve Tikhonov 2. derece ayrık eğrilik regülarizasyon operatörü ($L$) kullanılarak tersinmiştir. "
                "Elde edilen tepe parametreleri ve alt-iyonosferik toplam elektron içeriği aşağıda özetlenmiştir:\n\n"
                "#table(\n"
                "  columns: (auto, auto, auto, auto, auto),\n"
                "  align: center,\n"
                "  [*foF2 (Kritik Frekans)*], [*hmF2 (Tepe İrtifası)*], [*NmF2 (Elektron Yoğunluğu)*], [*Alt-Tepe TEC*], [*Kalıntı RMS*],\n"
                f"  [{foF2:.3f} MHz], [{hmF2:.1f} km], [{NmF2:.3e} el/m³], [{tec:.2f} TECU], [{rms:.4f} km],\n"
                ")\n"
            )

            if true_h is not None and pf is not None and len(true_h) > 0:
                n_layers = len(true_h)
                step = max(1, n_layers // 7)
                sampled_indices = list(range(0, n_layers, step))
                if (n_layers - 1) not in sampled_indices:
                    sampled_indices.append(n_layers - 1)

                layer_rows = "\n".join(
                    f"  [{pf[idx]:.2f} MHz], [{true_h[idx]:.1f} km], [{(ne[idx] if ne is not None else 1.24e10*(pf[idx]**2)):.3e} el/m³],"
                    for idx in sampled_indices
                )
                content += (
                    "\n*Tersinim ile Kestirilen Katman Profili Örneklemleri:*\n"
                    "#table(\n"
                    "  columns: (auto, auto, auto),\n"
                    "  align: center,\n"
                    "  [*Plazma Frekansı $f_p$*], [*Gerçek İrtifa $h$*], [*Elektron Yoğunluğu $N_e$*],\n"
                    f"{layer_rows}\n"
                    ")\n"
                )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def export_latex(
        self,
        plasma_table: List[Dict],
        filepath: str = "tez_bulgulari.tex",
        inversion_result: Optional[Any] = None,
    ) -> None:
        """IEEE SİU ve Tez formatında LaTeX tablosunu diske yazar."""
        latex_rows = "\n".join(
            f"{r['freq_mhz']:.3f} & {r['layer']} & {r['refl_height_km']:.1f} & "
            f"{r['loss_approx_db']:.2f} & {r['loss_O_db']:.2f} & {r['loss_X_db']:.2f} & "
            f"{r['discrepancy_X_db']:.2f} \\\\"
            for r in plasma_table
            if not r.get("penetrates", False)
        )

        lines = [
            "% Otomatik Üretilen IEEE SİU / Bitirme Tezi Tablosu (LaTeX)",
            f"% Yazar: {self.name} ({self.callsign})",
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

        if inversion_result is not None:
            foF2 = float(getattr(inversion_result, "foF2_mhz", inversion_result.get("foF2_mhz", 0.0) if isinstance(inversion_result, dict) else 0.0))
            hmF2 = float(getattr(inversion_result, "hmF2_km", inversion_result.get("hmF2_km", 0.0) if isinstance(inversion_result, dict) else 0.0))
            NmF2 = float(getattr(inversion_result, "NmF2_m3", inversion_result.get("NmF2_m3", 0.0) if isinstance(inversion_result, dict) else 0.0))
            tec = float(getattr(inversion_result, "subpeak_tec_tecu", inversion_result.get("subpeak_tec_tecu", 0.0) if isinstance(inversion_result, dict) else 0.0))
            rms = float(getattr(inversion_result, "residual_rms_km", inversion_result.get("residual_rms_km", 0.0) if isinstance(inversion_result, dict) else 0.0))

            inv_table_lines = [
                "",
                "% ============================================================================",
                "% Gerçek İrtifa İyonogram Tersinimi (True-Height Inversion foF2, hmF2) Bulguları",
                r"% Breit-Tuve Grubu Gecikmesi h'(f) = h_0 + \int_0^{h_r} \mu'(f, h) dh",
                "% Jackson (1969) / Tikhonov Regularization",
                "% ============================================================================",
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Dikey FMCW İyonosond İyonogramından Sayısal Tersinim ile Kestirilen İyonosferik Tepe Parametreleri}",
                r"\begin{tabular}{ccccc}",
                r"\hline",
                r"\textbf{$f_oF2$ (MHz)} & \textbf{$h_mF2$ (km)} & \textbf{$N_mF2$ ($\mathrm{m}^{-3}$)} & \textbf{$\mathrm{TEC}_{\mathrm{bottom}}$ (TECU)} & \textbf{Kalıntı RMS (km)} \\",
                r"\hline",
                f"{foF2:.3f} & {hmF2:.1f} & {NmF2:.3e} & {tec:.2f} & {rms:.4f} \\\\",
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
            ]
            lines.extend(inv_table_lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
