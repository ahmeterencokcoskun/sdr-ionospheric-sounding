#!/usr/bin/env python3
"""
Akademik Rapor Dışa Aktarıcı:
Bulguları Typst (.typ) ve LaTeX (.tex) formatında standart IEEE/SİU formatında üretir.
"""

from typing import List, Dict, Optional


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

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def export_latex(
        self,
        plasma_table: List[Dict],
        filepath: str = "tez_bulgulari.tex",
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
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
