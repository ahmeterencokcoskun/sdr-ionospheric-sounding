#!/usr/bin/env python3
"""IONO-SENSE TA7CAE — Main Command & Executive Console (CLI Orchestrator)."""

import argparse
import numpy as np
from iono_sense.iono_sense import IonoSensePipeline

def non_negative_int(value: str) -> int:
    try:
        val = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Seed must be an integer, got '{value}'")
    if val < 0:
        raise argparse.ArgumentTypeError(f"Seed must be a non-negative integer, got {val}")
    return val


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IONO-SENSE TA7CAE: Multi-Band Opportunistic Ionospheric Sounding & DSP Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "sim", "benchmark"],
        default="live",
        help="Operation mode: live (NOAA SWPC + stochastic RF), sim (full simulation), benchmark (fixed seed)",
    )
    parser.add_argument("--seed", type=non_negative_int, default=42, help="Benchmark random seed (Default: 42)")
    parser.add_argument("--iq-file", type=str, default=None, help="SDR raw I/Q recording file (.wav or .cf32)")
    parser.add_argument("--invert", action="store_true", default=False, help="Compute true height inversion h(f)")
    parser.add_argument(
        "--inversion-method",
        choices=["jackson", "tikhonov"],
        default="jackson",
        help="Inversion solver algorithm: jackson or tikhonov (Default: jackson)",
    )
    parser.add_argument("--reg-lambda", type=float, default=0.001, help="Tikhonov regularization coefficient (0.001)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    pipeline = IonoSensePipeline(mode=args.mode, fs=12000.0, seed=args.seed, iq_file=args.iq_file)
    meta = pipeline.get_execution_metadata()

    print("=" * 96)
    print(" IONO-SENSE TA7CAE // MULTI-BAND OPPORTUNISTIC IONOSPHERIC SOUNDING & DSP ENGINE")
    print(" QTH: Kayseri / Talas (38.6875°N, 35.5555°E - KM78sq) | Operator: TA7CAE")
    print("-" * 96)
    print(f" OPERATION MODE      : [{meta['mode_title']}]")
    print(f"  • Space Weather Data : {meta['weather_source']}")
    print(f"  • Receiver Hardware  : {meta['rf_hardware']}")
    print(f"  • RF Signal Traffic  : {meta['signal_data']}")
    print(f"  • Ionogram Resolver  : {meta['ionogram_source']}")
    if args.invert:
        print(f"  • Ionogram Inversion : ACTIVE [{args.inversion_method.upper()}] (λ={args.reg_lambda})")
    else:
        print("  • Ionogram Inversion : DISABLED (Virtual height h'(f) only)")
    print("=" * 96)

    plasma_results = pipeline.run_plasma_attenuation_analysis()
    print("\n[1] PLASMA LAYER INTERACTION AND APPLETON-HARTREE ATTENUATION (ALPHA) ERROR BUDGET:")
    print(
        f"{'Carrier':<10} | {'Layer':<11} | {'Reflection':<11} | {'Simple 1/f^2':<12} | "
        f"{'Full O-Mode':<11} | {'Full X-Mode':<11} | {'Model Error'}"
    )
    print("-" * 96)
    for r in plasma_results:
        h_str = f"{r['refl_height_km']:>5.1f} km" if not np.isnan(r.get("refl_height_km", np.nan)) else "   ---    "
        print(
            f"{r['freq_mhz']:>6.3f} MHz | {r['layer']:<11} | {h_str:<11} | "
            f"{r['loss_approx_db']:>9.2f} dB | {r['loss_O_db']:>8.2f} dB | "
            f"{r['loss_X_db']:>8.2f} dB | {r['discrepancy_X_db']:>8.2f} dB"
        )

    ft8_res = pipeline.run_ft8_micro_doppler_analysis(freq_mhz=14.074)
    sw = ft8_res["sw_data"]
    print(
        f"\n[2] 14.074 MHz (20m F2 LAYER) FT8 DEMODULATION & COMMON-MODE TCXO CALIBRATION:\n"
        f"    Space Weather: SFI={sw['SFI']:.1f}, Kp={sw['Kp']:.1f}, A-Index={sw['A_index']:.1f}, Bz={sw['Bz_nT']:+.1f} nT\n"
        f"    True Receiver TCXO Offset: +{ft8_res['frame_data']['tcxo_true_hz']:.3f} Hz  -->  "
        f"Estimated Common-Mode Offset: +{ft8_res['tcxo_est_hz']:.3f} Hz"
    )
    print(
        f"{'Callsign':<10} | {'Distance':<9} | {'SNR':<8} | {'Calib fd':<12} | "
        f"{'Doppler Sprd':<13} | {'Layer Vel vz':<15} | {'Pre-FEC BER'}"
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

    wspr_res = pipeline.run_wspr_drift_analysis(freq_mhz=14.0956)
    print(
        f"\n[3] 14.0956 MHz WSPR (4-FSK) DEMODULATION & IONOSPHERIC DRIFT ANALYSIS:\n"
        f"    True TCXO Drift Slope: +{wspr_res['wspr_slot']['tcxo_drift_slope_hz_min']:.3f} Hz/dk  -->  "
        f"Estimated TCXO Slope: +{wspr_res['est_drift_slope']:.3f} Hz/dk"
    )
    print(
        f"{'Callsign':<10} | {'Distance':<9} | {'SNR':<8} | {'Meas Drift':<15} | "
        f"{'Net Iono Drift':<16} | {'Layer Accel az'}"
    )
    print("-" * 96)
    for wm in wspr_res["wspr_results"]:
        print(
            f"{wm['callsign']:<10} | {wm['distance_km']:>6.0f} km | "
            f"{wm['snr_db']:>+5.1f} dB | {wm['meas_drift_hz_min']:>+11.2f} Hz/dk | "
            f"{wm['net_iono_drift_hz_min']:>+12.2f} Hz/dk | {wm['layer_accel_mps2']:>+13.4f} m/s²"
        )

    fmcw_res = pipeline.run_fmcw_ionosonde_sweep()
    crit = fmcw_res["critical_frequencies"]
    print(
        f"\n[4] VERTICAL FMCW IONOSONDE PHYSICAL IONOGRAM SWEEP (1 - 15 MHz):\n"
        f"    Analytical Critical Frequencies: foE = {crit['foE_mhz']:.1f} MHz | foF1 = {crit['foF1_mhz']:.1f} MHz | "
        f"foF2 = {crit['foF2_mhz']:.1f} MHz | fxF2 = {crit['fxF2_mhz']:.1f} MHz"
    )

    inversion_res = None
    if args.invert:
        inversion_dict = pipeline.run_true_height_inversion(
            ionogram_data=fmcw_res,
            method=args.inversion_method,
            reg_lambda=args.reg_lambda,
        )
        inversion_res = inversion_dict["inversion_result"]
        print(
            f"\n[4.1] VERTICAL IONOGRAM TRUE-HEIGHT INVERSION ({args.inversion_method.upper()} METHOD):\n"
            f"    • Kritik Frekans (foF2)       : {inversion_res.foF2_mhz:.3f} MHz\n"
            f"    • Peak Height (hmF2)        : {inversion_res.hmF2_km:.1f} km\n"
            f"    • Peak Plasma Density (NmF2): {inversion_res.NmF2_m3:.3e} el/m³\n"
            f"    • Sub-Peak TEC (up to hmF2): {inversion_res.subpeak_tec_tecu:.2f} TECU\n"
            f"    • Reconstruction RMS Error  : {inversion_res.residual_rms_km:.4f} km"
        )

    ray_fan = pipeline.run_ray_tracing_fan(freq_mhz=14.074)
    print(
        f"\n[5] 2D HASELGROVE RAY-TRACING CALCULATION (14.074 MHz):\n"
        f"{'Elevation Δ':<16} | {'Status':<10} | {'Apex Height':<13} | {'Ground Range'}"
    )
    print("-" * 96)
    for ray in ray_fan:
        status_str = "REFLECTED" if ray.reflected else "UZAYA KACTI"
        dist_str = f"{ray.ground_hit_km:>6.1f} km" if ray.reflected else "  ---   "
        print(f"Δ = {ray.elevation_deg:>4.1f}°        | {status_str:<10} | {ray.apex_height_km:>7.1f} km   | {dist_str}")

    pipeline.generate_all_artifacts(
        plasma_results=plasma_results,
        ft8_analysis=ft8_res,
        wspr_analysis=wspr_res,
        ionogram_data=fmcw_res,
        ray_fan=ray_fan,
        inversion_result=inversion_res,
    )

    print("\n" + "=" * 96)
    print(" [OK] OUTPUTS SUCCESSFULLY GENERATED:")
    print("  -> Dashboard Plot (PNG/PDF) : iono_sense_dashboard.png & iono_sense_dashboard.pdf")
    print("  -> SQLite Telemetry DB      : iono_telemetry_ta7cae.db")
    print("  -> Typst Thesis Report      : thesis_findings.typ")
    print("  -> LaTeX IEEE SIU Table     : thesis_findings.tex")
    print("=" * 96)


if __name__ == "__main__":
    main()
