# Project: SDR Ionospheric Sounder (IONO-SENSE TA7CAE) Offline Pipeline Enhancement & Academic Validation

## Architecture
The project is a local, offline SDR-based ionospheric sounder combining plasma physics modeling, Haselgrove numerical ray tracing, FMCW radar sounding, HF digital mode simulation (Watterson channel, FT8 8-GFSK, WSPR 4-CPFSK), DSP demodulation and Doppler extraction, LDPC belief propagation error correction, true-height profile inversion, and academic reporting (SQLite, Matplotlib vector dashboard, Typst, LaTeX).

Data flow:
1. Space Weather / Geo: `StationGeoConfig` & NOAA SWPC local cache -> `IonosphericPlasmaEngine` (Chapman profiles, collision frequencies, Appleton-Hartree magneto-ionic indices, Sen-Wyller kinetic corrections).
2. Radar Sounding: `FMCWIonosondeSimulator` synthesizes beat-frequency and group delay virtual heights $h'(f)$ via Breit-Tuve integral for O and X modes.
3. True-Height Inversion: `IonogramTrueHeightInverter` (Jackson lower-triangular matrix lamination + Tikhonov regularized inversion + Martyn's equivalence) inverts $h'(f) \to N_e(h)$ extracting $foF2, hmF2, NmF2, \text{TEC}_{\text{bottom}}$.
4. Waveform & Propagation: `WattersonChannelSimulator` generates synthetic HF multipath, TID Doppler, and Middleton Class-A noise; or `IQFilePlaybackHardware` reads recorded SDR I/Q.
5. Signal Processing: `IonosphericDSPCore` performs DDC, narrow-band FIR channel isolation, phase stripping, Costas sync, Savitzky-Golay micro-Doppler extraction, WSPR drift regression, and 2D delay-Doppler scattering function.
6. Error Correction: `FT8LDPCDecoder` performs Normalized Min-Sum belief propagation decoding with systematic parity check matrix.
7. Reporting: `AcademicDashboardPlotter` (6-panel vector plot), `AcademicReportExporter` (Typst & IEEE LaTeX tables), and `DummyLoadHardware` (SQLite telemetry logging).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Geomagnetic Coordinate Config | QTH coordinates, magnetic dip, gyrofrequency parameters (`StationGeoConfig`) | M1 | Survey |
| 2 | Dynamic Multi-Layer Chapman Profile | Multi-layer $N_e(h)$ ($D, E, F_1, F_2, E_s$) and electron collision frequency $\nu(h)$ | M1 | Survey |
| 3 | Appleton-Hartree Dispersion & Absorption | Magneto-ionic refractive indices $\mu_O, \mu_X$ and Sen-Wyller kinetic Dingle corrections | M1 | Survey |
| 4 | X-Mode Cutoff Truncation | Truncate X-mode absorption integration at $X=1-Y$ instead of integrating into evanescent stopband | M1 | Survey |
| 5 | Haselgrove 2D Ray Tracing | Numerical RK4 ray propagation in spherical coordinates $(r, \theta, \psi)$ with phase/group path | M1 | Survey |
| 6 | Ray Tracer Space Escape Guard | Correct sign check to $\cos\psi > 0$ for upward escaping rays exiting ionosphere at $h > 430$ km | M1 | Survey |
| 7 | Breit-Tuve Virtual Height with Free Space | Physical $h'(f) = 60.0 + \int_{60}^{h_{\text{refl}}} \mu' dh$ ensuring $h'(f) \ge h_{\text{refl}}$ | M1 | Survey |
| 8 | X-Mode Group Refractive Index Dispersion | Add magnetic dispersion derivative $\frac{1}{2}\frac{XY}{(1-Y)^2}$ to $\mu'_X$ | M1 | Survey |
| 9 | Watterson Multipath Channel | ITU-R F.1487 ionospheric channel simulator with Rayleigh fading and TID Doppler | M1 | Survey |
| 10 | Zero Multipath Delay Guard | Fix slicing bug `[:-delay_samples]` when delay is 0 | M1 | Survey |
| 11 | Systematic LDPC Parity Encoding | Harmonize `watterson.py` codeword generator with `FT8LDPCDecoder` systematic parity matrix | M1 | Survey |
| 12 | LDPC Min-Sum Erasure & Numerical Guard | Fix zero-LLR erasure silencing, prevent sign annihilation, clip LLRs to $[-50, +50]$ | M1 | Survey |
| 13 | Adaptive Savitzky-Golay Filter Window | Dynamically clamp filter window length to input array size preventing `ValueError` | M1 | Survey |
| 14 | Genuine WSPR Drift Regression | Replace mock ground-truth injection with carrier phase de-drift regression | M1 | Survey |
| 15 | Common-Mode TCXO Drift Estimation | Estimate TCXO drift from median measured multi-station drift without ground truth cheating | M1 | Survey |
| 16 | Dashboard Unbound Variable Guard | Guard `target_st` access in `AcademicDashboardPlotter` when stations list is empty | M1 | Survey |
| 17 | NOAA SWPC Missing Data Guard | Guard `-1.0` missing values in space weather parser | M1 | Survey |
| 18 | Dead Code Cleanup | Remove redundant legacy file `iono_sense/hardware/DummyLoadHardware.py` | M1 | Survey |
| 19 | True-Height Profile Inversion Engine | Jackson closed-form lower-triangular matrix lamination $M_{ik}$ inverting $h'(f) \to N_e(h)$ | M2 | Survey |
| 20 | Tikhonov Regularized Inversion | Regularized least-squares inversion with 2nd-order curvature regularization for noisy ionograms | M2 | Survey |
| 21 | Martyn's Equivalence Oblique Conversion | Oblique-to-vertical ionogram transformation $f_v = f_{oblique}\cos\phi_0, h'_v = \frac{1}{2}\sqrt{P'^2 - D^2}$ | M2 | Survey |
| 22 | Physical Parameter Extraction | Automatic extraction of $foF2, hmF2, NmF2$, and sub-peak Total Electron Content (TEC) | M2 | Survey |
| 23 | FMCW Dechirping & Beat Frequency Mapping | Mixing $s_{rx}(t) \cdot s_{tx}^*(t)$, beat frequency intermediate frequency processing $f_b = \frac{2R}{c}\alpha$ | M2 | Survey |
| 24 | Real 2D Delay-Doppler Scattering Function | Compute actual 2D matched-filter scattering function from input signals replacing synthetic Gaussians | M2 | Survey |
| 25 | Pipeline LDPC Integration | Integrate LDPC decoding into `IonoSensePipeline` for post-FEC frame success metrics | M2 | Survey |
| 26 | Real I/Q File Playback Integration | Wire `IQFilePlaybackHardware` into pipeline when `--iq-file` is specified | M2 | Survey |
| 27 | CLI Enhancement & Dashboard Overlay | Add `--invert`, `--inversion-method`, `--reg-lambda` CLI options and overlay $h(f)$ on ionogram | M2 | Survey |
| 28 | Budden Parabolic Inversion Benchmark | Analytical parabolic layer benchmark verifying true-height inversion error $< 1.0$ km | M3 | Survey |
| 29 | Linear Layer Virtual Height Doubling Benchmark | Analytical benchmark verifying $h'(f) - h_0 = 2(h(f) - h_0)$ | M3 | Survey |
| 30 | Haselgrove Bouguer Invariant Benchmark | Verify spherical Snell's law invariant $r \mu \sin\psi = \text{const}$ conserved within $< 0.25\%$ | M3 | Survey |
| 31 | Space Escape Termination Benchmark | Verify penetrating rays terminate at $h > 430$ km with $\cos\psi > 0$ in $< 1000$ steps | M3 | Survey |
| 32 | Breit-Tuve Positivity Benchmark | Verify $h'(f) > h_{\text{refl}}(f) > 60.0$ km across entire frequency sweep | M3 | Survey |
| 33 | Appleton-Hartree Asymptotics Benchmark | Verify high-frequency convergence to linearized plasma dispersion within $< 0.01\%$ | M3 | Survey |
| 34 | Martyn Oblique-to-Vertical Benchmark | Verify equivalence consistency between oblique ray tracing and vertical inversion $< 5.0$ km RMS | M3 | Survey |
| 35 | LDPC Waterfall & Noise Correction Benchmark | Verify belief propagation error correction across SNR range (0 dB to 20 dB) | M3 | Survey |
| 36 | Comprehensive E2E Verification & Reporting | Full end-to-end execution of `python3 main.py` in live, sim, and benchmark modes with verified artifacts | M4 | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Codebase Hardening & Physics Bug Fixes | Fix all 22 identified numerical, physical, and runtime bugs in `physics/`, `dsp/`, `simulation/`, `hardware/`, and `reporting/` | none | DONE |
| M2 | True-Height Inversion & Offline Pipeline Enhancement | Implement `iono_sense/physics/inversion.py`, FMCW dechirping, delay-Doppler scattering, LDPC & I/Q file pipeline wiring, CLI options | M1 | IN_PROGRESS |
| M3 | Academic Validation & Analytical Benchmarks | Implement dedicated academic test suite in `tests/test_academic_validation.py` covering all 7 analytical benchmarks + LDPC waterfall | M1, M2 | PLANNED |
| M4 | E2E Integration, Dual-Track E2E Test Suite & Hardening | Opaque-box E2E test suite (Tiers 1-5), CLI mode validation, database & publication artifact verification | M1, M2, M3 | PLANNED |

## Interface Contracts
### `iono_sense.physics.inversion` ↔ `iono_sense.pipeline`
- Class `IonogramTrueHeightInverter`:
  - `invert_jackson(freqs_mhz: np.ndarray, h_prime_km: np.ndarray, h0_km: float = 60.0) -> InversionResult`
  - `invert_tikhonov(freqs_mhz: np.ndarray, h_prime_km: np.ndarray, h0_km: float = 60.0, reg_lambda: float = 1e-3) -> InversionResult`
  - `convert_oblique_to_vertical(f_oblique_mhz: float, delay_ms: float, distance_km: float) -> Tuple[float, float]`
- Dataclass `InversionResult`:
  - `true_heights_km: np.ndarray`
  - `plasma_freqs_mhz: np.ndarray`
  - `electron_density_m3: np.ndarray`
  - `foF2_mhz: float`
  - `hmF2_km: float`
  - `NmF2_m3: float`
  - `subpeak_tec_tecu: float`
  - `residual_rms_km: float`

### `iono_sense.simulation.watterson` ↔ `iono_sense.dsp.ldpc_decoder`
- `WattersonChannelSimulator.generate_ldpc_codeword() -> Tuple[np.ndarray, np.ndarray]`
  - Generates 91 information bits and 83 parity bits using systematic generator $p = (H_{\text{info}} \cdot m) \pmod 2$.
  - Guaranteed syndrome $(H \cdot c^T) \pmod 2 == 0$.

### `iono_sense.simulation.fmcw_sounder` ↔ `iono_sense.pipeline`
- `FMCWIonosondeSimulator.synthesize_ionogram_profile(n_freq_steps: int = 40, n_height_bins: int = 150) -> Dict`
  - Returns $h'(f) = 60.0 + \int_{60}^{h_{\text{refl}}} \mu' dh \ge h_{\text{refl}}$.
  - Includes `h_prime_O_km`, `h_prime_X_km`, `true_refl_h_km`, `critical_frequencies`.

## Code Layout
- `iono_sense/`
  - `physics/`
    - `plasma_engine.py`: Chapman layer $N_e(h)$, collision frequency $\nu(h)$, Appleton-Hartree $\mu_{O,X}$, Sen-Wyller kinetic corrections.
    - `ray_tracer.py`: Haselgrove 2D RK4 ray tracing in spherical $(r, \theta, \psi)$, Hamiltonian invariants, phase/group path.
    - `inversion.py`: **[NEW]** Jackson piecewise linear lamination matrix inversion, Tikhonov regularized inversion, Martyn equivalence, physical ionospheric parameter extraction ($foF2, hmF2, NmF2, \text{TEC}$).
  - `simulation/`
    - `fmcw_sounder.py`: Breit-Tuve group delay integral with 60 km free space, magnetic dispersion $\mu'_X$, FMCW dechirping and beat frequency mapping.
    - `watterson.py`: ITU-R F.1487 multipath channel, systematic LDPC codeword generation, 8-GFSK and 4-CPFSK synthesis.
  - `dsp/`
    - `core.py`: DDC FIR channel isolation, phase stripping, micro-Doppler extraction, genuine WSPR carrier drift regression, 2D delay-Doppler scattering function.
    - `ldpc_decoder.py`: Normalized Min-Sum belief propagation decoding with zero-LLR erasure protection and numerical clamping.
  - `hardware/`
    - `dummy_hardware.py`: Local SQLite telemetry logger and robust space weather parser.
    - `file_playback.py`: Offline IQ file ingestion with sample-rate adaptation.
  - `reporting/`
    - `dashboard.py`: 6-panel academic publication vector dashboard with true-height profile overlay.
    - `document_exporter.py`: Typst and LaTeX IEEE SIU report exporter with inversion metrics.
  - `pipeline.py`: Central offline sounding pipeline orchestrating all stages.
- `tests/`
  - `test_pipeline.py`: Existing baseline tests (must continue passing without regression).
  - `test_academic_validation.py`: **[NEW]** Dedicated first-principles academic validation benchmarks and synthetic test suite.
  - `test_e2e_suite.py`: **[NEW]** Multi-tier opaque-box E2E test suite.
- `main.py`: CLI entry point supporting `--mode`, `--seed`, `--iq-file`, `--invert`, `--inversion-method`, `--reg-lambda`.
