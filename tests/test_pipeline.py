#!/usr/bin/env python3
"""IONO-SENSE TA7CAE — Unit and Integration Test Suite."""

import numpy as np

from iono_sense.dsp.core import IonosphericDSPCore
from iono_sense.dsp.ldpc_decoder import FT8LDPCDecoder
from iono_sense.hardware.file_playback import IQFilePlaybackHardware
from iono_sense.physics.inversion import IonogramTrueHeightInverter
from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine
from iono_sense.physics.ray_tracer import HaselgroveRayTracer
from iono_sense.iono_sense import IonoSensePipeline
from iono_sense.simulation.fmcw_sounder import FMCWIonosondeSimulator
from iono_sense.simulation.watterson import WattersonChannelSimulator


def test_plasma_engine_and_attenuation():
    day = IonosphericPlasmaEngine(is_daytime=True)
    night = IonosphericPlasmaEngine(is_daytime=False)
    assert np.all(day.Ne >= 0.0)
    assert np.max(day.Ne) > np.max(night.Ne)
    assert day.f_H[0] > day.f_H[-1]

    res_low = day.compute_attenuation_models(3.573e6, elevation_deg=18.0)
    res_high = day.compute_attenuation_models(14.074e6, elevation_deg=18.0)
    assert not res_low["penetrates"]
    assert res_low["loss_sen_wyller_db"] > res_high["loss_sen_wyller_db"]
    assert res_low["loss_X_db"] > res_low["loss_O_db"]


def test_haselgrove_ray_tracer():
    plasma = IonosphericPlasmaEngine(is_daytime=True)
    tracer = HaselgroveRayTracer(plasma)

    ray_low = tracer.trace_ray(freq_mhz=14.074, elevation_deg=10.0)
    assert ray_low.reflected is True
    assert ray_low.apex_height_km > 90.0
    assert ray_low.ground_hit_km > 400.0

    ray_neg = tracer.trace_ray(freq_mhz=14.074, elevation_deg=-5.0)
    assert ray_neg.reflected is False
    assert ray_neg.apex_height_km == 0.0


def test_fmcw_ionosonde_synthesis():
    plasma = IonosphericPlasmaEngine(is_daytime=True)
    sounder = FMCWIonosondeSimulator(plasma_engine=plasma)
    data = sounder.synthesize_ionogram_profile(n_freq_steps=25, n_height_bins=25)
    assert data["ionogram_db"].shape == (25, 25)
    assert data["critical_frequencies"]["foF2_mhz"] > 4.0
    assert data["critical_frequencies"]["fxF2_mhz"] > data["critical_frequencies"]["foF2_mhz"]


def test_true_height_inversion_and_martyn_theorem():
    inverter = IonogramTrueHeightInverter(default_h0_km=60.0)
    freqs = np.linspace(1.5, 9.5, 20)
    true_h = 60.0 + 2.2 * (freqs**2)

    M = inverter.build_jackson_kernel(freqs)
    dh = np.diff(np.concatenate([[60.0], true_h]))
    h_prime = 60.0 + M @ dh

    res_j = inverter.invert_jackson(freqs, h_prime, h0_km=60.0)
    assert np.allclose(res_j.true_heights_km, true_h, atol=1e-4)
    assert res_j.residual_rms_km < 1e-6

    res_t = inverter.invert_tikhonov(freqs, h_prime, h0_km=60.0, reg_lambda=1e-3)
    assert np.all(np.diff(res_t.true_heights_km) >= -1e-9)
    assert abs(res_t.hmF2_km - true_h[-1]) < 3.0

    fv, hv = inverter.convert_oblique_to_vertical(f_oblique_mhz=14.0, delay_ms=4.0, distance_km=800.0)
    assert 0.0 < fv < 14.0
    assert hv > 100.0


def test_watterson_ldpc_and_dsp():
    sim = WattersonChannelSimulator(fs=12000.0, rng_seed=42)
    decoder = FT8LDPCDecoder(max_iterations=15)
    dsp = IonosphericDSPCore(fs=12000.0)

    _, cw = sim.generate_ldpc_codeword()
    llrs = np.where(cw == 0, 6.0, -6.0)
    decoded, ok, _ = decoder.decode(llrs)
    assert ok is True
    assert np.array_equal(decoded, cw)

    frame = sim.simulate_band_frame(band_mhz=14.074, tcxo_common_drift_hz=1.35, station_count=3)
    raw = [dsp.estimate_station_parameters(frame["iq_buffer"], gt, 14.074) for gt in frame["ground_truth"]]
    tcxo_est, cal = dsp.calibrate_and_extract_doppler(raw)
    assert abs(tcxo_est - 1.35) < 0.8
    assert len(cal) == 3


def test_iq_file_playback_and_pipeline_modes():
    player = IQFilePlaybackHardware(file_path="non_existent.wav", fs=12000.0)
    slot = player.capture_slot(freq_mhz=14.074)
    assert len(slot["iq_buffer"]) == int(12000 * 15)

    pipe = IonoSensePipeline(mode="benchmark", seed=42)
    assert "BENCHMARK" in pipe.get_execution_metadata()["mode_title"]
    inv_dict = pipe.run_true_height_inversion(method="tikhonov", reg_lambda=0.001)
    assert inv_dict["foF2_mhz"] > 4.0
    assert 150.0 < inv_dict["hmF2_km"] < 420.0
