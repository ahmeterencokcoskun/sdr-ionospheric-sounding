#!/usr/bin/env python3
"""
Genişletilmiş Pytest Birim ve Entegrasyon Test Süiti.
1. Plazma Motoru ve Chapman Profili
2. Haselgrove 2B Işın İzleme (Ray-Tracing)
3. FMCW Breit-Tuve İyonosond Radarı
4. FT8 Min-Sum LLR LDPC(174,91) Kod Çözücüsü
5. IQ Dosya Oynatıcı Katmanı
6. Pipeline Entegrasyonu ve Çalışma Modları
"""

import numpy as np
import pytest

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine
from iono_sense.physics.ray_tracer import HaselgroveRayTracer
from iono_sense.simulation.watterson import WattersonChannelSimulator
from iono_sense.simulation.fmcw_sounder import FMCWIonosondeSimulator
from iono_sense.dsp.ldpc_decoder import FT8LDPCDecoder
from iono_sense.hardware.file_playback import IQFilePlaybackHardware
from iono_sense.pipeline import IonoSensePipeline


def test_plasma_engine_profiles():
    day = IonosphericPlasmaEngine(is_daytime=True)
    night = IonosphericPlasmaEngine(is_daytime=False)
    assert np.all(day.Ne >= 0.0)
    assert np.max(day.Ne) > np.max(night.Ne)
    assert day.f_H[0] > day.f_H[-1]  # Manyetik alan irtifa ile azalmalı


def test_haselgrove_ray_tracer():
    plasma = IonosphericPlasmaEngine(is_daytime=True)
    tracer = HaselgroveRayTracer(plasma)
    # Düşük açılı ışın yansımalı
    ray_low = tracer.trace_ray(freq_mhz=14.074, elevation_deg=10.0)
    assert len(ray_low.h_km) > 10
    assert ray_low.apex_height_km > 100.0


def test_fmcw_ionosonde_synthesis():
    plasma = IonosphericPlasmaEngine(is_daytime=True)
    sounder = FMCWIonosondeSimulator(plasma_engine=plasma)
    data = sounder.synthesize_ionogram_profile(n_freq_steps=30, n_height_bins=30)
    assert data["ionogram_db"].shape == (30, 30)
    assert data["critical_frequencies"]["foF2_mhz"] > 4.0


def test_ldpc_decoder_parity_and_syndrome():
    decoder = FT8LDPCDecoder(max_iterations=10)
    # Sıfır hatasız kod kelimesi
    zero_codeword = np.zeros(174, dtype=np.int32)
    # Güçlü LLR değerleri (+5.0 -> bit 0)
    ideal_llrs = np.full(174, 5.0)
    decoded, success, iters = decoder.decode(ideal_llrs)
    assert success is True
    assert np.all(decoded == 0)


def test_iq_file_playback():
    player = IQFilePlaybackHardware(file_path="non_existent.wav", fs=12000.0)
    slot = player.capture_slot(freq_mhz=14.074)
    assert len(slot["iq_buffer"]) == int(12000 * 15)
    assert np.all(slot["iq_buffer"] == 0)


def test_pipeline_modes():
    pipe_live = IonoSensePipeline(mode="live")
    pipe_sim = IonoSensePipeline(mode="sim")
    pipe_bench = IonoSensePipeline(mode="benchmark", seed=42)

    assert "LIVE" in pipe_live.get_execution_metadata()["mode_title"]
    assert "STOKASTİK" in pipe_sim.get_execution_metadata()["mode_title"]
    assert "BENCHMARK" in pipe_bench.get_execution_metadata()["mode_title"]
