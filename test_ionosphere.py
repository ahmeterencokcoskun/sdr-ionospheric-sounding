#!/usr/bin/env python3
"""
Otomatik Pytest Birim Test Süiti:
1. Plazma fiziği (Chapman profili, Appleton-Hartree, Sen-Wyller kırılma/emilim).
2. Watterson çok-yollu simülatörü ve FT8/WSPR dalga şekli üretimi.
3. DSP çekirdeği (Costas frekans kestirimi, TCXO kalibrasyonu, BER).
4. FMCW İyonosond kritik frekans ve sanal yükseklik tespiti.
"""

import numpy as np
import pytest

from plasma_physics import IonosphericPlasmaEngine
from watterson_simulator import WattersonChannelSimulator
from dsp_core import IonosphericDSPCore
from fmcw_ionosonde import FMCWIonosondeSimulator


def test_plasma_frequencies_and_electron_density():
    """Gündüz ve gece Chapman elektron yoğunluğu ve plazma frekansı sınırlarını doğrular."""
    engine_day = IonosphericPlasmaEngine(is_daytime=True)
    engine_night = IonosphericPlasmaEngine(is_daytime=False)

    # Elektron yoğunluğu sıfırdan büyük olmalı
    assert np.all(engine_day.Ne >= 0.0)
    assert np.all(engine_night.Ne >= 0.0)

    # Gündüz F2 tepe yoğunluğu geceye göre daha yüksek olmalı
    assert np.max(engine_day.Ne) > np.max(engine_night.Ne)

    # Kayseri için tepe plazma frekansı tipik HF sınırlarında (4 - 15 MHz) olmalı
    foF2_day = np.max(engine_day.fp_hz) / 1e6
    assert 4.0 <= foF2_day <= 15.0


def test_appleton_hartree_attenuation_order():
    """Appleton-Hartree formülasyonunda zayıflamanın frekans arttıkça düşmesini doğrular."""
    engine = IonosphericPlasmaEngine(is_daytime=True)
    res_3mhz = engine.compute_attenuation_models(3.573e6, elevation_deg=18.0)
    res_14mhz = engine.compute_attenuation_models(14.074e6, elevation_deg=18.0)

    # Düşük frekansta (3.5 MHz) D/E emilimi yüksek frekanstan (14 MHz) daha fazla olmalıdır
    assert res_3mhz["loss_O_db"] > res_14mhz["loss_O_db"]
    # Model hata payı tanımlı olmalı
    assert res_3mhz["discrepancy_X_db"] >= 0.0


def test_ft8_and_watterson_channel_generation():
    """Watterson simülatörünün 15s tamponu ve istasyon verilerini üretmesini doğrular."""
    sim = WattersonChannelSimulator(fs=12000.0, rng_seed=42)
    band_data = sim.simulate_band_frame(band_mhz=14.074, tcxo_common_drift_hz=1.35)

    assert band_data["band_mhz"] == 14.074
    assert len(band_data["iq_buffer"]) == int(15.0 * 12000.0)
    assert len(band_data["ground_truth"]) == 5
    assert not np.all(band_data["iq_buffer"] == 0)


def test_wspr_drift_and_dsp_estimation():
    """WSPR simülatörü ve DSP sürüklenme (drift) çözümlemesini doğrular."""
    sim = WattersonChannelSimulator(fs=12000.0, rng_seed=42)
    dsp = IonosphericDSPCore(fs=12000.0)

    wspr_slot = sim.simulate_wspr_slot(band_mhz=14.0956, tcxo_drift_slope_hz_min=0.45)
    est_drift_slope, results = dsp.analyze_wspr_signals(wspr_slot, f_carrier_mhz=14.0956)

    # TCXO eğimi hatası 0.1 Hz/dk altında olmalı
    assert abs(est_drift_slope - 0.45) < 0.1
    assert len(results) == 4
    for r in results:
        assert "net_iono_drift_hz_min" in r
        assert "layer_accel_mps2" in r


def test_fmcw_ionosonde_synthesis():
    """FMCW iyonosondunun sanal yükseklik ve kritik frekans matrisini doğrular."""
    ionosonde = FMCWIonosondeSimulator(f_start_mhz=1.0, f_stop_mhz=16.0)
    data = ionosonde.synthesize_ionogram_profile(n_freq_steps=80, n_height_bins=60)

    assert data["ionogram_db"].shape == (60, 80)
    crit = data["critical_frequencies"]
    assert crit["foE_mhz"] < crit["foF1_mhz"] < crit["foF2_mhz"]
    # En yüksek yansıma gücü mantıklı bir aralıkta olmalı
    assert np.max(data["ionogram_db"]) >= -1.0
