#!/usr/bin/env python3
"""
Test Süiti: Yeni nesne yönelimli paket yapısının (iono_sense) doğrulanması.
"""

import numpy as np
import pytest

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine
from iono_sense.simulation.watterson import WattersonChannelSimulator
from iono_sense.simulation.fmcw_sounder import FMCWIonosondeSimulator
from iono_sense.dsp.core import IonosphericDSPCore
from iono_sense.pipeline import IonoSensePipeline


def test_plasma_engine():
    day = IonosphericPlasmaEngine(is_daytime=True)
    night = IonosphericPlasmaEngine(is_daytime=False)
    assert np.all(day.Ne >= 0.0)
    assert np.max(day.Ne) > np.max(night.Ne)


def test_watterson_and_wspr():
    sim = WattersonChannelSimulator(fs=12000.0, rng_seed=42)
    slot = sim.simulate_wspr_slot(band_mhz=14.0956, tcxo_drift_slope_hz_min=0.45)
    assert len(slot["ground_truth"]) == 4


def test_fmcw_sounder():
    sounder = FMCWIonosondeSimulator()
    data = sounder.synthesize_ionogram_profile(n_freq_steps=50, n_height_bins=50)
    assert data["ionogram_db"].shape == (50, 50)


def test_pipeline_integration():
    pipeline = IonoSensePipeline(fs=12000.0)
    plasma_res = pipeline.run_plasma_attenuation_analysis([14.074e6])
    assert len(plasma_res) == 1
    assert "loss_O_db" in plasma_res[0]
