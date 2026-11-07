#!/usr/bin/env python3
"""İyonosferik Plazma Fiziği ve Küresel Magneto-İyonik Zayıflama Motoru."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


def integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class StationGeoConfig:
    grid: str = "KM78sq"
    lat_deg: float = 38.6875
    lon_deg: float = 35.5555
    alt_m: float = 1153.0
    R_earth_m: float = 6371.0e3
    B0_surface_T: float = 48.2e-6
    dip_angle_deg: float = 58.2


class IonosphericPlasmaEngine:
    """Küresel Snell yasası, irtifaya bağlı B(h), Te(h), H(h) ve Haselgrove yol çözücüsü."""

    def __init__(self, config: Optional[StationGeoConfig] = None, is_daytime: bool = True):
        self.cfg = config or StationGeoConfig()
        self.is_daytime = is_daytime

        self.e = 1.60217663e-19
        self.m_e = 9.1093837e-31
        self.eps0 = 8.8541878e-12
        self.c = 2.99792458e8
        self.k_B = 1.380649e-23
        self.amu = 1.660539e-27

        self.h_km = np.linspace(60.0, 450.0, 781)
        self.h_m = self.h_km * 1e3
        self.r_m = self.cfg.R_earth_m + self.h_m

        self.g = 9.80665 * (self.cfg.R_earth_m / self.r_m) ** 2
        self.B_tesla = self.cfg.B0_surface_T * (self.cfg.R_earth_m / self.r_m) ** 3
        self.omega_H = (self.e * self.B_tesla) / self.m_e
        self.f_H = self.omega_H / (2.0 * np.pi)

        self.Te = self._build_temperature_profile()
        self.m_neutral = self._build_molecular_mass_profile()
        self.H_m = (self.k_B * self.Te) / (self.m_neutral * self.g)
        self.H_km = self.H_m / 1e3

        self.Ne = self._build_dynamic_chapman_profile()
        self.nu = self._build_collision_profile()
        self.fp_hz = np.sqrt((self.Ne * self.e**2) / (self.eps0 * self.m_e)) / (2.0 * np.pi)

    def _build_temperature_profile(self) -> np.ndarray:
        T_meso = 185.0
        T_exo = 1450.0 if self.is_daytime else 850.0
        Te = T_exo - (T_exo - T_meso) * np.exp(-0.022 * np.maximum(self.h_km - 85.0, 0.0))
        d_mask = self.h_km < 85.0
        Te[d_mask] = T_meso + 35.0 * np.exp(-((self.h_km[d_mask] - 60.0) / 12.0) ** 2)
        return Te

    def _build_molecular_mass_profile(self) -> np.ndarray:
        mu_amu = 16.0 + (28.97 - 16.0) / (1.0 + np.exp((self.h_km - 150.0) / 25.0))
        return mu_amu * self.amu

    def _chapman_layer(self, hm_km: float, Nm: float, H_ref_km: float) -> np.ndarray:
        H_at_hm = float(np.interp(hm_km, self.h_km, self.H_km))
        H_layer = H_ref_km * (self.H_km / max(H_at_hm, 1e-3))
        z = (self.h_km - hm_km) / H_layer
        return Nm * np.exp(0.5 * (1.0 - z - np.exp(-z)))

    def _build_dynamic_chapman_profile(self) -> np.ndarray:
        if self.is_daytime:
            d = self._chapman_layer(76.0, 1.8e9, 7.5)
            e = self._chapman_layer(110.0, 1.4e11, 11.0)
            f1 = self._chapman_layer(185.0, 3.2e11, 28.0)
            f2 = self._chapman_layer(295.0, 1.35e12, 48.0)
            es = 4.5e11 * np.exp(-((self.h_km - 104.0) / 1.8) ** 2)
            return d + e + f1 + f2 + 0.15 * es
        return self._chapman_layer(112.0, 8.0e9, 12.0) + self._chapman_layer(310.0, 4.2e11, 52.0)

    def _build_collision_profile(self) -> np.ndarray:
        nu_en = 3.8e7 * np.exp(-(self.h_km - 60.0) / 6.4) * np.sqrt(self.Te / 200.0)
        nu_ei = 1.5e-6 * self.Ne * (self.Te ** (-1.5))
        return nu_en + nu_ei

    @staticmethod
    def _dingle_kinetic_factor(x_val: np.ndarray) -> np.ndarray:
        eps = np.clip(x_val, 1e-4, 1e4)
        return (1.0 + 0.52 / (1.0 + 0.28 * (eps**1.4))) / (1.0 + 0.15 / (1.0 + eps**2))

    def _appleton_hartree_mode(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, theta_mag: float, sign: float
    ):
        Y_L = Y * np.cos(theta_mag)
        Y_T = Y * np.sin(theta_mag)
        denom = 1.0 - X - 1j * Z
        term_T = (Y_T**2) / (2.0 * denom)
        rad = np.sqrt(((Y_T**4) / (4.0 * (denom**2)) + (Y_L**2)).astype(np.complex128))
        n = np.sqrt(1.0 - X / (1.0 - 1j * Z - term_T + sign * rad))
        return np.maximum(np.real(n), 0.05), np.abs(np.imag(n))

    def compute_attenuation_models(
        self, f_carrier_hz: float, elevation_deg: float = 18.0
    ) -> Dict[str, float]:
        omega = 2.0 * np.pi * f_carrier_hz
        omega_p2 = (self.Ne * self.e**2) / (self.eps0 * self.m_e)
        elev_rad = np.radians(elevation_deg)
        snell_const = (self.cfg.R_earth_m + self.cfg.alt_m) * np.cos(elev_rad)
        cutoff = snell_const / self.r_m

        mu_sq = 1.0 - (self.fp_hz / f_carrier_hz) ** 2
        valid_refl = np.where(mu_sq <= (cutoff**2))[0]

        if len(valid_refl) == 0:
            penetrates = True
            h_max_idx = len(self.h_km) - 1
            refl_height_km = float("nan")
            layer_name = "UZAYA KAÇIŞ"
            hop_dist_km = float("nan")
        else:
            penetrates = False
            h_max_idx = int(valid_refl[0])
            refl_height_km = float(self.h_km[h_max_idx])
            if refl_height_km < 92.0:
                layer_name = "D"
            elif refl_height_km < 145.0:
                layer_name = "E / Es"
            elif refl_height_km < 225.0:
                layer_name = "F1"
            else:
                layer_name = "F2"

            sin_psi_refl = np.clip(snell_const / (self.cfg.R_earth_m + refl_height_km * 1e3), -1.0, 1.0)
            gamma = np.radians(90.0) - elev_rad - np.arcsin(sin_psi_refl)
            hop_dist_km = float(2.0 * self.cfg.R_earth_m * gamma / 1e3)

        X_all = (self.fp_hz / f_carrier_hz) ** 2
        Y_all = self.f_H / f_carrier_hz
        mu_X_sq = 1.0 - X_all / np.maximum(1.0 - Y_all, 0.01)
        valid_refl_X = np.where((X_all >= (1.0 - Y_all)) | (mu_X_sq <= (cutoff**2)))[0]
        h_max_idx_X = int(valid_refl_X[0]) if len(valid_refl_X) > 0 else h_max_idx

        sl_O = slice(0, max(h_max_idx, 2))
        sl_X = slice(0, max(h_max_idx_X, 2))
        theta_mag = np.radians(abs(self.cfg.dip_angle_deg - elevation_deg))

        X_O = np.clip(omega_p2[sl_O] / (omega**2), 0.0, 0.99)
        Z_O = self.nu[sl_O] / omega
        Y_O = self.omega_H[sl_O] / omega
        mu_O, imag_O = self._appleton_hartree_mode(X_O, Y_O, Z_O, theta_mag, +1.0)
        alpha_O = (omega / self.c) * imag_O

        X_X = np.clip(omega_p2[sl_X] / (omega**2), 0.0, 0.99)
        Z_X = self.nu[sl_X] / omega
        Y_X = self.omega_H[sl_X] / omega
        mu_X, imag_X = self._appleton_hartree_mode(X_X, Y_X, Z_X, theta_mag, -1.0)
        alpha_X = (omega / self.c) * imag_X

        alpha_approx = (
            (self.e**2) / (2.0 * self.eps0 * self.m_e * self.c)
            * (self.Ne[sl_O] * self.nu[sl_O]) / (omega**2 + self.nu[sl_O]**2)
        )
        alpha_sw_O = alpha_O * self._dingle_kinetic_factor(omega / (self.nu[sl_O] + 1e-12))

        ds_dh_O = 1.0 / np.sqrt(np.maximum(1.0 - (snell_const / (self.r_m[sl_O] * mu_O)) ** 2, 0.015))
        ds_dh_X = 1.0 / np.sqrt(np.maximum(1.0 - (snell_const / (self.r_m[sl_X] * mu_X)) ** 2, 0.015))
        flat_ds_dh = 1.0 / np.sin(elev_rad)
        db_factor = 2.0 * 8.6858896

        loss_approx_db = integrate_trapz(alpha_approx * flat_ds_dh, self.h_m[sl_O]) * db_factor
        loss_O_db = integrate_trapz(alpha_O * ds_dh_O, self.h_m[sl_O]) * db_factor
        loss_X_db = integrate_trapz(alpha_X * ds_dh_X, self.h_m[sl_X]) * db_factor
        loss_sw_O_db = integrate_trapz(alpha_sw_O * ds_dh_O, self.h_m[sl_O]) * db_factor

        return {
            "freq_mhz": f_carrier_hz / 1e6,
            "penetrates": penetrates,
            "layer": layer_name,
            "refl_height_km": refl_height_km,
            "hop_dist_km": hop_dist_km,
            "loss_approx_db": loss_approx_db,
            "loss_O_db": loss_O_db,
            "loss_X_db": loss_X_db,
            "loss_sen_wyller_db": loss_sw_O_db,
            "discrepancy_O_db": abs(loss_sw_O_db - loss_approx_db),
            "discrepancy_X_db": abs(loss_X_db - loss_approx_db),
        }
