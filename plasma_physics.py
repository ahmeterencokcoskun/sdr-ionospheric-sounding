#!/usr/bin/env python3
"""
Modül 1: Hesaplamalı İyonosferik Plazma Fiziği ve Zayıflama (Alpha) Hata Bütçesi Motoru
Kayseri / Talas (38.72°N, 35.48°E - KM78) Orta Enlem Parametreleri
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


def integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 1.x ve 2.x uyumlu yamuk integrasyon fonksiyonu."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class StationGeoConfig:
    callsign: str = "TA7CAE"
    grid: str = "KM78"
    lat_deg: float = 38.72
    lon_deg: float = 35.48
    alt_m: float = 1100.0
    B0_tesla: float = 48.2e-6      # Kayseri toplam jeomanyetik alan şiddeti (~48.2 uT)
    dip_angle_deg: float = 56.4    # Manyetik eğim açısı (Inclination)


class IonosphericPlasmaEngine:
    """
    Chapman Çok-Katmanlı Elektron Yoğunluğu, Tam Magneto-İyonik Appleton-Hartree,
    Sen-Wyller Kinetik Düzeltmesi ve 2B Eğik Işın İzleme (Ray-Tracing) Çözücüsü.
    """

    def __init__(self, config: Optional[StationGeoConfig] = None, is_daytime: bool = True):
        self.cfg = config or StationGeoConfig()
        self.is_daytime = is_daytime

        # Temel Fiziksel Sabitler (SI)
        self.e = 1.60217663e-19
        self.m_e = 9.1093837e-31
        self.eps0 = 8.8541878e-12
        self.c = 2.99792458e8

        # Jiroskop (Cyclotron) Frekansı: f_H = e*B0 / (2*pi*m_e)
        self.omega_H = (self.e * self.cfg.B0_tesla) / self.m_e
        self.f_H = self.omega_H / (2.0 * np.pi)

        # İrtifa Vektörü: 60 km -> 450 km (0.5 km çözünürlük)
        self.h_km = np.linspace(60.0, 450.0, 781)
        self.h_m = self.h_km * 1e3

        self.Ne = self._build_chapman_profile()
        self.nu = self._build_collision_profile()
        self.fp_hz = np.sqrt((self.Ne * self.e**2) / (self.eps0 * self.m_e)) / (2.0 * np.pi)

    def _chapman_layer(self, hm_km: float, Nm: float, H_km: float) -> np.ndarray:
        z = (self.h_km - hm_km) / H_km
        return Nm * np.exp(0.5 * (1.0 - z - np.exp(-z)))

    def _build_chapman_profile(self) -> np.ndarray:
        """Gündüz (D, E, F1, F2) veya Gece (E, F) Chapman elektron yoğunluğu [m^-3]."""
        if self.is_daytime:
            d_layer = self._chapman_layer(hm_km=76.0, Nm=1.8e9, H_km=7.5)
            e_layer = self._chapman_layer(hm_km=110.0, Nm=1.4e11, H_km=11.0)
            f1_layer = self._chapman_layer(hm_km=185.0, Nm=3.2e11, H_km=28.0)
            f2_layer = self._chapman_layer(hm_km=295.0, Nm=1.35e12, H_km=48.0)
            # Anormal Sporadik-E (Es) ince bulut katmanı (104 km)
            es_cloud = 4.5e11 * np.exp(-((self.h_km - 104.0) / 1.8) ** 2)
            return d_layer + e_layer + f1_layer + f2_layer + 0.15 * es_cloud
        else:
            e_night = self._chapman_layer(hm_km=112.0, Nm=8.0e9, H_km=12.0)
            f_night = self._chapman_layer(hm_km=310.0, Nm=4.2e11, H_km=52.0)
            return e_night + f_night

    def _build_collision_profile(self) -> np.ndarray:
        """Elektron-nötr ve elektron-iyon toplam etkin çarpışma frekansı nu(h) [s^-1]."""
        nu_en = 3.8e7 * np.exp(-(self.h_km - 60.0) / 6.4)
        nu_ei = 1.5e-6 * self.Ne * (1000.0 ** (-1.5))
        return nu_en + nu_ei

    def compute_attenuation_models(
        self, f_carrier_hz: float, elevation_deg: float = 18.0
    ) -> Dict[str, float]:
        """
        Üç farklı teorik modelle iki yönlü (gidiş-dönüş) iyonosferik zayıflama (dB)
        ve kırılma irtifasını hesaplar:
          1. Kaba İzotropik 1/f^2 Yaklaşımı (Non-deviative, B0=0, mu~1)
          2. Tam Magneto-İyonik Appleton-Hartree (O-Modu ve X-Modu)
          3. Sen-Wyller Kinetik Termal Dağılım Düzeltmeli Model
        """
        omega = 2.0 * np.pi * f_carrier_hz
        omega_p2 = (self.Ne * self.e**2) / (self.eps0 * self.m_e)

        elev_rad = np.radians(elevation_deg)
        sin_elev = np.sin(elev_rad)

        # Eğik insidans kırılma şartı: fp(h) >= f_carrier * sin(elevation)
        req_fp = f_carrier_hz * sin_elev
        refl_idx = np.where(self.fp_hz >= req_fp)[0]

        if len(refl_idx) == 0:
            penetrates = True
            h_max_idx = len(self.h_km) - 1
            refl_height_km = float("nan")
            layer_name = "UZAYA KAÇIŞ"
            hop_dist_km = float("nan")
        else:
            penetrates = False
            h_max_idx = int(refl_idx[0])
            refl_height_km = float(self.h_km[h_max_idx])
            if refl_height_km < 92.0:
                layer_name = "D"
            elif refl_height_km < 145.0:
                layer_name = "E / Es"
            elif refl_height_km < 225.0:
                layer_name = "F1"
            else:
                layer_name = "F2"
            hop_dist_km = float(2.0 * refl_height_km / np.tan(elev_rad))

        # İntegrasyon bölgesi: 60 km'den yansıma irtifasının hemen altına kadar
        sl = slice(0, max(h_max_idx, 2))
        h_slice = self.h_m[sl]
        Ne_s = self.Ne[sl]
        nu_s = self.nu[sl]
        wp2_s = omega_p2[sl]

        # 1. Model A: Kaba Ders Kitabı Yaklaşımı (Non-deviative 1/f^2)
        alpha_approx = (
            (self.e**2) / (2.0 * self.eps0 * self.m_e * self.c)
            * (Ne_s * nu_s) / (omega**2 + nu_s**2)
        )

        # 2. Model B: Tam Magneto-İyonik Appleton-Hartree Denklemi
        X = np.clip(wp2_s / (omega**2), 0.0, 0.96)
        Z = nu_s / omega
        Y = self.omega_H / omega

        # Dalga vektörü ile Kayseri jeomanyetik eğim açısı arasındaki açı
        theta_mag = np.radians(abs(self.cfg.dip_angle_deg - elevation_deg))
        Y_L = Y * np.cos(theta_mag)
        Y_T = Y * np.sin(theta_mag)

        denom_base = 1.0 - X - 1j * Z
        term_T = (Y_T**2) / (2.0 * denom_base)
        radicand = (Y_T**4) / (4.0 * (denom_base**2)) + (Y_L**2)
        sqrt_term = np.sqrt(radicand.astype(np.complex128))

        n2_O = 1.0 - X / (1.0 - 1j * Z - term_T + sqrt_term)
        n2_X = 1.0 - X / (1.0 - 1j * Z - term_T - sqrt_term)

        n_O = np.sqrt(n2_O)
        n_X = np.sqrt(n2_X)

        alpha_O = (omega / self.c) * np.abs(np.imag(n_O))
        alpha_X = (omega / self.c) * np.abs(np.imag(n_X))

        # 3. Model C: Sen-Wyller Kinetik Termal Dağılım Düzeltmesi
        # D-tabakasında (Z'nin anlamlı olduğu bölgede) Maxwell-Boltzmann hız
        # dağılımı ve v^2 çarpışma kesiti etkisi (Dingle integrali parametrizasyonu)
        kinetic_factor = 1.0 + 0.48 / (1.0 + (Z * 2.5) ** 2)
        alpha_sen_wyller_O = alpha_O * kinetic_factor

        # Eğik yol çarpanı (2 / sin(elev)) ve Neper -> dB dönüşümü (8.685889)
        path_factor = (2.0 / sin_elev) * 8.6858896

        loss_approx_db = integrate_trapz(alpha_approx, h_slice) * path_factor
        loss_O_db = integrate_trapz(alpha_O, h_slice) * path_factor
        loss_X_db = integrate_trapz(alpha_X, h_slice) * path_factor
        loss_sw_O_db = integrate_trapz(alpha_sen_wyller_O, h_slice) * path_factor

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
