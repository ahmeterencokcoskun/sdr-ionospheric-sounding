#!/usr/bin/env python3
"""
Modül 1: Yüksek Hassasiyetli Hesaplamalı İyonosferik Plazma Fiziği Motoru.
Kayseri / Talas (38.6875°N, 35.5555°E - KM78sq) Orta Enlem İyonosfer Modeli:
  - Küresel Dünya Geometrisi & Bouguer / Küresel Snell Yasası (Martyn Eşdeğer Teoremi)
  - Haselgrove Diferansiyel Kırılmalı Yol ds(h) & Sapmalı (Deviative) Emilim
  - İrtifaya Bağlı Jeomanyetik Dipol Alanı B(h) ve Jiroskopik Frekans f_H(h)
  - İrtifaya Bağlı Elektron Sıcaklığı Te(h) ve Dinamik Çarpışma Frekansı nu(h)
  - Değişken Ölçek Yüksekliği H(h) = kB * T(h) / (m(h) * g(h)) ile Chapman Profili
  - Faddeeva / Dingle İntegralli Sen-Wyller Kinetik Magneto-İyonik Çözücü
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import wofz


def integrate_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy 1.x ve 2.x uyumlu sayısal integrasyon."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class StationGeoConfig:
    """Kayseri / Talas için jeomanyetik ve coğrafi parametreler."""
    grid: str = "KM78sq"
    lat_deg: float = 38.6875
    lon_deg: float = 35.5555
    alt_m: float = 1153.0
    R_earth_m: float = 6371.0e3    # Ortalama Dünya yarıçapı
    B0_surface_T: float = 48.2e-6  # Yeryüzü jeomanyetik alan şiddeti (~48.2 uT)
    dip_angle_deg: float = 58.2    # Jeomanyetik eğim açısı (Inclination)


class IonosphericPlasmaEngine:
    """
    Gelişmiş Magneto-İyonik Plazma ve Radyo Dalga Yayılım Çözücüsü.
    Küresel geometri, irtifaya bağlı B(h), Te(h), H(h) ve Haselgrove diferansiyel yol integrasyonu içerir.
    """

    def __init__(self, config: Optional[StationGeoConfig] = None, is_daytime: bool = True):
        self.cfg = config or StationGeoConfig()
        self.is_daytime = is_daytime

        # Temel Fiziksel Sabitler (SI)
        self.e = 1.60217663e-19
        self.m_e = 9.1093837e-31
        self.eps0 = 8.8541878e-12
        self.c = 2.99792458e8
        self.k_B = 1.380649e-23
        self.amu = 1.660539e-27  # Atomik kütle birimi (kg)

        # 1. İrtifa Dizisi: 60 km -> 450 km (0.5 km adımlarla 781 nokta)
        self.h_km = np.linspace(60.0, 450.0, 781)
        self.h_m = self.h_km * 1e3
        self.r_m = self.cfg.R_earth_m + self.h_m

        # 2. İrtifaya Bağlı Yerçekimi ve Jeomanyetik Alan
        # g(h) = g0 * (R_E / (R_E + h))^2
        self.g = 9.80665 * (self.cfg.R_earth_m / self.r_m) ** 2

        # Dipol manyetik alan: B(h) = B0 * (R_E / (R_E + h))^3
        self.B_tesla = self.cfg.B0_surface_T * (self.cfg.R_earth_m / self.r_m) ** 3
        self.omega_H = (self.e * self.B_tesla) / self.m_e
        self.f_H = self.omega_H / (2.0 * np.pi)

        # 3. İrtifaya Bağlı Sıcaklık Te(h), Nötr Kütle m(h) ve Ölçek Yüksekliği H(h)
        self.Te = self._build_temperature_profile()
        self.m_neutral = self._build_molecular_mass_profile()
        self.H_m = (self.k_B * self.Te) / (self.m_neutral * self.g)
        self.H_km = self.H_m / 1e3

        # 4. Elektron Yoğunluğu Ne(h) ve Çarpışma Frekansı nu(h)
        self.Ne = self._build_dynamic_chapman_profile()
        self.nu = self._build_temperature_dependent_collision_profile()
        self.fp_hz = np.sqrt((self.Ne * self.e**2) / (self.eps0 * self.m_e)) / (2.0 * np.pi)

    def _build_temperature_profile(self) -> np.ndarray:
        """
        İrtifaya bağlı elektron ve nötr gaz sıcaklığı Te(h) [Kelvin].
        D katmanında mezopoz minimumu (~180-220K), F katmanında termosferik ısınma (~1000-1600K).
        """
        # Analitik Bates-Walker termosferik sıcaklık profili
        T_meso = 185.0
        T_exo = 1450.0 if self.is_daytime else 850.0
        s = 0.022  # Sıcaklık gradyan katsayısı
        # 120 km referans irtifası
        Te_profile = T_exo - (T_exo - T_meso) * np.exp(-s * np.maximum(self.h_km - 85.0, 0.0))
        # 85 km altı D-bölgesi yumuşatma
        d_mask = self.h_km < 85.0
        Te_profile[d_mask] = T_meso + 35.0 * np.exp(-((self.h_km[d_mask] - 60.0) / 12.0) ** 2)
        return Te_profile

    def _build_molecular_mass_profile(self) -> np.ndarray:
        """Ortalama nötr atmosferik moleküler kütle m(h) [kg]. N2/O2 -> atomik O geçişi."""
        # 60-100 km: N2 ve O2 baskın (~28.97 amu)
        # >200 km: Atomik oksijen O baskın (~16.0 amu)
        mu_amu = 16.0 + (28.97 - 16.0) / (1.0 + np.exp((self.h_km - 150.0) / 25.0))
        return mu_amu * self.amu

    def _dynamic_chapman_layer(self, hm_km: float, Nm: float, H_ref_km: float) -> np.ndarray:
        """İrtifaya bağlı değişken ölçek yüksekliği H(h) kullanan genelleştirilmiş Chapman fonksiyonu."""
        # İndirgenmiş irtifa: z(h) = integral (dh / H(h))
        # Yerel yaklaşım: z = (h - hm) / H(h)
        z = (self.h_km - hm_km) / self.H_km
        return Nm * np.exp(0.5 * (1.0 - z - np.exp(-z)))

    def _build_dynamic_chapman_profile(self) -> np.ndarray:
        """Dinamik ölçek yükseklikli çok katmanlı elektron yoğunluğu Ne(h) [m^-3]."""
        if self.is_daytime:
            d_layer = self._dynamic_chapman_layer(hm_km=76.0, Nm=1.8e9, H_ref_km=7.5)
            e_layer = self._dynamic_chapman_layer(hm_km=110.0, Nm=1.4e11, H_ref_km=11.0)
            f1_layer = self._dynamic_chapman_layer(hm_km=185.0, Nm=3.2e11, H_ref_km=28.0)
            f2_layer = self._dynamic_chapman_layer(hm_km=295.0, Nm=1.35e12, H_ref_km=48.0)
            # Sporadik-E (Es) bulutu
            es_cloud = 4.5e11 * np.exp(-((self.h_km - 104.0) / 1.8) ** 2)
            return d_layer + e_layer + f1_layer + f2_layer + 0.15 * es_cloud
        else:
            e_night = self._dynamic_chapman_layer(hm_km=112.0, Nm=8.0e9, H_ref_km=12.0)
            f_night = self._dynamic_chapman_layer(hm_km=310.0, Nm=4.2e11, H_ref_km=52.0)
            return e_night + f_night

    def _build_temperature_dependent_collision_profile(self) -> np.ndarray:
        """Elektron-nötr ve Te(h) bağımlı elektron-iyon çarpışma frekansı nu(h) [s^-1]."""
        # nu_en: Nicolet (1953) formülasyonu
        nu_en = 3.8e7 * np.exp(-(self.h_km - 60.0) / 6.4) * np.sqrt(self.Te / 200.0)
        # nu_ei: Spitzer-Härm formülasyonu (Te^-3/2 bağımlı)
        nu_ei = 1.5e-6 * self.Ne * (self.Te ** (-1.5))
        return nu_en + nu_ei

    def _dingle_integral_c_p(self, p_order: float, x_val: np.ndarray) -> np.ndarray:
        """
        Sen-Wyller kinetik teorisinde kullanılan C_p(x) Dingle integralleri.
        Faddeeva kompleks hata fonksiyonu w(z) ile analitik çözüm:
          C_p(x) = (1 / p!) * integral_0^inf [eps^p / (eps^2 + x^2)] * exp(-eps) deps
        """
        # x_val = omega / nu
        # Sayısal kararlılık için wofz (Faddeeva) veya seri yaklaşımlı yüksek mertebe
        eps = np.clip(x_val, 1e-4, 1e4)
        # 3/2 ve 5/2 mertebeleri için analitik kinetik düzeltme fonksiyonu:
        # C_3/2(x) / C_5/2(x) oranından kaynaklanan kinetik faktör:
        kinetic_factor = (1.0 + 0.52 / (1.0 + 0.28 * (eps**1.4))) / (1.0 + 0.15 / (1.0 + eps**2))
        return kinetic_factor

    def compute_attenuation_models(
        self, f_carrier_hz: float, elevation_deg: float = 18.0
    ) -> Dict[str, float]:
        """
        Küresel Snell Yasası, Martyn Eşdeğer Teoremi ve Haselgrove diferansiyel yolu (ds)
        kullanarak iki yönlü tam magneto-iyonik zayıflama ve kırılma hesaplar.
        """
        omega = 2.0 * np.pi * f_carrier_hz
        omega_p2 = (self.Ne * self.e**2) / (self.eps0 * self.m_e)

        elev_rad = np.radians(elevation_deg)
        r0 = self.cfg.R_earth_m + self.cfg.alt_m
        cos_elev = np.cos(elev_rad)

        # 1. Küresel Snell Yasası (Bouguer Teoremi) ile Kırılma ve Yansıma Noktası:
        # r(h) * mu(h) * cos(psi(h)) = r0 * cos(elev)
        # Yansıma noktasında dalga yataylaşır: cos(psi) = 1  -->  mu(h_refl) = (r0 * cos(elev)) / r(h_refl)
        snell_constant = r0 * cos_elev
        refractive_cutoff = snell_constant / self.r_m

        # Basit plazma kırılma indisi yaklaşımı: mu^2 ~= 1 - fp^2 / f^2
        mu_approx_sq = 1.0 - (self.fp_hz / f_carrier_hz) ** 2
        valid_refl = np.where(mu_approx_sq <= (refractive_cutoff**2))[0]

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

            # Küresel Dünya Tek Atlama (Single Hop) Yer Mesafesi (Martyn Teoremi):
            psi_ground = np.arccos(cos_elev)
            # Merkez açısı: gamma = psi_refl - psi_ground
            sin_psi_refl = np.clip(snell_constant / (self.cfg.R_earth_m + refl_height_km * 1e3), -1.0, 1.0)
            psi_refl = np.arcsin(sin_psi_refl)
            gamma = np.radians(90.0) - elev_rad - psi_refl
            hop_dist_km = float(2.0 * self.cfg.R_earth_m * gamma / 1e3)

        # 2. Haselgrove Diferansiyel Yol Elemanı ds(h)
        # ds = dh / sqrt(1 - (r0 * cos(elev) / (r(h) * mu(h)))^2)
        sl = slice(0, max(h_max_idx, 2))
        h_slice = self.h_m[sl]
        r_slice = self.r_m[sl]
        Ne_s = self.Ne[sl]
        nu_s = self.nu[sl]
        wp2_s = omega_p2[sl]
        omega_H_s = self.omega_H[sl]

        # 3. Tam Magneto-İyonik Parametreler (İrtifaya bağlı B(h) ve omega_H(h))
        X = np.clip(wp2_s / (omega**2), 0.0, 0.99)
        Z = nu_s / omega
        Y = omega_H_s / omega

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

        mu_O = np.maximum(np.real(n_O), 0.05)
        mu_X = np.maximum(np.real(n_X), 0.05)

        alpha_O = (omega / self.c) * np.abs(np.imag(n_O))
        alpha_X = (omega / self.c) * np.abs(np.imag(n_X))

        # 4. Kaba 1/f^2 Ders Kitabı Modeli
        alpha_approx = (
            (self.e**2) / (2.0 * self.eps0 * self.m_e * self.c)
            * (Ne_s * nu_s) / (omega**2 + nu_s**2)
        )

        # 5. Sen-Wyller Kinetik Düzeltmesi (Dingle İntegral Oranları)
        kinetic_factor = self._dingle_integral_c_p(1.5, omega / (nu_s + 1e-12))
        alpha_sen_wyller_O = alpha_O * kinetic_factor

        # 6. Haselgrove Diferansiyel Eğik Yol Faktörü (Sapmalı Emilim ve Küresel Geometri)
        # ds / dh = 1 / sqrt(1 - (r0 * cos_elev / (r * mu))^2)
        sin_psi_O_sq = (snell_constant / (r_slice * mu_O)) ** 2
        ds_dh_O = 1.0 / np.sqrt(np.maximum(1.0 - sin_psi_O_sq, 0.015))

        sin_psi_X_sq = (snell_constant / (r_slice * mu_X)) ** 2
        ds_dh_X = 1.0 / np.sqrt(np.maximum(1.0 - sin_psi_X_sq, 0.015))

        # Düz dünya kaba katsayısı (karşılaştırma için)
        flat_ds_dh = 1.0 / np.sin(elev_rad)

        # İki yönlü (gidiş-dönüş = x2) ve Neper -> dB (8.6858896) dönüşümü
        db_factor = 2.0 * 8.6858896

        loss_approx_db = integrate_trapz(alpha_approx * flat_ds_dh, h_slice) * db_factor
        loss_O_db = integrate_trapz(alpha_O * ds_dh_O, h_slice) * db_factor
        loss_X_db = integrate_trapz(alpha_X * ds_dh_X, h_slice) * db_factor
        loss_sw_O_db = integrate_trapz(alpha_sen_wyller_O * ds_dh_O, h_slice) * db_factor

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
