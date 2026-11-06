#!/usr/bin/env python3
"""
Modül: 2B Haselgrove Sayısal Işın İzleyici (Ray-Tracer).
Küresel Dünya koordinatlarında (r, theta) ve Appleton-Hartree plazma kırılma indisi mu(r) altında
Runge-Kutta 4. Mertebe (RK4) integratörü ile HF gökyüzü dalgası yayılım yörüngelerini hesaplar.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine


@dataclass
class RayTrajectory:
    freq_mhz: float
    elevation_deg: float
    r_km: np.ndarray       # Merkezden uzaklık (km)
    h_km: np.ndarray       # Yer yüzeyinden irtifa (km)
    theta_rad: np.ndarray  # Açısal menzil (radyan)
    ground_dist_km: np.ndarray  # Yer yüzeyi mesafesi (km)
    reflected: bool
    apex_height_km: float
    ground_hit_km: float


class HaselgroveRayTracer:
    """
    Kayseri merkezli küresel iyonosferde Haselgrove 2B diferansiyel ışın denklemlerini çözer:
      dr/ds = sin(psi)
      dtheta/ds = cos(psi) / r
      dpsi/ds = - (cos(psi)/r) - (1/mu) * (dmu/dr * cos(psi) - (1/r) * dmu/dtheta * sin(psi))
    """

    def __init__(self, plasma_engine: IonosphericPlasmaEngine):
        self.plasma = plasma_engine
        self.R_E = self.plasma.cfg.R_earth_m / 1e3  # km cinsinden Dünya yarıçapı
        self.h_km_grid = self.plasma.h_km
        self.fp_hz_grid = self.plasma.fp_hz

    def _get_mu(self, f_hz: float, h_km: float) -> Tuple[float, float]:
        """Yerel irtifada kırılma indisi mu ve dmu/dr türevini hesaplar."""
        if h_km < 60.0:
            return 1.0, 0.0
        if h_km > 449.0:
            h_km = 449.0

        # İrtifaya karşılık gelen plazma frekansı
        fp = float(np.interp(h_km, self.h_km_grid, self.fp_hz_grid))
        X = (fp / f_hz) ** 2
        if X >= 1.0:
            return 0.01, 0.0

        mu = float(np.sqrt(max(1.0 - X, 0.001)))

        # Sayısal türev dmu/dr (dh = 0.5 km)
        dh = 0.5
        fp_plus = float(np.interp(h_km + dh, self.h_km_grid, self.fp_hz_grid))
        X_plus = (fp_plus / f_hz) ** 2
        mu_plus = float(np.sqrt(max(1.0 - X_plus, 0.001)))
        dmu_dr = (mu_plus - mu) / dh

        return mu, dmu_dr

    def trace_ray(
        self,
        freq_mhz: float,
        elevation_deg: float,
        step_km: float = 1.0,
        max_steps: int = 4000,
    ) -> RayTrajectory:
        """Belirtilen frekans ve kalkış açısı için tek bir ışını adım adım takip eder."""
        f_hz = freq_mhz * 1e6
        elev_rad = np.radians(elevation_deg)

        # Başlangıç koşulları (Yeryüzü: r = R_E, theta = 0)
        r = self.R_E
        theta = 0.0
        # psi: Işının yerel düşey ile yaptığı açı (Zenith açısı = 90 - elevation)
        psi = np.pi / 2.0 - elev_rad

        r_history = [r]
        theta_history = [theta]

        reflected = False
        apex_height = 0.0
        ground_hit = 0.0

        for step in range(max_steps):
            h_curr = r - self.R_E
            if h_curr > apex_height:
                apex_height = h_curr

            # Yeryüzüne geri dönüş kontrolü
            if step > 20 and r <= self.R_E:
                reflected = True
                ground_hit = theta * self.R_E
                break

            # Uzaya kaçış kontrolü (> 440 km ve yukarı doğru gidiyorsa)
            if h_curr > 430.0 and np.cos(psi) < 0:
                break

            # RK4 Diferansiyel Adımı
            # Durum vektörü: y = [r, theta, psi]
            def derivatives(r_val, theta_val, psi_val):
                h_val = r_val - self.R_E
                mu_val, dmu_dr = self._get_mu(f_hz, h_val)
                dr = np.cos(psi_val)
                dtheta = np.sin(psi_val) / r_val
                # Snell / Haselgrove kırılma açısı değişimi
                dpsi = - (np.sin(psi_val) / r_val) - (dmu_dr / mu_val) * np.sin(psi_val)
                return dr, dtheta, dpsi

            k1_r, k1_th, k1_psi = derivatives(r, theta, psi)
            k2_r, k2_th, k2_psi = derivatives(
                r + 0.5 * step_km * k1_r,
                theta + 0.5 * step_km * k1_th,
                psi + 0.5 * step_km * k1_psi,
            )
            k3_r, k3_th, k3_psi = derivatives(
                r + 0.5 * step_km * k2_r,
                theta + 0.5 * step_km * k2_th,
                psi + 0.5 * step_km * k2_psi,
            )
            k4_r, k4_th, k4_psi = derivatives(
                r + step_km * k3_r,
                theta + step_km * k3_th,
                psi + step_km * k3_psi,
            )

            r += (step_km / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
            theta += (step_km / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
            psi += (step_km / 6.0) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)

            r_history.append(r)
            theta_history.append(theta)

        r_arr = np.array(r_history)
        th_arr = np.array(theta_history)
        h_arr = r_arr - self.R_E
        dist_arr = th_arr * self.R_E

        return RayTrajectory(
            freq_mhz=freq_mhz,
            elevation_deg=elevation_deg,
            r_km=r_arr,
            h_km=h_arr,
            theta_rad=th_arr,
            ground_dist_km=dist_arr,
            reflected=reflected,
            apex_height_km=apex_height,
            ground_hit_km=ground_hit,
        )

    def trace_multihop_fan(
        self, freq_mhz: float, elevations: List[float] = [8.0, 15.0, 22.0, 32.0]
    ) -> List[RayTrajectory]:
        """Farklı kalkış açılarında gökyüzü dalgası ışın yelpazesini hesaplar."""
        return [self.trace_ray(freq_mhz, elev) for elev in elevations]
