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
    phase_path_km: np.ndarray = None
    group_path_km: np.ndarray = None
    absorption_loss_db: np.ndarray = None


class HaselgroveRayTracer:
    """
    Kayseri merkezli küresel iyonosferde Haselgrove 2B diferansiyel ışın denklemlerini çözer:
      dr/ds = cos(psi)
      dtheta/ds = sin(psi) / r
      dpsi/ds = - (sin(psi) / r) - (1/mu) * (dmu/dr * sin(psi) - (1/r) * dmu/dtheta * cos(psi))
      dP/ds = mu
      dP'/ds = mu'
      dA/ds = alpha
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

        # Merkezi fark türevi dmu/dr (dh = 0.5 km)
        dh = 0.5
        h_plus = min(h_km + dh, 449.0)
        h_minus = max(h_km - dh, 60.0)
        fp_plus = float(np.interp(h_plus, self.h_km_grid, self.fp_hz_grid))
        fp_minus = float(np.interp(h_minus, self.h_km_grid, self.fp_hz_grid))
        mu_plus = float(np.sqrt(max(1.0 - (fp_plus / f_hz) ** 2, 0.001)))
        mu_minus = float(np.sqrt(max(1.0 - (fp_minus / f_hz) ** 2, 0.001)))
        delta_h = h_plus - h_minus
        dmu_dr = (mu_plus - mu_minus) / delta_h if delta_h > 0 else 0.0

        return mu, dmu_dr

    def _get_absorption(self, f_hz: float, h_km: float, mu: float) -> float:
        """Yerel irtifa ve frekans için magneto-iyonik emilim kaybı oranını (dB/km) hesaplar."""
        if h_km < 60.0 or h_km > 449.0:
            return 0.0
        ne = float(np.interp(h_km, self.plasma.h_km, self.plasma.Ne))
        nu = float(np.interp(h_km, self.plasma.h_km, self.plasma.nu))
        omega = 2.0 * np.pi * f_hz
        mu_safe = max(mu, 0.05)
        # alpha [Np/m] = (e^2 / (2 * eps0 * m_e * c)) * (Ne * nu) / (mu * (omega^2 + nu^2))
        const_factor = (self.plasma.e**2) / (2.0 * self.plasma.eps0 * self.plasma.m_e * self.plasma.c)
        alpha_np_m = const_factor * (ne * nu) / (mu_safe * (omega**2 + nu**2))
        # dB/km = alpha_np_m * 1000 m/km * 8.6858896 dB/Np
        return float(alpha_np_m * 1000.0 * 8.6858896)

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
        cos_elev = np.cos(elev_rad)

        # Negatif veya ufuk altı kalkış açısı koruması:
        # Yer altına giren ışınlar derhal sonlandırılır, sahte yansıma oluşturulmaz.
        if elevation_deg <= 0.0:
            return RayTrajectory(
                freq_mhz=freq_mhz,
                elevation_deg=elevation_deg,
                r_km=np.array([self.R_E]),
                h_km=np.array([0.0]),
                theta_rad=np.array([0.0]),
                ground_dist_km=np.array([0.0]),
                reflected=False,
                apex_height_km=0.0,
                ground_hit_km=0.0,
                phase_path_km=np.array([0.0]),
                group_path_km=np.array([0.0]),
                absorption_loss_db=np.array([0.0]),
            )

        # Başlangıç koşulları (Yeryüzü: r = R_E, theta = 0)
        r = self.R_E
        theta = 0.0
        # psi: Işının yerel düşey ile yaptığı açı (Zenith açısı = 90 - elevation)
        psi = np.pi / 2.0 - elev_rad

        # Faz yolu, grup yolu ve kümülatif emilim entegrasyonu
        P = 0.0
        P_prime = 0.0
        A_loss = 0.0

        r_history = [r]
        theta_history = [theta]
        P_history = [P]
        P_prime_history = [P_prime]
        A_history = [A_loss]

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
                r_history[-1] = self.R_E  # Son iniş noktasını yer seviyesine sabitle
                break

            # Uzaya kaçış kontrolü (> 430 km ve yukarı doğru gidiyorsa)
            if h_curr > 430.0 and np.cos(psi) > 0:
                break

            # Dik ve dik-yakın açılarda Haselgrove Bouguer dönüş noktası kontrolü:
            # Yön yukarı doğru (cos(psi) > 0) iken iyonosfer içinde (h >= 60 km) plazma kesme
            # seviyesine (X >= 1.0 veya X >= X_turn) ulaşıldığında ışın fiziksel olarak yansır.
            if h_curr >= 60.0 and np.cos(psi) > 0:
                fp_curr = float(np.interp(h_curr, self.h_km_grid, self.fp_hz_grid))
                X_curr = (fp_curr / f_hz) ** 2
                X_turn = 1.0 - ((self.R_E * cos_elev) / r) ** 2
                if X_curr >= X_turn or X_curr >= 1.0:
                    psi = np.pi - psi

            # RK4 Diferansiyel Adımı
            # Durum vektörü: y = [r, theta, psi, P, P_prime, A]
            def derivatives(r_val, theta_val, psi_val):
                h_val = r_val - self.R_E
                mu_val, dmu_dr = self._get_mu(f_hz, h_val)
                dr = np.cos(psi_val)
                dtheta = np.sin(psi_val) / r_val
                # Snell / Haselgrove kırılma açısı değişimi
                dpsi = - (np.sin(psi_val) / r_val) - (dmu_dr / mu_val) * np.sin(psi_val)
                dP = mu_val
                dP_prime = 1.0 / max(mu_val, 0.05)
                dA = self._get_absorption(f_hz, h_val, mu_val)
                return dr, dtheta, dpsi, dP, dP_prime, dA

            k1_r, k1_th, k1_psi, k1_P, k1_Pp, k1_A = derivatives(r, theta, psi)
            k2_r, k2_th, k2_psi, k2_P, k2_Pp, k2_A = derivatives(
                r + 0.5 * step_km * k1_r,
                theta + 0.5 * step_km * k1_th,
                psi + 0.5 * step_km * k1_psi,
            )
            k3_r, k3_th, k3_psi, k3_P, k3_Pp, k3_A = derivatives(
                r + 0.5 * step_km * k2_r,
                theta + 0.5 * step_km * k2_th,
                psi + 0.5 * step_km * k2_psi,
            )
            k4_r, k4_th, k4_psi, k4_P, k4_Pp, k4_A = derivatives(
                r + step_km * k3_r,
                theta + step_km * k3_th,
                psi + step_km * k3_psi,
            )

            r += (step_km / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
            theta += (step_km / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
            psi += (step_km / 6.0) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
            P += (step_km / 6.0) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)
            P_prime += (step_km / 6.0) * (k1_Pp + 2 * k2_Pp + 2 * k3_Pp + k4_Pp)
            A_loss += (step_km / 6.0) * (k1_A + 2 * k2_A + 2 * k3_A + k4_A)

            r_history.append(r)
            theta_history.append(theta)
            P_history.append(P)
            P_prime_history.append(P_prime)
            A_history.append(A_loss)

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
            phase_path_km=np.array(P_history),
            group_path_km=np.array(P_prime_history),
            absorption_loss_db=np.array(A_history),
        )

    def trace_multihop_fan(
        self, freq_mhz: float, elevations: List[float] = [8.0, 15.0, 22.0, 32.0]
    ) -> List[RayTrajectory]:
        """Farklı kalkış açılarında gökyüzü dalgası ışın yelpazesini hesaplar."""
        return [self.trace_ray(freq_mhz, elev) for elev in elevations]
