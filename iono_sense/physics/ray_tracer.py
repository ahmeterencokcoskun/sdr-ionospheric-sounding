#!/usr/bin/env python3
"""2D Haselgrove Spherical Numerical Ray Tracer (RK4 Ray-Tracer)."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from iono_sense.physics.plasma_engine import IonosphericPlasmaEngine


@dataclass
class RayTrajectory:
    freq_mhz: float
    elevation_deg: float
    r_km: np.ndarray
    h_km: np.ndarray
    theta_rad: np.ndarray
    ground_dist_km: np.ndarray
    reflected: bool
    apex_height_km: float
    ground_hit_km: float
    phase_path_km: np.ndarray = None
    group_path_km: np.ndarray = None
    absorption_loss_db: np.ndarray = None


class HaselgroveRayTracer:
    """Solves Haselgrove 2D differential ray equations in spherical coordinates (r, theta)."""

    def __init__(self, plasma_engine: IonosphericPlasmaEngine):
        self.plasma = plasma_engine
        self.R_E = self.plasma.cfg.R_earth_m / 1e3
        self.h_km_grid = self.plasma.h_km
        self.fp_hz_grid = self.plasma.fp_hz

    def _get_mu(self, f_hz: float, h_km: float) -> Tuple[float, float]:
        if h_km < 60.0:
            return 1.0, 0.0
        h_km = min(h_km, 449.0)

        fp = float(np.interp(h_km, self.h_km_grid, self.fp_hz_grid))
        X = (fp / f_hz) ** 2
        if X >= 1.0:
            return 0.01, 0.0

        mu = float(np.sqrt(max(1.0 - X, 0.001)))
        h_plus = min(h_km + 0.5, 449.0)
        h_minus = max(h_km - 0.5, 60.0)
        fp_plus = float(np.interp(h_plus, self.h_km_grid, self.fp_hz_grid))
        fp_minus = float(np.interp(h_minus, self.h_km_grid, self.fp_hz_grid))
        mu_plus = float(np.sqrt(max(1.0 - (fp_plus / f_hz) ** 2, 0.001)))
        mu_minus = float(np.sqrt(max(1.0 - (fp_minus / f_hz) ** 2, 0.001)))
        delta_h = h_plus - h_minus
        dmu_dr = (mu_plus - mu_minus) / delta_h if delta_h > 0 else 0.0
        return mu, dmu_dr

    def _get_absorption(self, f_hz: float, h_km: float, mu: float) -> float:
        if h_km < 60.0 or h_km > 449.0:
            return 0.0
        ne = float(np.interp(h_km, self.plasma.h_km, self.plasma.Ne))
        nu = float(np.interp(h_km, self.plasma.h_km, self.plasma.nu))
        omega = 2.0 * np.pi * f_hz
        const_factor = (self.plasma.e**2) / (2.0 * self.plasma.eps0 * self.plasma.m_e * self.plasma.c)
        alpha_np_m = const_factor * (ne * nu) / (max(mu, 0.05) * (omega**2 + nu**2))
        return float(alpha_np_m * 1000.0 * 8.6858896)

    def trace_ray(
        self,
        freq_mhz: float,
        elevation_deg: float,
        step_km: float = 1.0,
        max_steps: int = 4000,
    ) -> RayTrajectory:
        f_hz = freq_mhz * 1e6
        elev_rad = np.radians(elevation_deg)
        cos_elev = np.cos(elev_rad)

        if elevation_deg <= 0.0:
            z = np.array([0.0])
            return RayTrajectory(
                freq_mhz=freq_mhz,
                elevation_deg=elevation_deg,
                r_km=np.array([self.R_E]),
                h_km=z,
                theta_rad=z,
                ground_dist_km=z,
                reflected=False,
                apex_height_km=0.0,
                ground_hit_km=0.0,
                phase_path_km=z,
                group_path_km=z,
                absorption_loss_db=z,
            )

        r, theta, psi = self.R_E, 0.0, np.pi / 2.0 - elev_rad
        P, P_prime, A_loss = 0.0, 0.0, 0.0

        r_hist, th_hist = [r], [theta]
        P_hist, Pp_hist, A_hist = [P], [P_prime], [A_loss]
        reflected = False
        apex_height = 0.0
        ground_hit = 0.0

        def derivatives(r_val, psi_val):
            h_val = r_val - self.R_E
            mu_val, dmu_dr = self._get_mu(f_hz, h_val)
            dr = np.cos(psi_val)
            dtheta = np.sin(psi_val) / r_val
            dpsi = -(np.sin(psi_val) / r_val) - (dmu_dr / mu_val) * np.sin(psi_val)
            return dr, dtheta, dpsi, mu_val, 1.0 / max(mu_val, 0.05), self._get_absorption(f_hz, h_val, mu_val)

        for step in range(max_steps):
            h_curr = r - self.R_E
            apex_height = max(apex_height, h_curr)

            if step > 20 and r <= self.R_E:
                reflected = True
                ground_hit = theta * self.R_E
                r_hist[-1] = self.R_E
                break

            if h_curr > 430.0 and np.cos(psi) > 0:
                break

            if h_curr >= 60.0 and np.cos(psi) > 0:
                fp_curr = float(np.interp(h_curr, self.h_km_grid, self.fp_hz_grid))
                X_curr = (fp_curr / f_hz) ** 2
                X_turn = 1.0 - ((self.R_E * cos_elev) / r) ** 2
                if X_curr >= X_turn or X_curr >= 1.0:
                    psi = np.pi - psi

            k1 = derivatives(r, psi)
            k2 = derivatives(r + 0.5 * step_km * k1[0], psi + 0.5 * step_km * k1[2])
            k3 = derivatives(r + 0.5 * step_km * k2[0], psi + 0.5 * step_km * k2[2])
            k4 = derivatives(r + step_km * k3[0], psi + step_km * k3[2])

            r += (step_km / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            theta += (step_km / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            psi += (step_km / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            P += (step_km / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
            P_prime += (step_km / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
            A_loss += (step_km / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

            r_hist.append(r)
            th_hist.append(theta)
            P_hist.append(P)
            Pp_hist.append(P_prime)
            A_hist.append(A_loss)

        r_arr = np.array(r_hist)
        th_arr = np.array(th_hist)
        return RayTrajectory(
            freq_mhz=freq_mhz,
            elevation_deg=elevation_deg,
            r_km=r_arr,
            h_km=r_arr - self.R_E,
            theta_rad=th_arr,
            ground_dist_km=th_arr * self.R_E,
            reflected=reflected,
            apex_height_km=apex_height,
            ground_hit_km=ground_hit,
            phase_path_km=np.array(P_hist),
            group_path_km=np.array(Pp_hist),
            absorption_loss_db=np.array(A_hist),
        )

    def trace_multihop_fan(
        self, freq_mhz: float, elevations: List[float] = [8.0, 15.0, 22.0, 32.0]
    ) -> List[RayTrajectory]:
        return [self.trace_ray(freq_mhz, elev) for elev in elevations]
