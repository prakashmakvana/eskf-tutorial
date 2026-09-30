"""Error-State Kalman Filter for IMU-driven 3D motion with pose measurements.

Nominal state  x  = [p, v, q, a_b, w_b]      (16 numbers, q is a unit quaternion)
Error state    dx = [dp, dv, dθ, da_b, dw_b]  (15 numbers, minimal: dθ in R^3)
True state     x_t = x ⊕ dx   with   q_t = q ⊗ Exp(dθ)

Equations follow Solà (2017), Sec. 5 (ESKF with local angular error, discrete time),
with gravity treated as a known constant (Solà additionally estimates g).
"""
from dataclasses import dataclass
import numpy as np

from .quaternion import skew, q_exp, q_to_R, q_boxplus, q_boxminus

I3 = np.eye(3)

# index slices into the 15-dim error state
P_IDX = slice(0, 3)
V_IDX = slice(3, 6)
TH_IDX = slice(6, 9)
AB_IDX = slice(9, 12)
WB_IDX = slice(12, 15)


@dataclass
class ESKFParams:
    """IMU noise parameters (discrete, per-sample), see Solà eq. (262).

    sigma_an : accelerometer white noise std        [m/s^2]
    sigma_wn : gyroscope white noise std            [rad/s]
    sigma_aw : accelerometer bias random walk       [m/s^2 / sqrt(s)]
    sigma_ww : gyroscope bias random walk           [rad/s / sqrt(s)]
    """
    sigma_an: float = 0.05
    sigma_wn: float = 0.005
    sigma_aw: float = 0.002
    sigma_ww: float = 0.0002
    gravity: tuple = (0.0, 0.0, -9.81)


class ESKF:
    def __init__(self, params: ESKFParams, p0, v0, q0, ab0=None, wb0=None, P0=None):
        self.params = params
        self.g = np.array(params.gravity, dtype=float)

        # ---- nominal state (large signal, integrated non-linearly) ----
        self.p = np.array(p0, dtype=float)
        self.v = np.array(v0, dtype=float)
        self.q = np.array(q0, dtype=float)
        self.ab = np.zeros(3) if ab0 is None else np.array(ab0, dtype=float)
        self.wb = np.zeros(3) if wb0 is None else np.array(wb0, dtype=float)

        # ---- error-state covariance (the error-state mean is always 0 after reset) ----
        self.P = np.eye(15) * 1e-2 if P0 is None else np.array(P0, dtype=float)

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #
    def error_state_jacobian(self, a_m, w_m, dt):
        """F_x = d(dx_{k+1}) / d(dx_k)  -- Solà eq. (269)."""
        R = q_to_R(self.q)
        a = a_m - self.ab
        w = w_m - self.wb

        Fx = np.eye(15)
        Fx[P_IDX, V_IDX] = I3 * dt
        Fx[V_IDX, TH_IDX] = -R @ skew(a) * dt
        Fx[V_IDX, AB_IDX] = -R * dt
        Fx[TH_IDX, TH_IDX] = q_to_R(q_exp(w * dt)).T
        Fx[TH_IDX, WB_IDX] = -I3 * dt
        return Fx

    def process_noise(self, dt):
        """F_i Q_i F_i^T  -- Solà eqs. (270)-(271)."""
        pr = self.params
        Fi = np.zeros((15, 12))
        Fi[3:15, :] = np.eye(12)           # noise enters v, θ, a_b, w_b (not p)
        Qi = np.diag(np.r_[np.full(3, pr.sigma_an ** 2 * dt ** 2),
                           np.full(3, pr.sigma_wn ** 2 * dt ** 2),
                           np.full(3, pr.sigma_aw ** 2 * dt),
                           np.full(3, pr.sigma_ww ** 2 * dt)])
        return Fi @ Qi @ Fi.T

    def predict(self, a_m, w_m, dt):
        """Propagate nominal state and error covariance with one IMU sample."""
        # 1) error-state covariance, linearised around the *current* nominal state
        Fx = self.error_state_jacobian(a_m, w_m, dt)
        self.P = Fx @ self.P @ Fx.T + self.process_noise(dt)

        # 2) nominal state, non-linear kinematics without noise -- Solà eq. (260)
        R = q_to_R(self.q)
        a = a_m - self.ab
        w = w_m - self.wb
        acc_world = R @ a + self.g
        self.p = self.p + self.v * dt + 0.5 * acc_world * dt ** 2
        self.v = self.v + acc_world * dt
        self.q = q_boxplus(self.q, w * dt)
        # biases are constant in the nominal model (random walk lives in the error state)

    # ------------------------------------------------------------------ #
    #  Correction                                                          #
    # ------------------------------------------------------------------ #
    def _correct(self, residual, H, V):
        """Generic ESKF correction: update -> inject -> reset (Solà Sec. 6)."""
        S = H @ self.P @ H.T + V
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual                                   # eq. (275)

        # Joseph form keeps P symmetric positive definite (eq. 276-277 discussion)
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ V @ K.T

        self._inject(dx)
        self._reset(dx)
        nis = float(residual @ np.linalg.solve(S, residual))  # normalised innovation squared
        return dx, nis

    def _inject(self, dx):
        """Fold the estimated error into the nominal state -- Solà eq. (282)."""
        self.p = self.p + dx[P_IDX]
        self.v = self.v + dx[V_IDX]
        self.q = q_boxplus(self.q, dx[TH_IDX])
        self.ab = self.ab + dx[AB_IDX]
        self.wb = self.wb + dx[WB_IDX]

    def _reset(self, dx):
        """Error mean goes back to zero; covariance is re-expressed around the new
        nominal orientation -- Solà eqs. (285)-(287)."""
        G = np.eye(15)
        G[TH_IDX, TH_IDX] = I3 - skew(0.5 * dx[TH_IDX])
        self.P = G @ self.P @ G.T

    # -------- measurement models -------- #
    def update_position(self, p_meas, sigma_p):
        """Position-only measurement y = p + n."""
        H = np.zeros((3, 15))
        H[:, P_IDX] = I3
        r = p_meas - self.p
        V = np.eye(3) * sigma_p ** 2
        return self._correct(r, H, V)

    def update_pose(self, p_meas, q_meas, sigma_p, sigma_theta):
        """Full 6D pose measurement (e.g. from a 6D pose estimator or mocap).

        Orientation residual is expressed in the *error-state tangent space*:
            r_θ = q_meas ⊟ q = Log(q* ⊗ q_meas)      so  H_θ = I  (to first order).
        """
        H = np.zeros((6, 15))
        H[0:3, P_IDX] = I3
        H[3:6, TH_IDX] = I3
        r = np.r_[p_meas - self.p, q_boxminus(q_meas, self.q)]
        V = np.diag(np.r_[np.full(3, sigma_p ** 2), np.full(3, sigma_theta ** 2)])
        return self._correct(r, H, V)

    # -------- helpers -------- #
    @property
    def state(self):
        return dict(p=self.p.copy(), v=self.v.copy(), q=self.q.copy(),
                    ab=self.ab.copy(), wb=self.wb.copy())

    def error_to(self, p, v, q, ab, wb):
        """Error state dx such that truth = self ⊕ dx (used for evaluation / NEES)."""
        return np.r_[p - self.p, v - self.v, q_boxminus(q, self.q), ab - self.ab, wb - self.wb]
