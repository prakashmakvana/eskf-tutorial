"""Synthetic dataset: a slowly manoeuvring vehicle (think ROV) with a 200 Hz IMU and
a 10 Hz 6D pose sensor that drops out for a while (occlusion / detector failure)."""
from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd

from .quaternion import q_exp, q_to_R, q_boxplus, q_normalize


@dataclass
class SimConfig:
    duration: float = 60.0
    imu_rate: float = 200.0
    pose_rate: float = 10.0
    dropout: tuple = (30.0, 40.0)          # seconds with no pose measurements
    # IMU errors (must match ESKFParams for a consistent filter)
    sigma_an: float = 0.05
    sigma_wn: float = 0.005
    sigma_aw: float = 0.002
    sigma_ww: float = 0.0002
    ab0: tuple = (0.08, -0.05, 0.10)
    wb0: tuple = (0.010, -0.008, 0.012)
    # pose sensor noise
    sigma_p: float = 0.05                  # [m]
    sigma_theta: float = np.deg2rad(2.0)   # [rad]
    gravity: tuple = (0.0, 0.0, -9.81)
    seed: int = 0


def _true_motion(t):
    """World-frame acceleration and body-frame angular rate of the true trajectory."""
    acc_w = np.array([0.3 * np.cos(0.3 * t),
                      0.3 * np.sin(0.2 * t),
                      0.05 * np.sin(0.1 * t)])
    w_b = np.array([0.20 * np.sin(0.4 * t),
                    0.15 * np.sin(0.3 * t + 1.0),
                    0.25 * np.cos(0.15 * t)])
    return acc_w, w_b


def simulate(cfg: SimConfig = SimConfig()):
    """Returns (truth, imu, pose) as pandas DataFrames."""
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.imu_rate
    n = int(round(cfg.duration * cfg.imu_rate))
    g = np.array(cfg.gravity)
    pose_every = int(round(cfg.imu_rate / cfg.pose_rate))

    p = np.zeros(3)
    v = np.array([0.0, -1.5, -0.5])
    q = q_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    ab = np.array(cfg.ab0, dtype=float)
    wb = np.array(cfg.wb0, dtype=float)

    truth, imu, pose = [], [], []
    for k in range(n + 1):
        t = k * dt
        truth.append([t, *p, *v, *q, *ab, *wb])

        if k % pose_every == 0 and not (cfg.dropout[0] <= t < cfg.dropout[1]):
            p_m = p + cfg.sigma_p * rng.standard_normal(3)
            q_m = q_boxplus(q, cfg.sigma_theta * rng.standard_normal(3))
            pose.append([t, *p_m, *q_m])

        if k == n:
            break

        # --- IMU measurement at time t (what a real IMU would output) ---
        acc_w, w_true = _true_motion(t)
        R = q_to_R(q)
        a_m = R.T @ (acc_w - g) + ab + cfg.sigma_an * rng.standard_normal(3)  # specific force
        w_m = w_true + wb + cfg.sigma_wn * rng.standard_normal(3)
        imu.append([t, *a_m, *w_m])

        # --- propagate truth with the same discretisation the filter uses ---
        p = p + v * dt + 0.5 * acc_w * dt ** 2
        v = v + acc_w * dt
        q = q_boxplus(q, w_true * dt)
        ab = ab + cfg.sigma_aw * np.sqrt(dt) * rng.standard_normal(3)
        wb = wb + cfg.sigma_ww * np.sqrt(dt) * rng.standard_normal(3)

    truth = pd.DataFrame(truth, columns=["t", "px", "py", "pz", "vx", "vy", "vz",
                                         "qw", "qx", "qy", "qz",
                                         "abx", "aby", "abz", "wbx", "wby", "wbz"])
    imu = pd.DataFrame(imu, columns=["t", "ax", "ay", "az", "wx", "wy", "wz"])
    pose = pd.DataFrame(pose, columns=["t", "px", "py", "pz", "qw", "qx", "qy", "qz"])
    return truth, imu, pose


def save_dataset(folder, truth, imu, pose):
    os.makedirs(folder, exist_ok=True)
    truth.to_csv(os.path.join(folder, "ground_truth.csv"), index=False, float_format="%.9f")
    imu.to_csv(os.path.join(folder, "imu.csv"), index=False, float_format="%.9f")
    pose.to_csv(os.path.join(folder, "pose_measurements.csv"), index=False, float_format="%.9f")


def load_dataset(folder):
    truth_path = os.path.join(folder, "ground_truth.csv")
    truth = pd.read_csv(truth_path) if os.path.exists(truth_path) else None
    imu = pd.read_csv(os.path.join(folder, "imu.csv"))
    pose = pd.read_csv(os.path.join(folder, "pose_measurements.csv"))
    return truth, imu, pose
