"""Glue code: run the ESKF over time-stamped IMU + pose logs."""
import numpy as np
from .filter import ESKF, ESKFParams


def run_eskf(imu, pose, params: ESKFParams, x0: dict, P0, sigma_p, sigma_theta,
             use_pose=True):
    """Run the filter over DataFrames `imu` (t,ax..wz) and `pose` (t,px..qz).

    Measurements with timestamp <= current IMU time are applied before the next
    prediction step. Returns a dict of arrays logged at every IMU step.
    """
    f = ESKF(params, x0["p"], x0["v"], x0["q"], x0.get("ab"), x0.get("wb"), P0)
    t_imu = imu["t"].to_numpy()
    A = imu[["ax", "ay", "az"]].to_numpy()
    W = imu[["wx", "wy", "wz"]].to_numpy()
    t_pose = pose["t"].to_numpy()
    Pm = pose[["px", "py", "pz"]].to_numpy()
    Qm = pose[["qw", "qx", "qy", "qz"]].to_numpy()

    n = len(t_imu)
    log = {k: np.zeros((n + 1, d)) for k, d in
           [("p", 3), ("v", 3), ("q", 4), ("ab", 3), ("wb", 3), ("Pdiag", 15)]}
    log["t"] = np.r_[t_imu, t_imu[-1] + (t_imu[-1] - t_imu[-2])]
    log["P"] = np.zeros((n + 1, 15, 15))
    log["nis"] = []

    j = 0
    for k in range(n + 1):
        t = log["t"][k]
        # --- corrections available at this time ---
        while use_pose and j < len(t_pose) and t_pose[j] <= t + 1e-9:
            _, nis = f.update_pose(Pm[j], Qm[j], sigma_p, sigma_theta)
            log["nis"].append((t_pose[j], nis))
            j += 1
        # --- log posterior at time t ---
        for key in ("p", "v", "q", "ab", "wb"):
            log[key][k] = getattr(f, key)
        log["P"][k] = f.P
        log["Pdiag"][k] = np.diag(f.P)
        # --- predict to next time ---
        if k < n:
            dt = (t_imu[k + 1] - t_imu[k]) if k + 1 < n else (t_imu[k] - t_imu[k - 1])
            f.predict(A[k], W[k], dt)
    log["nis"] = np.array(log["nis"])
    return log
