"""Run the ESKF on the CSV dataset in ./data and plot the result.

    python scripts/run_eskf.py            # uses data/*.csv
    python scripts/run_eskf.py --save fig.png
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eskf import ESKFParams, SimConfig, load_dataset, run_eskf, q_boxminus

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
ap.add_argument("--save", default=None)
args = ap.parse_args()

cfg = SimConfig()                         # noise values the data was generated with
truth, imu, pose = load_dataset(args.data)
params = ESKFParams(cfg.sigma_an, cfg.sigma_wn, cfg.sigma_aw, cfg.sigma_ww, cfg.gravity)

# initialise from the first pose measurement, zero velocity guess is NOT used:
# we give a rough velocity and large uncertainty instead
x0 = dict(p=pose.iloc[0][["px", "py", "pz"]].to_numpy(),
          v=np.zeros(3),
          q=pose.iloc[0][["qw", "qx", "qy", "qz"]].to_numpy())
P0 = np.diag(np.r_[np.full(3, 0.1**2), np.full(3, 2.0**2), np.full(3, np.deg2rad(5)**2),
                   np.full(3, 0.2**2), np.full(3, 0.02**2)])
log = run_eskf(imu, pose, params, x0, P0, cfg.sigma_p, cfg.sigma_theta)

t = log["t"]
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
if truth is not None:
    Tp = truth[["px", "py", "pz"]].to_numpy()
    Tq = truth[["qw", "qx", "qy", "qz"]].to_numpy()
    ep = Tp - log["p"]
    eth = np.degrees([q_boxminus(Tq[k], log["q"][k]) for k in range(len(t))])
    for i, c in enumerate("xyz"):
        ax[0].plot(t, ep[:, i], label=f"e_p{c}")
        ax[0].plot(t, 3 * np.sqrt(log["Pdiag"][:, i]), "k--", lw=0.7)
        ax[0].plot(t, -3 * np.sqrt(log["Pdiag"][:, i]), "k--", lw=0.7)
        ax[1].plot(t, eth[:, i], label=f"e_θ{c}")
        ax[2].plot(t, truth[f"ab{c}"], "k:", lw=1)
        ax[2].plot(t, log["ab"][:, i], label=f"a_b{c}")
ax[0].set_ylabel("position error [m]"); ax[0].legend(ncol=3)
ax[1].set_ylabel("attitude error [deg]"); ax[1].legend(ncol=3)
ax[2].set_ylabel("accel bias [m/s²]"); ax[2].legend(ncol=3); ax[2].set_xlabel("t [s]")
for a in ax:
    a.axvspan(*cfg.dropout, color="grey", alpha=0.15)
fig.suptitle("ESKF: IMU (200 Hz) + 6D pose (10 Hz), grey = measurement dropout")
fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=120)
else:
    plt.show()
if truth is not None:
    print(f"position RMSE: {np.sqrt(np.mean(np.sum(ep**2, 1))):.3f} m")
