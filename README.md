# Error-State Kalman Filter (ESKF): a hands-on tutorial

A single Jupyter notebook that builds an **Error-State Kalman Filter** for IMU + 6D pose fusion
from first principles. The filter is written as a handful of short functions, each one right after
its equation and followed by a small test. The Jacobians are verified numerically, the filter
runs on a dataset, and its statistical consistency is tested.

It is written in the spirit of Roger Labbe's
[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
(chapter 11, EKF) and follows the notation and equation numbers of
J. Solà, [*Quaternion kinematics for the error-state Kalman filter*](https://arxiv.org/abs/1711.02508) (2017).

## Contents

```
ESKF_tutorial.ipynb        the complete tutorial: theory, code, plots, consistency tests
data/
  imu.csv                  t, ax, ay, az, wx, wy, wz               (body frame)
  pose_measurements.csv    t, px, py, pz, qw, qx, qy, qz
  ground_truth.csv         t, p, v, q, accel bias, gyro bias
requirements.txt
```

The CSV files are written by the notebook's simulator. They are included so you can see the
data format without running anything.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ESKF_tutorial.ipynb
```

Then run all cells from top to bottom (Kernel → Restart & Run All). The Monte Carlo
consistency test near the end takes about 30 seconds.

## What the notebook covers

1. Why filtering the *error* avoids the quaternion-covariance problem
2. A small quaternion toolbox built one function at a time: [·]×, ⊗, Exp, Log, R(q), ⊞, ⊟
3. True, nominal and error state
4. Prediction: nominal kinematics, the error-state Jacobian `Fx`, how uncertainty grows without measurements, and a numerical check of `Fx`
   (including which O(Δt²) terms Solà drops on purpose)
5. Correction: update, injection and reset, with a 6D pose measurement model and a one-step demo
6. A simulated dataset: 200 Hz IMU, 10 Hz 6D pose, and a 10 s measurement dropout
7. Running the filter: errors with ±3σ bounds, bias estimation, comparison with IMU dead reckoning
8. Monte Carlo **NEES** and **NIS** consistency tests

## The filter in one table

| step | nominal state `x = [p, v, q, a_b, ω_b]` | error state `δx = [δp, δv, δθ, δa_b, δω_b]` |
|---|---|---|
| IMU sample | integrate non-linear kinematics | `P ← Fx P Fxᵀ + Fi Qi Fiᵀ` |
| measurement | – | `δx̂ = K r`, Joseph-form update of `P` |
| inject | `p += δp, …, q ← q ⊗ Exp(δθ)` | – |
| reset | – | `δx̂ ← 0`, `P ← G P Gᵀ` |

## Using your own data

Replace the CSV files in `data/` with your own logs (same column names) and skip the simulation
cell. `ground_truth.csv` is optional: without it you can still evaluate the filter with the NIS
test. Set the IMU noise parameters in `ESKFParams` from your IMU's datasheet or an Allan-variance
analysis, and make sure the IMU and the pose refer to the same body frame.

## References

* J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017. [arXiv:1711.02508](https://arxiv.org/abs/1711.02508)
* R. Labbe, *Kalman and Bayesian Filters in Python*. [GitHub](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
* J. Solà, J. Deray, D. Atchuthan, *A micro Lie theory for state estimation in robotics*, 2018. [arXiv:1812.01537](https://arxiv.org/abs/1812.01537)
