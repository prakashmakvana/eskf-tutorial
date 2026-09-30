# Error-State Kalman Filter (ESKF): a hands-on tutorial

A from-scratch, fully commented implementation of the **Error-State Kalman Filter** for
IMU + 6D pose fusion, with a tutorial notebook that derives every equation, verifies the
Jacobians numerically, runs the filter on a dataset and checks its statistical consistency.

It is written in the spirit of Roger Labbe's
[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
(chapter 11, EKF) and follows the notation and equation numbers of
J. Solà, [*Quaternion kinematics for the error-state Kalman filter*](https://arxiv.org/abs/1711.02508) (2017).

## Contents

```
ESKF_tutorial.ipynb        the tutorial (theory, code, plots, consistency tests, exercises)
eskf/
  quaternion.py            ⊗, Exp, Log, ⊞, ⊟, R(q)  (Hamilton, [w, x, y, z])
  filter.py                ESKF: predict, update_pose, update_position, inject, reset
  simulate.py              synthetic 200 Hz IMU + 10 Hz 6D pose dataset with a dropout
  runner.py                runs the filter over time-stamped logs
scripts/
  generate_dataset.py      regenerate data/*.csv
  run_eskf.py              run the filter on data/*.csv and plot the errors
data/
  imu.csv                  t, ax, ay, az, wx, wy, wz               (body frame)
  pose_measurements.csv    t, px, py, pz, qw, qx, qy, qz
  ground_truth.csv         t, p, v, q, accel bias, gyro bias
```

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook ESKF_tutorial.ipynb      # the tutorial
python scripts/run_eskf.py                # or just run the filter on the CSV data
```

## The filter in one picture

| step | nominal state `x = [p, v, q, a_b, ω_b]` | error state `δx = [δp, δv, δθ, δa_b, δω_b]` |
|---|---|---|
| IMU sample | integrate non-linear kinematics | `P ← Fx P Fxᵀ + Fi Qi Fiᵀ` |
| measurement | – | `δx̂ = K r`, Joseph-form update of `P` |
| inject | `p += δp, …, q ← q ⊗ Exp(δθ)` | – |
| reset | – | `δx̂ ← 0`, `P ← G P Gᵀ` |

The orientation error `δθ` is a minimal 3-vector in the local (body) frame, so the covariance
is 15×15 and never singular, and a 6D pose measurement has the simple Jacobian
`H = [I 0 0 0 0; 0 0 I 0 0]` with residual `[p_m − p ; Log(q* ⊗ q_m)]`.

## What the notebook shows

* why filtering the *error* avoids the quaternion-covariance problem
* the error-state Jacobian `Fx`, checked against finite differences (and which O(Δt²) terms Solà drops)
* fusion of a 200 Hz IMU with 10 Hz pose measurements, including a 10 s measurement dropout
* bias estimation and its observability
* comparison against pure IMU dead reckoning
* Monte Carlo **NEES** and **NIS** consistency tests
* exercises: mistuning, observability, position-only updates, global vs local error, gravity estimation

## Using your own data

Replace the CSV files in `data/` with your own logs (same column names). `ground_truth.csv` is
optional: without it you can still evaluate the filter with the NIS test. Set the IMU noise
parameters in `ESKFParams` from your IMU's datasheet or an Allan-variance analysis, and make
sure the IMU and the pose refer to the same body frame (or add the extrinsic calibration).

## References

* J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017. [arXiv:1711.02508](https://arxiv.org/abs/1711.02508)
* R. Labbe, *Kalman and Bayesian Filters in Python*. [GitHub](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
* J. Solà, J. Deray, D. Atchuthan, *A micro Lie theory for state estimation in robotics*, 2018. [arXiv:1812.01537](https://arxiv.org/abs/1812.01537)
