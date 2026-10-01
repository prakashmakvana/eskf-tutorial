# The Error-State Kalman Filter for temporally consistent 6D pose estimation

A single Jupyter notebook that explains the **Error-State Kalman Filter (ESKF)** through a
concrete problem: turning the output of a per-frame 6D object pose estimator into a
**temporally consistent** track.

A per-frame estimator looks at every image on its own. Its output jitters from frame to frame,
is sometimes completely wrong, can flip between symmetric solutions, and is missing whenever the
object is not detected. The notebook builds an ESKF step by step, as short functions, each
followed by a small test, and shows how every part of the filter addresses one of these problems.

It is written in the spirit of Roger Labbe's
[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
and follows the notation of J. Solà,
[*Quaternion kinematics for the error-state Kalman filter*](https://arxiv.org/abs/1711.02508) (2017).

## What each part of the filter is for

| problem of a per-frame estimator | part of the ESKF |
|---|---|
| jitter | prediction and the Kalman gain: blending prediction with measurement averages out random error |
| frames of varying quality | a per-frame measurement noise `V_k`: a clear close-range frame gets more weight than a turbid one |
| symmetry flips | choosing the symmetric solution closest to the prediction |
| outliers | NIS (Mahalanobis) gating |
| dropouts | prediction alone, with an honestly growing covariance |
| orientation is not a vector | the error state: a minimal 3-vector `δθ`, and a simple Jacobian for a 6D pose |

## Results on the simulated sequence

| | pos RMSE | rot RMSE | pos jitter | rot jitter |
|---|---|---|---|---|
| raw per-frame estimates | 6.79 cm | 14.70° | 9.40 cm | 52.17° |
| naive filter (accepts every estimate) | 2.28 cm | 14.50° | 0.76 cm | 6.00° |
| robust ESKF (symmetry + gating) | 1.66 cm | 3.14° | 0.39 cm | 0.79° |

Jitter measures how much the frame-to-frame motion of a track differs from the true
frame-to-frame motion. Rotation errors are measured up to the object's symmetry.

## How the code is organised

The notebook defines a few small types rather than passing loose arrays and tuples around, so
the same metrics and plots work on every filter variant without change:

| type | what it holds |
|---|---|
| `Pose` | one 6D pose: `p`, `q` |
| `Estimate` | what the estimator reports for one frame: pose, its own `sigma_p` / `sigma_theta`, validity |
| `Truth` | the true trajectory |
| `State` | the nominal state: `p`, `v`, `q`, `w` |
| `Track` | a sequence of poses: the raw estimates, or the output of any filter |
| `FilterResult` | states, covariances, per-frame status and NIS |
| `SimConfig`, `MotionParams` | the simulation settings and the process noise |

## Contents of the notebook

1. The problem (jitter, outliers, symmetry flips, dropouts) and the idea of filtering
2. Why an *error-state* filter for poses
3. A small quaternion toolbox, built one function at a time
4. The data: a simulated per-frame pose estimator, and how to measure temporal consistency
5. True, nominal and error state
6. Prediction with a constant-velocity model, how uncertainty grows during a dropout, and a numerical check of the Jacobian
7. Correction with per-frame measurement uncertainty: why good frames get more weight than bad ones
8. A first, naive filter
9. Robustness: symmetry resolution, NIS gating and re-initialisation
10. Results: accuracy, jitter and honest ±3σ bounds
11. Tuning the process noise: smoothness versus lag
12. Consistency tests with NEES and NIS
13. Summary and references

## Files

```
ESKF_tutorial.ipynb        the complete tutorial
data/
  ground_truth.csv         t, p, v, q, w      true object pose and velocities in the camera frame
  pose_estimates.csv       t, valid, kind, px, py, pz, qw, qx, qy, qz, sigma_p, sigma_theta
requirements.txt
```

The CSV files are written by the notebook's simulator. `kind` (good / outlier / flip / missing)
is only used for plotting; the filter never sees it.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ESKF_tutorial.ipynb
```

Then run all cells from top to bottom (Kernel → Restart & Run All). The tuning sweep in
Section 11 and the Monte Carlo test in Section 12 take about a minute together.

## Using your own estimator output

Replace `data/pose_estimates.csv` with the per-frame output of your estimator (same columns,
`kind` can be left empty) and load it instead of calling the simulator. If your estimator reports
no uncertainty, use constant values for `sigma_p` and `sigma_theta`. Without ground truth you can
still judge the filter with the NIS test of Section 12. Adapt `SYMMETRIES` to your object.

## References

* J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017. [arXiv:1711.02508](https://arxiv.org/abs/1711.02508)
* R. Labbe, *Kalman and Bayesian Filters in Python*. [GitHub](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
* J. Solà, J. Deray, D. Atchuthan, *A micro Lie theory for state estimation in robotics*, 2018. [arXiv:1812.01537](https://arxiv.org/abs/1812.01537)
