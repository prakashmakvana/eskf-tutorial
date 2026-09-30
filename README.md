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

## Results on the simulated sequence

| | pos RMSE | rot RMSE | pos jitter | rot jitter |
|---|---|---|---|---|
| raw per-frame estimates | 6.78 cm | 14.71° | 9.40 cm | 52.24° |
| naive filter (accepts every estimate) | 2.20 cm | 14.59° | 0.77 cm | 6.11° |
| robust ESKF (symmetry + gating) | 1.55 cm | 2.88° | 0.41 cm | 0.78° |
| RTS smoother (offline, uses future frames) | 0.55 cm | 0.99° | 0.02 cm | 0.06° |

Jitter measures how much the frame-to-frame motion of a track differs from the true
frame-to-frame motion. Rotation errors are measured up to the object's symmetry.

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
13. Filter versus RTS smoother, and the link to factor-graph optimisation
14. Summary and references

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

Then run all cells from top to bottom (Kernel → Restart & Run All). The Monte Carlo test in
Section 12 and the tuning sweep in Section 11 take about a minute together.

## Using your own estimator output

Replace `data/pose_estimates.csv` with the per-frame output of your estimator (same columns,
`kind` can be left empty) and load it instead of calling the simulator. If your estimator reports
no uncertainty, use constant values for `sigma_p` and `sigma_theta`. Without ground truth you can
still judge the filter with the NIS test of Section 12. Adapt `SYMMETRIES` to your object.

## References

* J. Solà, *Quaternion kinematics for the error-state Kalman filter*, 2017. [arXiv:1711.02508](https://arxiv.org/abs/1711.02508)
* R. Labbe, *Kalman and Bayesian Filters in Python*. [GitHub](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001.
* H. E. Rauch, F. Tung, C. T. Striebel, *Maximum likelihood estimates of linear dynamic systems*, AIAA Journal, 1965.
* J. Solà, J. Deray, D. Atchuthan, *A micro Lie theory for state estimation in robotics*, 2018. [arXiv:1812.01537](https://arxiv.org/abs/1812.01537)
* F. Dellaert, M. Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics, 2017.
