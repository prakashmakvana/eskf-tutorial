"""Minimal Error-State Kalman Filter (ESKF) for IMU + pose fusion.

Notation and equations follow:
    J. Solà, "Quaternion kinematics for the error-state Kalman filter", 2017.
    (Hamilton quaternions [w, x, y, z], local / right-hand orientation error.)
"""
from .quaternion import (skew, q_mult, q_conj, q_exp, q_log, q_to_R,
                         q_normalize, q_to_euler, q_boxplus, q_boxminus)
from .filter import ESKF, ESKFParams
from .simulate import SimConfig, simulate, save_dataset, load_dataset
from .runner import run_eskf

__all__ = ["skew", "q_mult", "q_conj", "q_exp", "q_log", "q_to_R", "q_normalize",
           "q_to_euler", "q_boxplus", "q_boxminus", "ESKF", "ESKFParams",
           "SimConfig", "simulate", "save_dataset", "load_dataset", "run_eskf"]
