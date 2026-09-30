"""Quaternion toolbox (Hamilton convention, q = [w, x, y, z]).

Section references (Sec. X) point to Solà (2017).
"""
import numpy as np


def skew(v):
    """[v]x : the skew-symmetric (cross-product) matrix of a 3-vector (Sec. 1.2.4)."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def q_mult(p, q):
    """Quaternion product p ⊗ q (Sec. 1.2.2, eq. 12)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def q_conj(q):
    """Conjugate q* (= inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_normalize(q):
    return q / np.linalg.norm(q)


def q_exp(rotvec):
    """Exp map: rotation vector θu in R^3  ->  unit quaternion (Sec. 1.3.5, eq. 101)."""
    rotvec = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-10:
        # first-order approximation, then re-normalise
        return q_normalize(np.array([1.0, *(0.5 * rotvec)]))
    u = rotvec / theta
    return np.array([np.cos(theta / 2.0), *(np.sin(theta / 2.0) * u)])


def q_log(q):
    """Log map: unit quaternion -> rotation vector θu in R^3 (Sec. 1.3.5, eq. 105)."""
    q = np.asarray(q, dtype=float)
    if q[0] < 0.0:            # q and -q are the same rotation; pick the short way
        q = -q
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-10:
        return 2.0 * v
    theta = 2.0 * np.arctan2(nv, q[0])
    return theta * v / nv


def q_to_R(q):
    """Rotation matrix R(q) such that x_world = R x_body (Sec. 1.3.3, eq. 115)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def q_boxplus(q, dtheta):
    """q ⊞ δθ = q ⊗ Exp(δθ)   (local / right perturbation, used by the ESKF)."""
    return q_normalize(q_mult(q, q_exp(dtheta)))


def q_boxminus(q1, q0):
    """q1 ⊟ q0 = Log(q0* ⊗ q1)  : the local rotation vector that takes q0 to q1."""
    return q_log(q_mult(q_conj(q0), q1))


def q_to_euler(q):
    """Roll, pitch, yaw (ZYX convention) in radians. Only for plotting."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])
