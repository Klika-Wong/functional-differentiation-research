"""Linear Gaussian state-space filtering utilities (NumPy only).

Model for one independent sequence b:

    x_0 ~ N(m0, P0)
    y_t = H x_t + v_t,         v_t ~ N(0, R)      for t = 0, 1, ...
    x_{t+1} = F x_t + w_t,     w_t ~ N(0, Q)

The time index t = 0 uses the prior mean/covariance m0/P0 unchanged; the
transition F and process noise Q only apply when advancing from t to t+1.

`kalman_sequence` returns, for each observation, its prior predictive Gaussian
density (before the update), the filtered posterior state, and the one-step-
ahead predictive distribution of the next observation given data through t.

Numerical conventions:
  * No explicit matrix inverses: Cholesky factorizations and triangular
    solves are used throughout.
  * The posterior covariance uses the Joseph form, which stays symmetric and
    positive semidefinite under rounding.
  * Symmetry and PSD checks use an absolute tolerance of 1e-10; R only needs
    eigenvalues strictly above zero. No jitter is added to the model.
  * The prior predictive covariance, its Cholesky factor (which yields both
    the log-determinant and the Kalman gain) and the next-step covariance are
    computed once per time step, not per batch element.
  * Results are computed in float64 regardless of the input dtype; inputs are
    never modified and read-only arrays are accepted.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kalman_sequence"]

_DTYPE = np.float64
_SYM_TOL = 1e-10           # absolute tolerance for symmetry / PSD checks
_PD_TOL = 0.0              # R must be strictly positive definite


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _check_finite(name, arr):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")


def _as_matrix(name, value, rows, cols):
    try:
        arr = np.asarray(value, dtype=_DTYPE)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} cannot be converted to a float64 array: {exc}")
    if arr.shape != (rows, cols):
        raise ValueError(f"{name} must have shape {(rows, cols)}, got {arr.shape}")
    _check_finite(name, arr)
    return arr


def _matrix_as_symmetric(name, value, n):
    arr = _as_matrix(name, value, n, n)
    asym = float(np.max(np.abs(arr - arr.T))) if arr.size else 0.0
    if asym > _SYM_TOL:
        raise ValueError(
            f"{name} must be symmetric (max |A - A^T| = {asym:.3e}, "
            f"tolerance {_SYM_TOL:.0e})"
        )
    # Symmetrize to drop rounding-level asymmetry (keeps chol well-posed).
    sym = 0.5 * (arr + arr.T)
    return np.ascontiguousarray(sym, dtype=_DTYPE)


def _as_symmetric_cov(name, value, n, require_pd):
    """Validate a symmetric covariance: PSD (or PD when required)."""
    arr = _matrix_as_symmetric(name, value, n)
    eigvals = np.linalg.eigvalsh(arr)
    min_eig = float(eigvals[0])
    if min_eig < -_SYM_TOL:
        raise ValueError(
            f"{name} must be positive semi-definite "
            f"(min eigenvalue {min_eig:.3e}, tolerance {_SYM_TOL:.0e})"
        )
    if require_pd and min_eig <= _PD_TOL:
        raise ValueError(
            f"{name} must be positive definite "
            f"(min eigenvalue {min_eig:.3e} <= {_PD_TOL:.0e})"
        )
    return arr


# --------------------------------------------------------------------------- #
# Linear algebra kernels
# --------------------------------------------------------------------------- #
def _cholesky(mat):
    """Factor the supplied covariance without silently changing the model."""
    return np.linalg.cholesky(mat)


def _forward_substitute(lower, rhs):
    """Solve L X = rhs for lower-triangular L; rhs shape (d,) or (d, K)."""
    d = lower.shape[0]
    out = np.array(rhs, dtype=_DTYPE, copy=True)
    for i in range(d):
        if i:
            out[i] -= lower[i, :i] @ out[:i]
        out[i] /= lower[i, i]
    return out


def _chol_solve(lower, rhs):
    """Solve L L^T x = rhs; rhs shape (d,) or (d, K)."""
    out = _forward_substitute(lower, rhs)
    for i in range(lower.shape[0] - 1, -1, -1):
        out[i] -= lower[i+1:, i] @ out[i+1:]
        out[i] /= lower[i, i]
    return out


def _symmetrize(mat):
    return 0.5 * (mat + mat.T)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def kalman_sequence(observations, F, H, Q, R, m0, P0):
    """Filter a batch of independent linear Gaussian sequences.

    Parameters
    ----------
    observations : array_like, shape (B, T, d)
        Observed vectors; all sequences share the same matrices and prior.
    F : array_like, shape (n, n)
    H : array_like, shape (d, n)
    Q : array_like, shape (n, n)   symmetric PSD process covariance
    R : array_like, shape (d, d)   symmetric PD observation covariance
    m0 : array_like, shape (n,)
    P0 : array_like, shape (n, n)  symmetric PSD initial covariance

    Returns
    -------
    dict of float64 ndarrays:

    ``predicted_observation`` : (B, T, d)
        Prior predictive means H m_{t|t-1} of each observation.
    ``innovation_covariance`` : (T, d, d)
        Prior predictive covariances H P_{t|t-1} H^T + R (batch independent).
    ``log_likelihood`` : (B, T)
        log N(y_t ; H m_{t|t-1}, H P_{t|t-1} H^T + R), including the
        ``-d/2 log(2 pi)`` constant and the log-determinant term.
    ``filtered_state`` : (B, T, n)
        Posterior means m_{t|t}.
    ``next_observation`` : (B, T, d)
        E[y_{t+1} | y_{0:t}] = H F m_{t|t}.
    ``next_covariance`` : (T, d, d)
        Cov(y_{t+1} | y_{0:t}) = H (F P_{t|t} F^T + Q) H^T + R.

    Notes
    -----
    The caller's arrays are never modified. When ``B == 0`` the per-sequence
    outputs are empty while the time-indexed covariance arrays are still filled
    (matrices and initial distribution do not depend on data).
    """
    obs = np.asarray(observations, dtype=_DTYPE)
    if obs.ndim != 3:
        raise ValueError(f"observations must be 3-D (B, T, d), got ndim {obs.ndim}")
    batch, steps, d = obs.shape
    if d < 1:
        raise ValueError(f"observation dimension d must be positive, got {d}")

    m0_arr = np.asarray(m0, dtype=_DTYPE)
    if m0_arr.ndim != 1:
        raise ValueError(f"m0 must be 1-D with shape (n,), got ndim {m0_arr.ndim}")
    n = m0_arr.shape[0]
    if n < 1:
        raise ValueError(f"state dimension n must be positive, got {n}")

    _check_finite("m0", m0_arr)
    F_arr = _as_matrix("F", F, n, n)
    H_arr = _as_matrix("H", H, d, n)
    Q_arr = _as_symmetric_cov("Q", Q, n, require_pd=False)
    R_arr = _as_symmetric_cov("R", R, d, require_pd=True)
    P0_arr = _as_symmetric_cov("P0", P0, n, require_pd=False)
    _check_finite("observations", obs)

    # Owned float64 copies: inputs may be read-only or non-contiguous.
    m0_arr = np.array(m0_arr, dtype=_DTYPE, copy=True)
    F_arr = np.array(F_arr, dtype=_DTYPE, copy=True)
    H_arr = np.array(H_arr, dtype=_DTYPE, copy=True)
    Q_arr = np.array(Q_arr, dtype=_DTYPE, copy=True)
    R_arr = np.array(R_arr, dtype=_DTYPE, copy=True)
    P0_arr = np.array(P0_arr, dtype=_DTYPE, copy=True)
    obs = np.array(obs, dtype=_DTYPE, copy=True)

    predicted_observation = np.empty((batch, steps, d), dtype=_DTYPE)
    log_likelihood = np.empty((batch, steps), dtype=_DTYPE)
    filtered_state = np.empty((batch, steps, n), dtype=_DTYPE)
    next_observation = np.empty((batch, steps, d), dtype=_DTYPE)
    innovation_covariance = np.empty((steps, d, d), dtype=_DTYPE)
    next_covariance = np.empty((steps, d, d), dtype=_DTYPE)

    Ht = np.ascontiguousarray(H_arr.T)                    # (n, d)
    log_2pi = np.log(2.0 * np.pi)
    eye_n = np.eye(n, dtype=_DTYPE)

    # Means depend on each sequence's observations; covariance is shared.
    mean = np.broadcast_to(m0_arr, (batch, n)).copy()
    cov = P0_arr
    for t in range(steps):
        # Prior predictive of the observation: S = H P H^T + R, factorized once.
        pred_mean = mean @ Ht                                          # (B,d)
        pred_cov = _symmetrize(H_arr @ cov @ Ht + R_arr)               # (d, d)
        lower = _cholesky(pred_cov)                                    # S = L L^T
        innovation_covariance[t] = pred_cov
        log_det = 2.0 * float(np.sum(np.log(np.diag(lower))))

        gain = _chol_solve(lower, H_arr @ cov).T
        if batch > 0:
            diff = obs[:, t, :] - pred_mean                            # (B, d)

            predicted_observation[:, t] = pred_mean
            maha = np.einsum("bi,bi->b", diff, _chol_solve(lower, diff.T).T)
            log_likelihood[:, t] = -0.5 * (d * log_2pi + log_det + maha)
            post_mean = mean + diff @ gain.T                           # (B, n)
            filtered_state[:, t] = post_mean

        else:
            post_mean = mean

        A = eye_n - gain @ H_arr
        post_cov = _symmetrize(A @ cov @ A.T + gain @ R_arr @ gain.T)

        next_cov = _symmetrize(F_arr @ post_cov @ F_arr.T + Q_arr)     # (n, n)
        next_covariance[t] = _symmetrize(H_arr @ next_cov @ Ht + R_arr)
        mean = post_mean @ F_arr.T
        cov = next_cov
        if batch > 0:
            next_observation[:, t] = (post_mean @ F_arr.T) @ Ht         # H F m_{t|t}

    return {
        "predicted_observation": predicted_observation,
        "innovation_covariance": innovation_covariance,
        "log_likelihood": log_likelihood,
        "filtered_state": filtered_state,
        "next_observation": next_observation,
        "next_covariance": next_covariance,
    }
