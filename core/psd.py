"""PSD-cone utilities."""

from __future__ import annotations

import numpy as np

NUM_ZERO_TOL = 1e-12


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + arr.T)


def eigvalsh(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetric part of ``matrix``."""
    return np.linalg.eigvalsh(symmetrize(matrix))


def project_psd(matrix: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the positive semidefinite cone."""
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    clipped = np.where(values > NUM_ZERO_TOL, values, 0.0)
    return symmetrize((vectors * clipped) @ vectors.T)


def min_eig(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric part of ``matrix``."""
    values = eigvalsh(matrix)
    if values.size == 0:
        return 0.0
    return float(values[0])


def rank_from_eigs(values: np.ndarray, *, atol: float = NUM_ZERO_TOL, rtol: float = NUM_ZERO_TOL) -> int:
    """Numerical rank from eigenvalues."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0
    threshold = max(atol, rtol * float(np.max(np.abs(values))))
    return int(np.count_nonzero(values > threshold))


def psd_rank(matrix: np.ndarray, *, atol: float = NUM_ZERO_TOL, rtol: float = NUM_ZERO_TOL) -> int:
    """Numerical PSD rank using positive eigenvalues."""
    return rank_from_eigs(eigvalsh(matrix), atol=atol, rtol=rtol)


def lambda_min_abs(matrix: np.ndarray) -> float:
    """Smallest absolute eigenvalue of the symmetric part of ``matrix``."""
    values = eigvalsh(matrix)
    if values.size == 0:
        return 0.0
    return float(np.min(np.abs(values)))


def min_positive_eig(matrix: np.ndarray, *, tol: float = NUM_ZERO_TOL) -> float | None:
    """Smallest eigenvalue above ``tol`` for the symmetric part of ``matrix``."""
    values = eigvalsh(matrix)
    positive = values[values > tol]
    if positive.size == 0:
        return None
    return float(positive[0])
