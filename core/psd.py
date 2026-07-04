"""PSD-cone utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_ZERO_TOL = 1e-12


@dataclass(frozen=True)
class EigenInfo:
    """Reusable eigenvalue diagnostics for a symmetric matrix."""

    values: np.ndarray
    rank: int
    min_eig: float
    min_positive_eig: float | None
    min_abs_eig: float


@dataclass(frozen=True)
class PSDProjection:
    """PSD projection and spectral data from a single eigendecomposition."""

    matrix: np.ndarray
    eigvals: np.ndarray
    clipped_eigvals: np.ndarray
    rank: int
    min_eig: float
    negative_norm_squared: float


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    arr = np.asarray(matrix, dtype=float)
    return 0.5 * (arr + arr.T)


def eigen_info_from_values(
    values: np.ndarray,
    *,
    atol: float = NUM_ZERO_TOL,
    rtol: float = NUM_ZERO_TOL,
) -> EigenInfo:
    """Return rank and extremal-value diagnostics from eigenvalues."""
    eigs = np.asarray(values, dtype=float).reshape(-1)
    if eigs.size == 0:
        return EigenInfo(eigs, 0, 0.0, None, 0.0)
    threshold = max(atol, rtol * float(np.max(np.abs(eigs))))
    positive = eigs[eigs > threshold]
    return EigenInfo(
        values=eigs,
        rank=int(np.count_nonzero(eigs > threshold)),
        min_eig=float(np.min(eigs)),
        min_positive_eig=None if positive.size == 0 else float(np.min(positive)),
        min_abs_eig=float(np.min(np.abs(eigs))),
    )


def eigvalsh(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetric part of ``matrix``."""
    return np.linalg.eigvalsh(symmetrize(matrix))


def project_psd_with_eigs(matrix: np.ndarray) -> PSDProjection:
    """Project onto the PSD cone and keep reusable spectral diagnostics."""
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    clipped = np.where(values > NUM_ZERO_TOL, values, 0.0)
    projected = symmetrize((vectors * clipped) @ vectors.T)
    negative = np.where(values < 0.0, values, 0.0)
    return PSDProjection(
        matrix=projected,
        eigvals=values,
        clipped_eigvals=clipped,
        rank=rank_from_eigs(clipped),
        min_eig=float(values[0]) if values.size else 0.0,
        negative_norm_squared=float(negative @ negative),
    )


def project_psd(matrix: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the positive semidefinite cone."""
    return project_psd_with_eigs(matrix).matrix


def min_eig(matrix: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric part of ``matrix``."""
    return eigen_info_from_values(eigvalsh(matrix)).min_eig


def rank_from_eigs(values: np.ndarray, *, atol: float = NUM_ZERO_TOL, rtol: float = NUM_ZERO_TOL) -> int:
    """Numerical rank from eigenvalues."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0
    threshold = max(atol, rtol * float(np.max(np.abs(values))))
    return int(np.count_nonzero(values > threshold))


def psd_rank(matrix: np.ndarray, *, atol: float = NUM_ZERO_TOL, rtol: float = NUM_ZERO_TOL) -> int:
    """Numerical PSD rank using positive eigenvalues."""
    return eigen_info_from_values(eigvalsh(matrix), atol=atol, rtol=rtol).rank


def lambda_min_abs(matrix: np.ndarray) -> float:
    """Smallest absolute eigenvalue of the symmetric part of ``matrix``."""
    return eigen_info_from_values(eigvalsh(matrix)).min_abs_eig


def min_positive_eig(matrix: np.ndarray, *, tol: float = NUM_ZERO_TOL) -> float | None:
    """Smallest eigenvalue above ``tol`` for the symmetric part of ``matrix``."""
    return eigen_info_from_values(eigvalsh(matrix), atol=tol, rtol=tol).min_positive_eig
