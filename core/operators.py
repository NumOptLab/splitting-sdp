"""Dense SDP affine operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from core.psd import symmetrize


def apply_A(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply ``A(X) = (<A_i, X>)_i``."""
    return np.tensordot(np.asarray(A, dtype=float), np.asarray(X, dtype=float), axes=((1, 2), (0, 1)))


def apply_AT(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply ``A^* y = sum_i y_i A_i``."""
    result = np.tensordot(np.asarray(y, dtype=float).reshape(-1), np.asarray(A, dtype=float), axes=(0, 0))
    return symmetrize(result)


def gram_matrix(A: np.ndarray) -> np.ndarray:
    """Return the Gram matrix ``A A^*``."""
    mats = np.asarray(A, dtype=float)
    if mats.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    flat = mats.reshape(mats.shape[0], -1)
    return symmetrize(flat @ flat.T)


def operator_norm_squared(A: np.ndarray) -> float:
    """Return ``||A||^2`` for the Frobenius-to-Euclidean operator norm."""
    G = gram_matrix(A)
    if G.size == 0:
        return 0.0
    return float(max(np.linalg.eigvalsh(G)[-1], 0.0))


@dataclass(frozen=True)
class GramSolver:
    """Stable solver for systems with ``A A^*``."""

    gram: np.ndarray
    rank: int
    min_singular: float
    condition: float
    chol_factor: tuple[np.ndarray, bool] | None
    pinv: np.ndarray | None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float)
        if self.gram.size == 0:
            return np.zeros_like(rhs_arr)
        if self.chol_factor is not None:
            return linalg.cho_solve(self.chol_factor, rhs_arr, check_finite=False)
        if self.pinv is not None:
            return self.pinv @ rhs_arr
        raise RuntimeError("GramSolver is not initialized.")


def build_gram_solver(A: np.ndarray, *, rcond: float = 1e-12) -> GramSolver:
    """Build a stable solver and diagnostics for ``A A^*``."""
    G = gram_matrix(A)
    if G.size == 0:
        return GramSolver(G, 0, 0.0, 1.0, None, None)

    singular = np.linalg.svd(G, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = rcond * max(1.0, largest)
    rank = int(np.count_nonzero(singular > threshold))
    positive = singular[singular > threshold]
    min_singular = float(positive[-1]) if positive.size else 0.0
    condition = float(largest / min_singular) if min_singular > 0.0 else float("inf")

    if rank == G.shape[0]:
        try:
            factor = linalg.cho_factor(G, lower=True, check_finite=False)
            return GramSolver(G, rank, min_singular, condition, factor, None)
        except linalg.LinAlgError:
            pass
    return GramSolver(G, rank, min_singular, condition, None, np.linalg.pinv(G, rcond=rcond))


def svec(matrix: np.ndarray) -> np.ndarray:
    """Symmetric vectorization with sqrt(2)-scaled off-diagonal entries."""
    M = symmetrize(matrix)
    n = M.shape[0]
    values: list[float] = []
    root2 = float(np.sqrt(2.0))
    for j in range(n):
        for i in range(j, n):
            value = float(M[i, j])
            if i != j:
                value *= root2
            values.append(value)
    return np.asarray(values, dtype=float)


def operator_matrix(A: np.ndarray) -> np.ndarray:
    """Matrix representation of ``A`` under ``svec`` coordinates."""
    mats = np.asarray(A, dtype=float)
    if mats.shape[0] == 0:
        n = mats.shape[1]
        return np.zeros((0, n * (n + 1) // 2), dtype=float)
    return np.vstack([svec(Ai) for Ai in mats])
