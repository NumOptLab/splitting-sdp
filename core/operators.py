"""SDP affine operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from core.psd import symmetrize


def triangular_n(length: int) -> int:
    """Return n such that length = n(n+1)/2."""
    n_float = (np.sqrt(8 * length + 1) - 1) / 2
    n = int(round(n_float))
    if n * (n + 1) // 2 != length:
        raise ValueError(f"Vector length {length} is not triangular.")
    return n


def svec_index(n: int, i: int, j: int) -> int:
    """Index of entry (i,j) in the lower-triangular svec order."""
    row = max(i, j)
    col = min(i, j)
    return col * n - (col * (col - 1)) // 2 + (row - col)


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


def smat(vector: np.ndarray) -> np.ndarray:
    """Inverse of ``svec``."""
    values = np.asarray(vector, dtype=float).reshape(-1)
    n = triangular_n(int(values.size))
    matrix = np.zeros((n, n), dtype=float)
    root2 = float(np.sqrt(2.0))
    cursor = 0
    for j in range(n):
        for i in range(j, n):
            value = float(values[cursor])
            if i != j:
                value /= root2
                matrix[j, i] = value
            matrix[i, j] = value
            cursor += 1
    return matrix


def operator_matrix(A: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.csr_matrix:
    """Matrix representation of ``A`` under ``svec`` coordinates."""
    if sparse.issparse(A):
        return A.tocsr()
    mats = np.asarray(A, dtype=float)
    if mats.ndim == 2:
        return mats
    if mats.shape[0] == 0:
        n = mats.shape[1]
        return np.zeros((0, n * (n + 1) // 2), dtype=float)
    return np.vstack([svec(Ai) for Ai in mats])


def apply_A(A: np.ndarray | sparse.spmatrix, X: np.ndarray) -> np.ndarray:
    """Apply ``A(X) = (<A_i, X>)_i``."""
    matrix = operator_matrix(A)
    x_vec = svec(X)
    return np.asarray(matrix @ x_vec, dtype=float).reshape(-1)


def apply_AT(A: np.ndarray | sparse.spmatrix, y: np.ndarray) -> np.ndarray:
    """Apply ``A^* y = sum_i y_i A_i``."""
    matrix = operator_matrix(A)
    y_vec = np.asarray(y, dtype=float).reshape(-1)
    vector = matrix.T @ y_vec if sparse.issparse(matrix) else y_vec @ matrix
    return smat(np.asarray(vector, dtype=float).reshape(-1))


def gram_matrix(A: np.ndarray | sparse.spmatrix) -> np.ndarray | sparse.csr_matrix:
    """Return the Gram matrix ``A A^*``."""
    matrix = operator_matrix(A)
    if matrix.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    gram = matrix @ matrix.T
    if sparse.issparse(gram):
        return gram.tocsr()
    return symmetrize(np.asarray(gram, dtype=float))


def operator_norm_squared(A: np.ndarray | sparse.spmatrix) -> float:
    """Return ``||A||^2`` for the Frobenius-to-Euclidean operator norm."""
    matrix = operator_matrix(A)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0.0
    if sparse.issparse(matrix):
        if matrix.nnz == 0:
            return 0.0
        if min(matrix.shape) == 1:
            value = sparse_linalg.norm(matrix)
        else:
            value = sparse_linalg.svds(matrix, k=1, return_singular_vectors=False)[0]
        return float(max(value * value, 0.0))
    G = gram_matrix(matrix)
    return float(max(np.linalg.eigvalsh(G)[-1], 0.0))


@dataclass(frozen=True)
class GramSolver:
    """Stable solver for systems with ``A A^*``."""

    gram: np.ndarray | sparse.csr_matrix
    rank: int
    min_singular: float
    condition: float
    chol_factor: tuple[np.ndarray, bool] | None
    pinv: np.ndarray | None
    sparse_factor: Callable[[np.ndarray], np.ndarray] | None = None
    sparse_lsmr: bool = False

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float)
        if self.gram.shape[0] == 0:
            return np.zeros_like(rhs_arr)
        if self.sparse_factor is not None:
            return np.asarray(self.sparse_factor(rhs_arr), dtype=float)
        if self.sparse_lsmr:
            if rhs_arr.ndim == 1:
                return sparse_linalg.lsmr(self.gram, rhs_arr, atol=1e-12, btol=1e-12)[0]
            return np.column_stack(
                [sparse_linalg.lsmr(self.gram, rhs_arr[:, col], atol=1e-12, btol=1e-12)[0] for col in range(rhs_arr.shape[1])]
            )
        if self.chol_factor is not None:
            return linalg.cho_solve(self.chol_factor, rhs_arr, check_finite=False)
        if self.pinv is not None:
            return self.pinv @ rhs_arr
        raise RuntimeError("GramSolver is not initialized.")


def _build_dense_gram_solver(G: np.ndarray, *, rcond: float) -> GramSolver:
    """Build a stable solver and diagnostics for ``A A^*``."""
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


def build_gram_solver(A: np.ndarray | sparse.spmatrix, *, rcond: float = 1e-12) -> GramSolver:
    """Build a stable solver and diagnostics for ``A A^*``."""
    G = gram_matrix(A)
    if sparse.issparse(G):
        if G.shape[0] == 0:
            return GramSolver(G, 0, 0.0, 1.0, None, None)
        try:
            factor = sparse_linalg.factorized(G.tocsc())
            return GramSolver(G, G.shape[0], float("nan"), float("nan"), None, None, sparse_factor=factor)
        except RuntimeError:
            return GramSolver(G, 0, 0.0, float("inf"), None, None, sparse_lsmr=True)
    return _build_dense_gram_solver(np.asarray(G, dtype=float), rcond=rcond)
