"""SDP affine operators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _svec_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    offdiag: list[bool] = []
    for j in range(n):
        for i in range(j, n):
            rows.append(i)
            cols.append(j)
            offdiag.append(i != j)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(offdiag, dtype=bool),
    )


def svec(matrix: np.ndarray) -> np.ndarray:
    """Symmetric vectorization with sqrt(2)-scaled off-diagonal entries."""
    M = symmetrize(matrix)
    n = M.shape[0]
    rows, cols, offdiag = _svec_indices(n)
    values = np.asarray(M[rows, cols], dtype=float).copy()
    root2 = float(np.sqrt(2.0))
    values[offdiag] *= root2
    return values


def smat(vector: np.ndarray) -> np.ndarray:
    """Inverse of ``svec``."""
    values = np.asarray(vector, dtype=float).reshape(-1)
    n = triangular_n(int(values.size))
    matrix = np.zeros((n, n), dtype=float)
    rows, cols, offdiag = _svec_indices(n)
    scaled = values.copy()
    root2 = float(np.sqrt(2.0))
    scaled[offdiag] /= root2
    matrix[rows, cols] = scaled
    matrix[cols[offdiag], rows[offdiag]] = scaled[offdiag]
    return matrix


def operator_matrix(A: np.ndarray | sparse.spmatrix | SDPOperator) -> sparse.csr_matrix:
    """Sparse matrix representation of ``A`` under ``svec`` coordinates."""
    if isinstance(A, SDPOperator):
        return A.matrix
    if sparse.issparse(A):
        return A.tocsr()
    mats = np.asarray(A, dtype=float)
    if mats.ndim == 2:
        return sparse.csr_matrix(mats)
    if mats.shape[0] == 0:
        n = mats.shape[1]
        return sparse.csr_matrix((0, n * (n + 1) // 2), dtype=float)
    return sparse.csr_matrix(np.vstack([svec(Ai) for Ai in mats]))


@dataclass
class SDPOperator:
    """Sparse ``svec`` representation of the SDP affine map."""

    matrix: sparse.csr_matrix
    n: int
    _gram: sparse.csr_matrix | None = None
    _norm_squared: float | None = None

    @classmethod
    def from_data(cls, A: np.ndarray | sparse.spmatrix | SDPOperator, *, n: int | None = None) -> SDPOperator:
        if isinstance(A, SDPOperator):
            return A
        matrix = operator_matrix(A)
        inferred_n = triangular_n(matrix.shape[1])
        if n is not None and inferred_n != int(n):
            raise ValueError(f"Operator has svec dimension for n={inferred_n}, expected n={n}.")
        return cls(matrix=matrix, n=inferred_n)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def apply(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.matrix @ svec(X), dtype=float).reshape(-1)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        vector = self.matrix.T @ np.asarray(y, dtype=float).reshape(-1)
        return smat(np.asarray(vector, dtype=float).reshape(-1))

    def gram(self) -> sparse.csr_matrix:
        if self._gram is None:
            self._gram = (self.matrix @ self.matrix.T).tocsr()
        return self._gram

    def norm_squared(self) -> float:
        if self._norm_squared is not None:
            return self._norm_squared
        matrix = self.matrix
        if matrix.shape[0] == 0 or matrix.shape[1] == 0 or matrix.nnz == 0:
            self._norm_squared = 0.0
        elif min(matrix.shape) == 1:
            value = sparse_linalg.norm(matrix)
            self._norm_squared = float(max(value * value, 0.0))
        else:
            value = sparse_linalg.svds(matrix, k=1, return_singular_vectors=False)[0]
            self._norm_squared = float(max(value * value, 0.0))
        return self._norm_squared


def as_sdp_operator(A: np.ndarray | sparse.spmatrix | SDPOperator, *, n: int | None = None) -> SDPOperator:
    """Return the sparse operator wrapper for ``A``."""
    return SDPOperator.from_data(A, n=n)


def apply_A(A: np.ndarray | sparse.spmatrix | SDPOperator, X: np.ndarray) -> np.ndarray:
    """Apply ``A(X) = (<A_i, X>)_i``."""
    return as_sdp_operator(A, n=X.shape[0]).apply(X)


def apply_AT(A: np.ndarray | sparse.spmatrix | SDPOperator, y: np.ndarray) -> np.ndarray:
    """Apply ``A^* y = sum_i y_i A_i``."""
    return as_sdp_operator(A).adjoint(y)


def gram_matrix(A: np.ndarray | sparse.spmatrix | SDPOperator) -> sparse.csr_matrix:
    """Return the Gram matrix ``A A^*``."""
    return as_sdp_operator(A).gram()


def operator_norm_squared(A: np.ndarray | sparse.spmatrix | SDPOperator) -> float:
    """Return ``||A||^2`` for the Frobenius-to-Euclidean operator norm."""
    return as_sdp_operator(A).norm_squared()


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

    try:
        factor = linalg.cho_factor(G, lower=True, check_finite=False)
        return GramSolver(G, G.shape[0], float("nan"), float("nan"), factor, None)
    except linalg.LinAlgError:
        pass

    singular = np.linalg.svd(G, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    threshold = rcond * max(1.0, largest)
    rank = int(np.count_nonzero(singular > threshold))
    positive = singular[singular > threshold]
    min_singular = float(positive[-1]) if positive.size else 0.0
    condition = float(largest / min_singular) if min_singular > 0.0 else float("inf")
    return GramSolver(G, rank, min_singular, condition, None, np.linalg.pinv(G, rcond=rcond))


def build_gram_solver(A: np.ndarray | sparse.spmatrix | SDPOperator, *, rcond: float = 1e-12) -> GramSolver:
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
