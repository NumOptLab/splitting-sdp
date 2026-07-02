"""Solution diagnostics for SDP iterates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from core.operators import operator_matrix, svec
from core.psd import eigvalsh, lambda_min_abs, psd_rank, symmetrize
from utils.h5io import read_h5_array, read_sdp_operator


@dataclass(frozen=True)
class NondegeneracyReport:
    holds: bool
    rank_left: int
    rank_right: int
    rank_combined: int
    singular_values: list[float]
    tolerance: float


def rank_stabilization_iteration(ranks: list[int]) -> int | None:
    from utils.plotting import rank_stabilization_iteration as _rank_stabilization_iteration

    return _rank_stabilization_iteration(ranks)


def strict_complementarity(X: np.ndarray, S: np.ndarray, *, tol: float = 1e-8) -> bool:
    return psd_rank(X, atol=tol, rtol=tol) + psd_rank(S, atol=tol, rtol=tol) == X.shape[0]


def _symmetric_basis_from_columns(columns: np.ndarray) -> np.ndarray:
    n, r = columns.shape
    values: list[np.ndarray] = []
    for a in range(r):
        for b in range(a, r):
            M = np.outer(columns[:, a], columns[:, b])
            if a != b:
                M += M.T
            values.append(svec(M))
    if not values:
        return np.zeros((n * (n + 1) // 2, 0), dtype=float)
    return np.column_stack(values)


def _dense_matrix(values: np.ndarray | sparse.spmatrix) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values, dtype=float)


def _rank(values: np.ndarray | sparse.spmatrix, tol: float) -> tuple[int, np.ndarray]:
    dense = _dense_matrix(values)
    if dense.size == 0:
        return 0, np.asarray([], dtype=float)
    singular = np.linalg.svd(dense, compute_uv=False)
    return int(np.count_nonzero(singular > tol)), singular


def _intersection_report(left: np.ndarray | sparse.spmatrix, right: np.ndarray | sparse.spmatrix, tol: float) -> NondegeneracyReport:
    left_dense = _dense_matrix(left)
    right_dense = _dense_matrix(right)
    rank_left, _ = _rank(left, tol)
    rank_right, _ = _rank(right, tol)
    combined = (
        np.column_stack([left_dense, right_dense])
        if left_dense.shape[1] or right_dense.shape[1]
        else np.zeros((left_dense.shape[0], 0), dtype=float)
    )
    rank_combined, singular = _rank(combined, tol)
    return NondegeneracyReport(
        holds=(rank_left + rank_right == rank_combined),
        rank_left=rank_left,
        rank_right=rank_right,
        rank_combined=rank_combined,
        singular_values=[float(v) for v in singular],
        tolerance=tol,
    )


def nondegeneracy_reports(
    X: np.ndarray,
    S: np.ndarray,
    A: np.ndarray | sparse.spmatrix,
    *,
    tol: float = 1e-8,
) -> dict[str, NondegeneracyReport]:
    Xs = symmetrize(X)
    Ss = symmetrize(S)
    x_values, x_vectors = np.linalg.eigh(Xs)
    s_values, s_vectors = np.linalg.eigh(Ss)
    x_null = x_vectors[:, x_values <= tol]
    s_null = s_vectors[:, s_values <= tol]

    Aop = operator_matrix(A)
    range_at = Aop.T
    nx_basis = _symmetric_basis_from_columns(x_null)
    primal = _intersection_report(nx_basis, range_at, tol)

    ns_basis = _symmetric_basis_from_columns(s_null)
    if Aop.shape[0] == 0:
        null_a = np.eye(ns_basis.shape[0])
    else:
        _, singular, vt = np.linalg.svd(_dense_matrix(Aop), full_matrices=True)
        rank = int(np.count_nonzero(singular > tol))
        null_a = vt[rank:].T
    dual = _intersection_report(ns_basis, null_a, tol)
    return {"primal": primal, "dual": dual}


def _maxcut_diag_map(columns: np.ndarray) -> np.ndarray:
    n, r = columns.shape
    values: list[np.ndarray] = []
    for a in range(r):
        for b in range(a, r):
            column = columns[:, a] * columns[:, b]
            if a != b:
                column = 2.0 * column
            values.append(column)
    if not values:
        return np.zeros((n, 0), dtype=float)
    return np.column_stack(values)


def maxcut_nondegeneracy_reports(X: np.ndarray, S: np.ndarray, *, tol: float = 1e-8) -> dict[str, NondegeneracyReport]:
    Xs = symmetrize(X)
    Ss = symmetrize(S)
    n = Xs.shape[0]
    p = n * (n + 1) // 2
    x_values, x_vectors = np.linalg.eigh(Xs)
    s_values, s_vectors = np.linalg.eigh(Ss)
    x_null = x_vectors[:, x_values <= tol]
    s_null = s_vectors[:, s_values <= tol]

    primal_left = x_null.shape[1] * (x_null.shape[1] + 1) // 2
    coordinate_projection = np.sum(x_null * x_null, axis=1) if x_null.size else np.zeros(n, dtype=float)
    primal_intersection = int(np.count_nonzero(1.0 - coordinate_projection <= tol))
    primal = NondegeneracyReport(
        holds=(primal_intersection == 0),
        rank_left=primal_left,
        rank_right=n,
        rank_combined=primal_left + n - primal_intersection,
        singular_values=[float(v) for v in coordinate_projection],
        tolerance=tol,
    )

    diag_map = _maxcut_diag_map(s_null)
    dual_rank, singular = _rank(diag_map, tol)
    dual_left = s_null.shape[1] * (s_null.shape[1] + 1) // 2
    dual_right = p - n
    dual = NondegeneracyReport(
        holds=(dual_rank == dual_left),
        rank_left=dual_left,
        rank_right=dual_right,
        rank_combined=dual_right + dual_rank,
        singular_values=[float(v) for v in singular],
        tolerance=tol,
    )
    return {"primal": primal, "dual": dual}


def _attr(handle: h5py.File, key: str, default: object = None) -> object:
    value = handle.attrs.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _adapt_label(handle: h5py.File) -> str:
    adapt = str(_attr(handle, "adapt", "none"))
    if adapt == "first-k":
        return f"first-k{int(_attr(handle, 'adapt_iters', 0))}"
    return adapt


def _seed_value(handle: h5py.File) -> int | None:
    seed = _attr(handle, "seed", None)
    if seed in (None, -1):
        return None
    return int(seed)


def _nd_dict(report: NondegeneracyReport) -> dict[str, bool | int | float]:
    return {
        "holds": report.holds,
        "rank_left": report.rank_left,
        "rank_right": report.rank_right,
        "rank_combined": report.rank_combined,
        "tolerance": report.tolerance,
    }


def diagnose_solution(path: str | Path, *, check_nd: bool = False, tol: float = 1e-8, maxcut: bool = False) -> dict[str, object]:
    """Read a solution HDF5 file and return requested diagnostics."""
    path_obj = Path(path)
    with h5py.File(path_obj, "r") as handle:
        X = np.asarray(read_h5_array(handle["X_last"]))
        S = np.asarray(read_h5_array(handle["S_last"]))
        algorithm = str(_attr(handle, "algorithm", ""))
        init = str(_attr(handle, "init", "zero"))
        result: dict[str, object] = {
            "solution": str(path_obj),
            "prob_id": str(_attr(handle, "problem_name", "")),
            "n": int(X.shape[0]),
            "m": int(np.asarray(read_h5_array(handle["b"])).size),
            "algorithm": algorithm,
            "init": init,
            "adapt": _adapt_label(handle),
            "max_iters": _attr(handle, "max_iters", None),
        }
        if init == "random":
            result["seed"] = _seed_value(handle)
        if algorithm == "pdhg":
            result["tau_0"] = float(_attr(handle, "tau_0", _attr(handle, "tau", float("nan"))))
            result["sigma_0"] = float(_attr(handle, "sigma_0", _attr(handle, "sigma", float("nan"))))
            if "Z_last" in handle:
                result["min_eig_|Z|"] = lambda_min_abs(np.asarray(read_h5_array(handle["Z_last"])))
        elif algorithm == "admm":
            result["rho_0"] = float(_attr(handle, "rho_0", _attr(handle, "rho", float("nan"))))
            if "Z_last" in handle:
                result["min_eig_|Z|"] = lambda_min_abs(np.asarray(read_h5_array(handle["Z_last"])))
        if check_nd:
            if maxcut:
                reports = maxcut_nondegeneracy_reports(X, S, tol=tol)
            else:
                reports = nondegeneracy_reports(X, S, read_sdp_operator(handle["A"]), tol=tol)
            result["primal_nd"] = _nd_dict(reports["primal"])
            result["dual_nd"] = _nd_dict(reports["dual"])
        return result
