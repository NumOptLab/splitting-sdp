"""Objectives and residuals for dense SDP iterates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from core.operators import apply_A, apply_AT
from core.psd import PSDProjection, eigen_info_from_values, eigvalsh, project_psd_with_eigs, symmetrize


@dataclass(frozen=True)
class ResidualMetrics:
    primal_obj: float
    dual_obj: float
    gap: float
    rel_gap: float
    primal_res: float
    dual_res: float
    kkt_res: float
    complementarity: float
    rank_X: int
    min_eig_X: float
    min_eig_S: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def objective(C: np.ndarray, X: np.ndarray) -> float:
    """Return ``<C, X>``."""
    return float(np.sum(np.asarray(C, dtype=float) * np.asarray(X, dtype=float)))


def _gap_metrics(C: np.ndarray, X: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    primal_obj = objective(C, X)
    dual_obj = float(np.asarray(b, dtype=float).reshape(-1) @ np.asarray(y, dtype=float).reshape(-1))
    gap = primal_obj - dual_obj
    rel_gap = gap / (1.0 + abs(primal_obj) + abs(dual_obj))
    return primal_obj, dual_obj, gap, rel_gap


def _rank_min_from_eigs(values: np.ndarray) -> tuple[int, float]:
    info = eigen_info_from_values(values)
    return info.rank, info.min_eig


def admm_residuals(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *,
    AX: np.ndarray | None = None,
    ATy: np.ndarray | None = None,
    X_eigs: np.ndarray | None = None,
    S_eigs: np.ndarray | None = None,
) -> ResidualMetrics:
    """Residuals for ADMM, using primal equality, dual equality, and objective gap."""
    Xs = symmetrize(X)
    Ss = symmetrize(S)
    primal_obj, dual_obj, gap, rel_gap = _gap_metrics(C, Xs, b, y)
    AX = apply_A(A, Xs) if AX is None else np.asarray(AX, dtype=float).reshape(-1)
    ATy = apply_AT(A, y) if ATy is None else symmetrize(ATy)
    primal_res = float(np.linalg.norm(AX - b) / (1.0 + np.linalg.norm(b)))
    dual_res = float(np.linalg.norm(ATy + Ss - C, ord="fro") / (1.0 + np.linalg.norm(C, ord="fro")))
    complementarity = float(np.sum(Xs * Ss) / (1.0 + abs(primal_obj) + abs(dual_obj)))
    if X_eigs is None:
        x_info = eigen_info_from_values(eigvalsh(Xs))
        rank_x, min_x = x_info.rank, x_info.min_eig
    else:
        rank_x, min_x = _rank_min_from_eigs(X_eigs)
    if S_eigs is None:
        min_s = eigen_info_from_values(eigvalsh(Ss)).min_eig
    else:
        _, min_s = _rank_min_from_eigs(S_eigs)
    kkt_res = max(primal_res, dual_res, rel_gap)
    return ResidualMetrics(
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        gap=gap,
        rel_gap=rel_gap,
        primal_res=primal_res,
        dual_res=dual_res,
        kkt_res=kkt_res,
        complementarity=complementarity,
        rank_X=rank_x,
        min_eig_X=min_x,
        min_eig_S=min_s,
    )


def pdhg_residuals(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *,
    AX: np.ndarray | None = None,
    S: np.ndarray | None = None,
    S_projection: PSDProjection | None = None,
    X_eigs: np.ndarray | None = None,
) -> ResidualMetrics:
    """Residuals for PDHG.

    PDHG constructs ``S = C - A^* y`` exactly, so the dual residual is the
    squared Frobenius distance from this slack to the PSD cone.
    """
    Xs = symmetrize(X)
    S = symmetrize(C - apply_AT(A, y)) if S is None else symmetrize(S)
    primal_obj, dual_obj, gap, rel_gap = _gap_metrics(C, Xs, b, y)
    AX = apply_A(A, Xs) if AX is None else np.asarray(AX, dtype=float).reshape(-1)
    primal_res = float(np.linalg.norm(AX - b) / (1.0 + np.linalg.norm(b)))
    if X_eigs is None:
        x_info = eigen_info_from_values(eigvalsh(Xs))
        rank_x, min_x = x_info.rank, x_info.min_eig
    else:
        rank_x, min_x = _rank_min_from_eigs(X_eigs)
    S_projection = project_psd_with_eigs(S) if S_projection is None else S_projection
    min_s = float(S_projection.eigvals[0]) if S_projection.eigvals.size else 0.0
    dual_res = float(S_projection.negative_norm_squared / (1.0 + np.linalg.norm(C, ord="fro")))
    complementarity = float(np.sum(Xs * S_projection.matrix) / (1.0 + abs(primal_obj) + abs(dual_obj)))
    kkt_res = max(primal_res, dual_res, abs(rel_gap), abs(complementarity), max(0.0, -min_x))
    return ResidualMetrics(
        primal_obj=primal_obj,
        dual_obj=dual_obj,
        gap=gap,
        rel_gap=rel_gap,
        primal_res=primal_res,
        dual_res=dual_res,
        kkt_res=kkt_res,
        complementarity=complementarity,
        rank_X=rank_x,
        min_eig_X=min_x,
        min_eig_S=min_s,
    )
