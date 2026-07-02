"""Objectives and residuals for dense SDP iterates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from core.operators import apply_A, apply_AT
from core.psd import min_eig, project_psd, psd_rank, symmetrize


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


def admm_residuals(X: np.ndarray, y: np.ndarray, S: np.ndarray, C: np.ndarray, A: np.ndarray, b: np.ndarray) -> ResidualMetrics:
    """Residuals for ADMM, using primal equality, dual equality, and objective gap."""
    Xs = symmetrize(X)
    Ss = symmetrize(S)
    primal_obj, dual_obj, gap, rel_gap = _gap_metrics(C, Xs, b, y)
    primal_res = float(np.linalg.norm(apply_A(A, Xs) - b) / (1.0 + np.linalg.norm(b)))
    dual_res = float(np.linalg.norm(apply_AT(A, y) + Ss - C, ord="fro") / (1.0 + np.linalg.norm(C, ord="fro")))
    complementarity = float(np.sum(Xs * Ss) / (1.0 + abs(primal_obj) + abs(dual_obj)))
    min_x = min_eig(Xs)
    min_s = min_eig(Ss)
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
        rank_X=psd_rank(Xs),
        min_eig_X=min_x,
        min_eig_S=min_s,
    )


def pdhg_residuals(X: np.ndarray, y: np.ndarray, C: np.ndarray, A: np.ndarray, b: np.ndarray) -> ResidualMetrics:
    """Residuals for PDHG.

    PDHG constructs ``S = C - A^* y`` exactly, so the dual residual is the
    squared Frobenius distance from this slack to the PSD cone.
    """
    Xs = symmetrize(X)
    S = symmetrize(C - apply_AT(A, y))
    primal_obj, dual_obj, gap, rel_gap = _gap_metrics(C, Xs, b, y)
    primal_res = float(np.linalg.norm(apply_A(A, Xs) - b) / (1.0 + np.linalg.norm(b)))
    min_x = min_eig(Xs)
    min_s = min_eig(S)
    S_psd = project_psd(S)
    dual_res = float(np.linalg.norm(S - S_psd, ord="fro") ** 2 / (1.0 + np.linalg.norm(C, ord="fro")))
    complementarity = float(np.sum(Xs * S_psd) / (1.0 + np.linalg.norm(Xs, ord="fro") * np.linalg.norm(S_psd, ord="fro")))
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
        rank_X=psd_rank(Xs),
        min_eig_X=min_x,
        min_eig_S=min_s,
    )
