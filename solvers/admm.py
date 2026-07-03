"""Sparse-aware ADMM for standard-form SDP."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from core.operators import GramSolver, as_sdp_operator, build_gram_solver
from core.psd import project_psd_with_eigs, symmetrize
from core.residuals import admm_residuals
from solvers.common import (
    SolverResult,
    adapt_is_enabled,
    history_row,
    init_xys,
    print_progress_header,
    print_progress_row,
    print_final_summary_table,
    print_run_info_table,
    should_log_progress,
    validate_period,
)
from utils.dataio import SDPData

AdaptMode = Literal["none", "always", "first-k"]
InitMode = Literal["zero", "random"]


def drs_step(
    Z: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    rho: float,
    *,
    solver: GramSolver | None = None,
) -> np.ndarray:
    """One DRS/ADMM step from Eq. (4) in the ADMM-SDP paper."""
    op = as_sdp_operator(A, n=C.shape[0])
    solver = build_gram_solver(op) if solver is None else solver
    PZ = project_psd_with_eigs(Z).matrix
    argument = -2.0 * PZ + Z
    projection = op.adjoint(solver.solve(op.apply(argument)))
    b_term = op.adjoint(solver.solve(b))
    c_term = rho * op.adjoint(solver.solve(op.apply(C)))
    return symmetrize(projection + PZ + b_term + c_term - rho * C)


def run_admm(
    data: SDPData,
    *,
    rho: float = 1.0,
    max_iters: int = 1_000,
    tol: float = 1e-6,
    adapt: AdaptMode = "none",
    adapt_iters: int = 100,
    init: InitMode = "zero",
    seed: int | None = None,
    adaptive_mu: float = 10.0,
    adaptive_tau: float = 2.0,
    rho_min: float = 1e-8,
    rho_max: float = 1e8,
    verbose: bool = True,
    period: int = 100,
) -> SolverResult:
    """Run ADMM using Eqs. (2a)-(2c) from the ADMM-SDP paper."""
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if adapt_iters < 0:
        raise ValueError("adapt_iters must be nonnegative")
    validate_period(period)

    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    op = as_sdp_operator(data.A, n=data.n)
    solver = build_gram_solver(op)
    gram = solver.gram
    rho_0 = rho
    X, y, S = init_xys(data, init, seed)
    AX = op.apply(X)
    AS = op.apply(S)
    AC = op.apply(data.C)
    ATy = op.adjoint(y)
    setup_time = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    history: list[dict[str, float | int | str]] = []
    metrics = admm_residuals(X, y, S, data.C, op, data.b, AX=AX, ATy=ATy)
    row0 = history_row(0, metrics, rho=rho, Z_diff=0.0, elapsed=0.0)
    history.append(row0)
    if verbose:
        initial_stepsize = f"rho={rho:.6g}"
        print_run_info_table(
            data,
            "admm",
            initial_stepsize,
            adapt,
            adapt_iters,
            init,
            tol,
            max_iters,
        )
        print_progress_header("admm", ("rho",))
        print_progress_row(row0, total_elapsed=time.perf_counter() - total_start, extra_values=(rho,))
    status = "max_iters_reached"
    reason = "maximum iterations reached"
    tuning_events = 0
    X_prev = X.copy()
    S_prev = S.copy()

    for iteration in range(1, max_iters + 1):
        Z_prev = symmetrize(X - rho * S)
        S_prev = S.copy()
        X_prev = X.copy()
        ATy_prev = ATy.copy()

        rhs = (data.b / rho) - (AX / rho + AS - AC)
        y = solver.solve(rhs)
        ATy = op.adjoint(y)
        projection = project_psd_with_eigs(data.C - ATy - X / rho)
        S = projection.matrix
        AS = op.apply(S)
        primal_link = S + ATy - data.C
        X = symmetrize(X + rho * primal_link)
        AX = AX + rho * (AS + np.asarray(gram @ y, dtype=float).reshape(-1) - AC)

        tuning_event = ""
        if adapt_is_enabled(adapt, iteration, adapt_iters):
            dual_step = (ATy - ATy_prev) + (S - S_prev)
            r_norm = float(np.linalg.norm(primal_link, ord="fro"))
            s_norm = float(rho * np.linalg.norm(dual_step, ord="fro"))
            old_rho = rho
            if r_norm > adaptive_mu * s_norm:
                rho = min(rho * adaptive_tau, rho_max)
            elif s_norm > adaptive_mu * r_norm:
                rho = max(rho / adaptive_tau, rho_min)
            if rho != old_rho:
                tuning_events += 1
                tuning_event = f"rho={rho:.6g}"

        metrics = admm_residuals(
            X,
            y,
            S,
            data.C,
            op,
            data.b,
            AX=AX,
            ATy=ATy,
            S_eigs=projection.clipped_eigvals,
        )
        elapsed = time.perf_counter() - solve_start
        Z_next = symmetrize(X - rho * S)
        Z_diff = float(np.linalg.norm(Z_next - Z_prev, ord="fro"))
        row = history_row(
            iteration,
            metrics,
            rho=rho,
            Z_diff=Z_diff,
            tuning_event=tuning_event,
            elapsed=elapsed,
        )
        history.append(row)
        converged = metrics.kkt_res <= tol
        if verbose and should_log_progress(iteration, period, is_final=converged or iteration == max_iters):
            print_progress_row(row, total_elapsed=time.perf_counter() - total_start, extra_values=(rho,))
        if converged:
            status = "converged"
            reason = "kkt_res <= tol"
            break

    solve_time = time.perf_counter() - solve_start
    Z_last = symmetrize(X - rho * S)
    total_time = time.perf_counter() - total_start
    if verbose:
        print_final_summary_table(
            data,
            "admm",
            f"rho={rho_0:.6g}",
            f"rho={rho:.6g}",
            adapt,
            adapt_iters,
            init,
            tol,
            max_iters,
            status,
            int(history[-1]["iter"]),
            X,
            S,
            Z_last,
        )
    return SolverResult(
        algorithm="admm",
        status=status,
        termination_reason=reason,
        iterations=int(history[-1]["iter"]),
        X=X,
        y=y,
        S=S,
        history=history,
        metrics=metrics,
        setup_time=setup_time,
        solve_time=solve_time,
        total_time=total_time,
        params={
            "rho_0": rho_0,
            "rho": rho,
            "max_iters": max_iters,
            "tol": tol,
            "adapt": adapt,
            "adapt_iters": adapt_iters,
            "init": init,
            "seed": -1 if seed is None else seed,
            "verbose": verbose,
            "period": period,
        },
        data=data,
        X_prev=X_prev,
        S_prev=S_prev,
        Z_last=Z_last,
        tuning_summary={"events": tuning_events, "final_rho": rho},
    )
