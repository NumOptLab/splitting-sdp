"""Dense PDHG for standard-form SDP."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np

from core.operators import apply_A, apply_AT, operator_norm_squared
from core.psd import project_psd, symmetrize
from core.residuals import pdhg_residuals
from solvers.common import (
    SolverResult,
    adapt_is_enabled,
    history_row,
    init_xy,
    print_progress_header,
    print_progress_row,
    print_run_info,
    should_log_progress,
    validate_period,
)
from utils.dataio import SDPData

AdaptMode = Literal["none", "always", "first-k"]
InitMode = Literal["zero", "random"]


def pdhg_step(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau: float,
    sigma: float,
    *,
    theta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One fixed-step PDHG update from Eq. (2) in the PDHG-SDP paper."""
    X_next = project_psd(X - tau * (C - apply_AT(A, y)))
    X_bar = X_next + theta * (X_next - X)
    y_next = y + sigma * (b - apply_A(A, X_bar))
    return X_next, y_next


def _line_search_step(
    X: np.ndarray,
    y: np.ndarray,
    C: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau: float,
    beta: float,
    *,
    delta: float,
    backtrack: float,
    growth: float,
    max_ls_iters: int,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Malitsky-Pock Algorithm 1 line search for the SDP saddle form.

    We use ``K=-A``, ``g(X)=<C,X>+I_{S_+}(X)``, and
    ``f(z)=delta_{-b}(z)`` so that ``prox_{sigma f*}(q)=q+sigma*b``.
    The accepted test is
    ``sqrt(beta) * tau * ||K^*(y_new-y)|| <= delta * ||y_new-y||``.
    """
    tau_trial = tau * growth
    accepted_steps = 0
    for ls_iter in range(1, max_ls_iters + 1):
        theta_trial = tau_trial / tau
        sigma_trial = beta * tau_trial
        X_trial = project_psd(X - tau_trial * (C - apply_AT(A, y)))
        X_bar = X_trial + theta_trial * (X_trial - X)
        y_trial = y + sigma_trial * (b - apply_A(A, X_bar))
        dy = y_trial - y
        lhs = float(np.sqrt(beta) * tau_trial * np.linalg.norm(apply_AT(A, dy), ord="fro"))
        rhs = float(delta * np.linalg.norm(dy))
        if lhs <= rhs + 1e-15:
            accepted_steps = ls_iter
            return X_trial, y_trial, tau_trial, sigma_trial, accepted_steps
        tau_trial *= backtrack
    sigma_trial = beta * tau_trial
    X_trial = project_psd(X - tau_trial * (C - apply_AT(A, y)))
    X_bar = X_trial + (tau_trial / tau) * (X_trial - X)
    y_trial = y + sigma_trial * (b - apply_A(A, X_bar))
    return X_trial, y_trial, tau_trial, sigma_trial, max_ls_iters


def run_pdhg(
    data: SDPData,
    *,
    tau: float | None = None,
    sigma: float | None = None,
    theta: float = 1.0,
    max_iters: int = 1_000,
    tol: float = 1e-6,
    adapt: AdaptMode = "none",
    adapt_iters: int = 100,
    init: InitMode = "zero",
    seed: int | None = None,
    ls_delta: float = 0.99,
    ls_backtrack: float = 0.7,
    ls_growth: float = 1.02,
    max_ls_iters: int = 25,
    verbose: bool = True,
    period: int = 100,
) -> SolverResult:
    """Run dense PDHG with optional Malitsky-Pock line search."""
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if theta < 0.0:
        raise ValueError("theta must be nonnegative")
    if adapt_iters < 0:
        raise ValueError("adapt_iters must be nonnegative")
    validate_period(period)

    total_start = time.perf_counter()
    setup_start = time.perf_counter()
    L2 = max(operator_norm_squared(data.A), 1e-16)
    if tau is None and sigma is None:
        step = 0.99 / np.sqrt(L2)
        tau = step
        sigma = step
    elif tau is None:
        if sigma is None or sigma <= 0.0:
            raise ValueError("sigma must be positive")
        tau = 0.99 / (sigma * L2)
    elif sigma is None:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        sigma = 0.99 / (tau * L2)
    if tau <= 0.0 or sigma <= 0.0:
        raise ValueError("tau and sigma must be positive")
    # if adapt == "none" and tau * sigma * L2 > 4/3:
        # raise ValueError("Fixed PDHG steps require tau*sigma*||A||^2 < 4/3")

    tau_0 = tau
    sigma_0 = sigma
    beta = sigma / tau
    X, y = init_xy(data, init, seed)
    setup_time = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    history: list[dict[str, float | int | str]] = []
    metrics = pdhg_residuals(X, y, data.C, data.A, data.b)
    row0 = history_row(0, metrics, tau=tau, sigma=sigma, W_diff=0.0, elapsed=0.0)
    history.append(row0)
    if verbose:
        print_run_info(
            data,
            "pdhg",
            (
                f"init={init}",
                f"stepsize=tau={tau:.6g},sigma={sigma:.6g},theta={theta:.6g}",
                f"line-search={'on' if adapt != 'none' else 'off'}",
                f"adapt={adapt}",
                f"adapt_iters={adapt_iters}",
            ),
        )
        print_progress_header("pdhg", ("tau", "sigma"))
        print_progress_row(row0, total_elapsed=time.perf_counter() - total_start, extra_values=(tau, sigma))
    status = "max_iters_reached"
    reason = "maximum iterations reached"
    total_line_search_steps = 0
    X_prev = X.copy()
    S_prev = symmetrize(data.C - apply_AT(data.A, y))

    for iteration in range(1, max_iters + 1):
        W_prev = symmetrize(X - tau * symmetrize(data.C - apply_AT(data.A, y)))
        X_prev = X.copy()
        S_prev = symmetrize(data.C - apply_AT(data.A, y))
        if adapt_is_enabled(adapt, iteration, adapt_iters):
            X, y, tau, sigma, accepted = _line_search_step(
                X,
                y,
                data.C,
                data.A,
                data.b,
                tau,
                beta,
                delta=ls_delta,
                backtrack=ls_backtrack,
                growth=ls_growth,
                max_ls_iters=max_ls_iters,
            )
            total_line_search_steps += accepted
        else:
            X, y = pdhg_step(X, y, data.C, data.A, data.b, tau, sigma, theta=theta)
            accepted = 0

        metrics = pdhg_residuals(X, y, data.C, data.A, data.b)
        elapsed = time.perf_counter() - solve_start
        S_current = symmetrize(data.C - apply_AT(data.A, y))
        W_next = symmetrize(X - tau * S_current)
        W_diff = float(np.linalg.norm(W_next - W_prev, ord="fro"))
        row = history_row(
            iteration,
            metrics,
            tau=tau,
            sigma=sigma,
            alpha=theta,
            W_diff=W_diff,
            accepted_line_search_steps=accepted,
            elapsed=elapsed,
        )
        history.append(row)
        converged = metrics.kkt_res <= tol
        if verbose and should_log_progress(iteration, period, is_final=converged or iteration == max_iters):
            print_progress_row(row, total_elapsed=time.perf_counter() - total_start, extra_values=(tau, sigma))
        if converged:
            status = "converged"
            reason = "kkt_res <= tol"
            break

    solve_time = time.perf_counter() - solve_start
    S = symmetrize(data.C - apply_AT(data.A, y))
    W_last = symmetrize(X - tau * S)
    total_time = time.perf_counter() - total_start
    return SolverResult(
        algorithm="pdhg",
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
            "tau_0": tau_0,
            "sigma_0": sigma_0,
            "tau": tau,
            "sigma": sigma,
            "theta": theta,
            "max_iters": max_iters,
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
        W_last=W_last,
        line_search_summary={"accepted_steps_total": total_line_search_steps, "beta": beta},
    )
