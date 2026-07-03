"""Shared solver helpers."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import TextIO

import numpy as np

from core.psd import lambda_min_abs, min_positive_eig, psd_rank
from core.residuals import ResidualMetrics
from utils.dataio import SDPData


@dataclass
class SolverResult:
    algorithm: str
    status: str
    termination_reason: str
    iterations: int
    X: np.ndarray
    y: np.ndarray
    S: np.ndarray
    history: list[dict[str, float | int | str]]
    metrics: ResidualMetrics
    setup_time: float
    solve_time: float
    total_time: float
    params: dict[str, float | int | str | bool]
    data: SDPData
    X_prev: np.ndarray | None = None
    S_prev: np.ndarray | None = None
    Z_last: np.ndarray | None = None
    line_search_summary: dict[str, float | int | str] | None = None
    tuning_summary: dict[str, float | int | str] | None = None


def adapt_is_enabled(mode: str, iteration: int, adapt_iters: int) -> bool:
    if mode == "none":
        return False
    if mode == "always":
        return True
    if mode == "first-k":
        return iteration <= adapt_iters
    raise ValueError("adapt must be one of: none, always, first-k")


def validate_period(period: int) -> None:
    if period <= 0:
        raise ValueError("period must be positive")


def should_log_progress(iteration: int, period: int, *, is_final: bool = False) -> bool:
    return iteration == 0 or iteration % period == 0 or is_final


def print_run_info(
    data: SDPData,
    algorithm: str,
    details: tuple[str, ...],
    *,
    stream: TextIO | None = None,
) -> None:
    stream = sys.stderr if stream is None else stream
    print(f"SDP info: name={data.problem_name} n={data.n} m={data.m} block_sizes={data.block_sizes}", file=stream)
    print(f"Solver info: algorithm={algorithm.upper()} {' '.join(details)}", file=stream)


def _format_table_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, tuple):
        return ", ".join(_format_table_value(item) for item in value)
    return str(value)


def _print_key_value_table(
    title: str,
    rows: tuple[tuple[str, object], ...],
    *,
    stream: TextIO | None = None,
) -> None:
    stream = sys.stderr if stream is None else stream
    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(_format_table_value(value)) for _, value in rows)
    border = "+" + "-" * (key_width + 2) + "+" + "-" * (value_width + 2) + "+"
    print(title, file=stream)
    print(border, file=stream)
    for key, value in rows:
        formatted = _format_table_value(value)
        print(f"| {key:<{key_width}} | {formatted:<{value_width}} |", file=stream)
    print(border, file=stream)


def adapt_label(mode: str, adapt_iters: int) -> str:
    if mode == "first-k":
        return f"first-k ({adapt_iters})"
    return mode


def run_info_rows(
    data: SDPData,
    algorithm: str,
    initial_stepsize: str,
    adapt: str,
    adapt_iters: int,
    init: str,
    tol: float,
    max_iters: int,
) -> tuple[tuple[str, object], ...]:
    return (
        ("SDP name", data.problem_name),
        ("n", data.n),
        ("m", data.m),
        ("algorithm", algorithm.upper()),
        ("initial stepsize", initial_stepsize),
        ("adapt", adapt_label(adapt, adapt_iters)),
        ("initialization", init),
        ("tolerance", tol),
        ("maximum iteration", max_iters),
    )


def print_run_info_table(
    data: SDPData,
    algorithm: str,
    initial_stepsize: str,
    adapt: str,
    adapt_iters: int,
    init: str,
    tol: float,
    max_iters: int,
    *,
    stream: TextIO | None = None,
) -> None:
    rows = run_info_rows(data, algorithm, initial_stepsize, adapt, adapt_iters, init, tol, max_iters)
    _print_key_value_table("Run configuration", rows, stream=stream)


def print_final_summary_table(
    data: SDPData,
    algorithm: str,
    initial_stepsize: str,
    final_stepsize: str,
    adapt: str,
    adapt_iters: int,
    init: str,
    tol: float,
    max_iters: int,
    status: str,
    iteration: int,
    X: np.ndarray,
    S: np.ndarray,
    Z: np.ndarray | None,
    *,
    stream: TextIO | None = None,
) -> None:
    rows = run_info_rows(data, algorithm, initial_stepsize, adapt, adapt_iters, init, tol, max_iters)
    rows = rows + (
        ("final stepsize", final_stepsize),
        ("final iteration", iteration),
        ("solver status", status),
        ("rank_X", psd_rank(X)),
        ("rank_S", psd_rank(S)),
        ("min_pos_eig_X", min_positive_eig(X)),
        ("min_pos_eig_S", min_positive_eig(S)),
        ("min_eig_|Z|", None if Z is None else lambda_min_abs(Z)),
    )
    _print_key_value_table("Final summary", rows, stream=stream)


def print_progress_header(
    algorithm: str,
    extra_columns: tuple[str, ...],
    *,
    include_complementarity: bool = False,
    stream: TextIO | None = None,
) -> None:
    stream = sys.stderr if stream is None else stream
    columns = ["iter", "kkt_res", "primal_res", "dual_res"]
    if include_complementarity:
        columns.append("compl")
    columns.extend(("rel_gap", "rank_X", "time", *extra_columns))
    print(f"{algorithm.upper()} progress", file=stream)
    print(" ".join(f"{column:>12}" for column in columns), file=stream)


def print_progress_row(
    row: dict[str, float | int | str],
    *,
    total_elapsed: float,
    extra_values: tuple[float, ...],
    include_complementarity: bool = False,
    stream: TextIO | None = None,
) -> None:
    stream = sys.stderr if stream is None else stream
    values = [
        f"{int(row['iter']):12d}",
        f"{float(row['kkt_res']):12.4e}",
        f"{float(row['primal_res']):12.4e}",
        f"{float(row['dual_res']):12.4e}",
    ]
    if include_complementarity:
        values.append(f"{float(row['compl']):12.4e}")
    values.extend(
        [
            f"{float(row['rel_gap']):12.4e}",
            f"{int(row['rank_X']):12d}",
            f"{total_elapsed:12.3f}",
        ]
    )
    values.extend(f"{value:12.4e}" for value in extra_values)
    print(" ".join(values), file=stream)


def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def symmetric_normal(rng: np.random.Generator, n: int) -> np.ndarray:
    M = rng.standard_normal((n, n))
    return 0.5 * (M + M.T)


def init_xy(data: SDPData, init: str, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    if init == "zero":
        return np.zeros((data.n, data.n), dtype=float), np.zeros(data.m, dtype=float)
    if init == "random":
        rng = rng_from_seed(seed)
        return symmetric_normal(rng, data.n), rng.standard_normal(data.m)
    raise ValueError("init must be 'zero' or 'random'")


def init_xys(data: SDPData, init: str, seed: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if init == "zero":
        return (
            np.zeros((data.n, data.n), dtype=float),
            np.zeros(data.m, dtype=float),
            np.zeros((data.n, data.n), dtype=float),
        )
    if init == "random":
        rng = rng_from_seed(seed)
        return symmetric_normal(rng, data.n), rng.standard_normal(data.m), symmetric_normal(rng, data.n)
    raise ValueError("init must be 'zero' or 'random'")


def history_row(
    iteration: int,
    metrics: ResidualMetrics,
    *,
    rho: float = float("nan"),
    tau: float = float("nan"),
    sigma: float = float("nan"),
    alpha: float = float("nan"),
    Z_diff: float | None = None,
    accepted_line_search_steps: int = 0,
    tuning_event: str = "",
    include_complementarity: bool = False,
    elapsed: float = 0.0,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "iter": int(iteration),
        "primal_obj": metrics.primal_obj,
        "dual_obj": metrics.dual_obj,
        "gap": metrics.gap,
        "rel_gap": metrics.rel_gap,
        "primal_res": metrics.primal_res,
        "dual_res": metrics.dual_res,
        "kkt_res": metrics.kkt_res,
        "rank_X": metrics.rank_X,
        "min_eig_X": metrics.min_eig_X,
        "min_eig_S": metrics.min_eig_S,
        "rho": rho,
        "tau": tau,
        "sigma": sigma,
        "alpha": alpha,
        "accepted_line_search_steps": int(accepted_line_search_steps),
        "tuning_event": tuning_event,
        "time": elapsed,
    }
    if include_complementarity:
        row["compl"] = metrics.complementarity
    if Z_diff is not None:
        row["Z_diff"] = Z_diff
    return row
