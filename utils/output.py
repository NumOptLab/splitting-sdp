"""Run artifact writing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from core.psd import eigvalsh, lambda_min_abs
from solvers.common import SolverResult
from utils.h5io import write_maybe_sparse, write_sdp_operator
from version import __version__


def results_dir() -> Path:
    """Return the package-local results directory."""
    path = Path(__file__).resolve().parents[1] / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str) -> str:
    """Make a string safe for local artifact filenames."""
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", value).strip("_")


def run_stem(result: SolverResult) -> str:
    data = result.data
    if result.algorithm == "pdhg":
        param = f"tau={float(result.params['tau']):.3g}_sigma={float(result.params['sigma']):.3g}"
    else:
        param = f"rho={float(result.params['rho']):.3g}"
    adapt = str(result.params.get("adapt", "none"))
    adapt_iters = int(result.params.get("adapt_iters", 0))
    if adapt == "first-k":
        adapt_label = f"adapt=first-k{adapt_iters}"
    else:
        adapt_label = f"adapt={adapt}"
    init = f"init={result.params.get('init', 'zero')}"
    return safe_filename(f"{data.problem_name}_{result.algorithm}_{param}_{adapt_label}_{init}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _adapt_label(params: dict[str, Any]) -> str:
    adapt = str(params.get("adapt", "none"))
    if adapt == "first-k":
        return f"first-k{int(params.get('adapt_iters', 0))}"
    return adapt


def _summary_status(status: str) -> str:
    if status == "converged":
        return "solved"
    return status


def _seed_value(params: dict[str, Any]) -> int | None:
    seed = params.get("seed")
    if seed in (None, -1):
        return None
    return int(seed)


def write_history(result: SolverResult, stem: str) -> Path:
    path = results_dir() / f"{stem}_history.csv"
    pd.DataFrame(result.history).to_csv(path, index=False)
    return path


def write_solution(result: SolverResult, stem: str) -> Path:
    path = results_dir() / f"{stem}_solution.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X_last", data=result.X)
        handle.create_dataset("y_last", data=result.y)
        write_maybe_sparse(handle, "S_last", result.S)
        if result.X_prev is not None:
            handle.create_dataset("X_prev", data=result.X_prev)
        if result.S_prev is not None:
            write_maybe_sparse(handle, "S_prev", result.S_prev)
        if result.Z_last is not None:
            handle.create_dataset("Z_last", data=result.Z_last)
            handle.create_dataset("eig_Z_last", data=eigvalsh(result.Z_last))
        write_maybe_sparse(handle, "C", result.data.C)
        write_sdp_operator(handle, "A", result.data.A, n=result.data.n)
        handle.create_dataset("b", data=result.data.b)
        handle.create_dataset("eig_X_last", data=eigvalsh(result.X))
        handle.create_dataset("eig_S_last", data=eigvalsh(result.S))
        handle.attrs["solver_name"] = "pdhg-sdp"
        handle.attrs["solver_version"] = __version__
        handle.attrs["algorithm"] = result.algorithm
        handle.attrs["problem_name"] = result.data.problem_name
        handle.attrs["input_path"] = "" if result.data.input_path is None else result.data.input_path
        handle.attrs["input_format"] = result.data.input_format
        for key, value in result.params.items():
            handle.attrs[key] = _jsonable(value)
    return path


def write_summary(result: SolverResult, stem: str, history_path: Path, solution_path: Path, plot_paths: list[Path]) -> Path:
    path = results_dir() / f"{stem}_summary.json"
    metrics = result.metrics.as_dict()
    params = result.params
    summary = {
        "solver_name": "pdhg-sdp",
        "solver_version": __version__,
        "prob_id": result.data.problem_name,
        "algorithm": result.algorithm,
        "problem_name": result.data.problem_name,
        "input_path": result.data.input_path,
        "input_format": result.data.input_format,
        "n": result.data.n,
        "m": result.data.m,
        "block_sizes": result.data.block_sizes,
        "status": _summary_status(result.status),
        "total_iter": result.iterations,
        "iterations": result.iterations,
        "max_iters": params.get("max_iters"),
        "stopping_tolerance": params.get("tol"),
        "primal_objective": metrics["primal_obj"],
        "dual_objective": metrics["dual_obj"],
        "absolute_gap": metrics["gap"],
        "relative_gap": metrics["rel_gap"],
        "primal_residual": metrics["primal_res"],
        "dual_residual": metrics["dual_res"],
        "max_kkt_residual": metrics["kkt_res"],
        "rho_0": params.get("rho_0"),
        "rho": params.get("rho"),
        "tau_0": params.get("tau_0"),
        "tau": params.get("tau"),
        "sigma_0": params.get("sigma_0"),
        "sigma": params.get("sigma"),
        "min_eig_|Z|": None if result.Z_last is None else lambda_min_abs(result.Z_last),
        "adapt": _adapt_label(params),
        "adaptation_mode": params.get("adapt"),
        "adapt_iters": params.get("adapt_iters"),
        "init": params.get("init"),
        "line_search_summary": result.line_search_summary or {},
        "tuning_summary": result.tuning_summary or {},
        "setup_time": result.setup_time,
        "solve_time": result.solve_time,
        "total_time": result.total_time,
        "termination_reason": result.termination_reason,
        "output_file_paths": {
            "summary": str(path),
            "solution": str(solution_path),
            "history": str(history_path),
            "plots": [str(plot_path) for plot_path in plot_paths],
        },
    }
    if params.get("init") == "random":
        summary["seed"] = _seed_value(params)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_jsonable)
    return path


def write_run_artifacts(result: SolverResult, *, make_plot: bool = True) -> dict[str, Path | list[Path]]:
    from utils.plotting import plot_convergence

    stem = run_stem(result)
    history_path = write_history(result, stem)
    solution_path = write_solution(result, stem)
    plot_paths: list[Path] = []
    if make_plot:
        plot_paths = plot_convergence(history_path, stem=stem)
    summary_path = write_summary(result, stem, history_path, solution_path, plot_paths)
    return {
        "summary": summary_path,
        "solution": solution_path,
        "history": history_path,
        "plots": plot_paths,
    }
