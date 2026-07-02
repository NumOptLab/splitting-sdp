"""Command-line interface for dense SDP experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solvers.admm import run_admm
from solvers.pdhg import run_pdhg
from utils.dataio import mat_to_dat_s, read_dat_s
from utils.diagnostics import diagnose_solution
from utils.h5io import rewrite_solution_sparse
from utils.output import write_run_artifacts
from utils.plotting import plot_convergence


def _bool_arg(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _load_input(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".mat":
        path = str(mat_to_dat_s(path))
    return read_dat_s(path)


def solve_command(args: argparse.Namespace) -> None:
    data = _load_input(args.input)
    common = {
        "max_iters": args.max_iters,
        "tol": args.tol,
        "adapt": args.adapt,
        "adapt_iters": args.adapt_iters,
        "init": args.init,
        "seed": args.seed,
        "verbose": args.verbose,
        "period": args.period,
    }
    if args.algorithm == "pdhg":
        result = run_pdhg(data, tau=args.tau, sigma=args.sigma, theta=args.theta, **common)
    else:
        result = run_admm(data, rho=args.rho, **common)
    result.params["tol"] = args.tol
    artifacts = write_run_artifacts(result, make_plot=True)
    print(json.dumps({key: [str(p) for p in value] if isinstance(value, list) else str(value) for key, value in artifacts.items()}, indent=2))


def convert_mat_command(args: argparse.Namespace) -> None:
    print(str(mat_to_dat_s(args.input)))


def plot_command(args: argparse.Namespace) -> None:
    paths = plot_convergence(args.history)
    print(json.dumps([str(path) for path in paths], indent=2))


def _format_diagnostic_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8e}"
    if value is None:
        return "n/a"
    return str(value)


def format_diagnostics(report: dict[str, object]) -> str:
    scalar_order = [
        "prob_id",
        "n",
        "m",
        "algorithm",
        "init",
        "seed",
        "tau_0",
        "sigma_0",
        "rho_0",
        "adapt",
        "max_iters",
        "min_eig_|Z|",
    ]
    labels = {
        "prob_id": "prob_id",
        "n": "n",
        "m": "m",
        "algorithm": "algorithm",
        "init": "init",
        "seed": "seed",
        "tau_0": "tau_0",
        "sigma_0": "sigma_0",
        "rho_0": "rho_0",
        "adapt": "adapt",
        "max_iters": "max_iters",
        "min_eig_|Z|": "min_eig_|Z|",
    }
    rows = [(labels[key], report[key]) for key in scalar_order if key in report]
    key_width = max(len(label) for label, _ in rows)
    lines = ["Diagnostics"]
    lines.extend(f"  {label:<{key_width}}  {_format_diagnostic_value(value)}" for label, value in rows)

    nd_rows = []
    for key, label in (("primal_nd", "primal ND"), ("dual_nd", "dual ND")):
        if key in report:
            item = report[key]
            if not isinstance(item, dict):
                continue
            nd_rows.append(
                (
                    label,
                    _format_diagnostic_value(item["holds"]),
                    int(item["rank_left"]),
                    int(item["rank_right"]),
                    int(item["rank_combined"]),
                    float(item["tolerance"]),
                )
            )
    if nd_rows:
        lines.append("")
        lines.append("Nondegeneracy")
        lines.append(f"  {'entry':<9} {'holds':>7} {'rank_left':>10} {'rank_right':>11} {'rank_combined':>14} {'tolerance':>12}")
        for label, holds, rank_left, rank_right, rank_combined, tolerance in nd_rows:
            lines.append(f"  {label:<9} {holds:>7} {rank_left:10d} {rank_right:11d} {rank_combined:14d} {tolerance:12.4e}")
    return "\n".join(lines)


def diagnose_command(args: argparse.Namespace) -> None:
    report = diagnose_solution(args.solution, check_nd=args.check_nd, tol=args.tol, maxcut=args.maxcut)
    print(format_diagnostics(report))


def sparsify_h5_command(args: argparse.Namespace) -> None:
    paths = [rewrite_solution_sparse(path) for path in args.solutions]
    print(json.dumps([str(path) for path in paths], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdhg-sdp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve")
    solve.add_argument("--input", required=True)
    solve.add_argument("--algorithm", choices=["pdhg", "admm"], required=True)
    solve.add_argument("--tau", type=float, default=None)
    solve.add_argument("--sigma", type=float, default=None)
    solve.add_argument("--rho", type=float, default=1.0)
    solve.add_argument("--theta", type=float, default=1.0)
    solve.add_argument("--max-iters", type=int, default=1000000)
    solve.add_argument("--tol", type=float, default=1e-8)
    solve.add_argument("--adapt", choices=["none", "always", "first-k"], default="none")
    solve.add_argument("--adapt-iters", type=int, default=100)
    solve.add_argument("--init", choices=["zero", "random"], default="zero")
    solve.add_argument("--seed", type=int, default=None)
    solve.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    solve.add_argument("--period", type=_positive_int_arg, default=100)
    solve.set_defaults(func=solve_command)

    convert = subparsers.add_parser("convert-mat")
    convert.add_argument("--input", required=True)
    convert.set_defaults(func=convert_mat_command)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--history", required=True)
    plot.add_argument("--summary", default=None)
    plot.set_defaults(func=plot_command)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--solution", required=True)
    diagnose.add_argument("--check-nd", type=_bool_arg, default=False)
    diagnose.add_argument("--maxcut", action="store_true", default=False)
    diagnose.add_argument("--tol", type=float, default=1e-8)
    diagnose.set_defaults(func=diagnose_command)

    sparsify = subparsers.add_parser("sparsify-h5")
    sparsify.add_argument("solutions", nargs="+")
    sparsify.set_defaults(func=sparsify_h5_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
