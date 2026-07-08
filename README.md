# Local linear convergence of the primal-dual hybrid gradient (PDHG) algorithm for semidefinite programming

This codebas contains Python code and data utilities for numerical experiments on primal-dual first-order methods, in particular, PDHG and ADMM, for semidefinite programming (SDP). It accompanies the paper 
> Local linear convergence of the primal-dual hybrid gradient (PDHG) algorithm for semidefinite programming

It also includes a Python reimplementation of the experiments from 
> Local linear convergence of the alternating direction method of multipliers for semidefinite programming under strict complementarity \[[arXiv](https://arxiv.org/abs/2503.20142)\]

**Caveat:** This implementation is designed only to verify two phenomena in primal-dual first-order methods for SDP: local R-linear convergence and rank identification. It is not optimized for numerical stability or efficiency.

The primal-dual pair of SDPs considered here is

$$
\begin{array}{lll}
\text{Primal:}\quad &
\text{minimize} \quad & \langle C, X\rangle \\
& \text{subject to} \quad & \mathcal{A}(X) = b \\
& & X \succeq 0
\end{array}
\qquad \quad
\begin{array}{lll}
\text{Dual:} \quad &
\text{maximize} \quad & b^T y \\
& \text{subject to} \quad & \mathcal{A}^*y + S = C \\
& & S \succeq 0.
\end{array}
$$

The implementation supports basic PDHG and ADMM algorithms, SDPA sparse `.dat-s` input, conversion from supported MATLAB data, saved run artifacts, convergence plots, and solution diagnostics.

## Installation

The Python virtual environment is maintained by `uv`. From the repository root, run:

```bash
uv sync --extra dev
```

Commands can then be run with `uv run`:

```bash
uv run python -m cli --help
```

## Command Line Usage

### Basic Usage

Run PDHG on an SDPA instance:

```bash
uv run python -m cli solve \
  --input path/to/instance.dat-s \
  --algorithm pdhg \
  --tau 0.2 \
  --sigma 0.2 \
  --max-iters 1000 \
  --tol 1e-6 \
  --adapt none \
  --init zero \
  --period 100 \
  --output-dir results/example-run
```

Run ADMM on the same input:

```bash
uv run python -m cli solve \
  --input path/to/instance.dat-s \
  --algorithm admm \
  --rho 1.0 \
  --max-iters 1000 \
  --tol 1e-6 \
  --adapt first-k \
  --adapt-iters 100 \
  --init random \
  --seed 1 \
  --period 100 \
  --output-dir results/example-run
```

Common solve options:

- `--adapt`: parameter adaptation mode; use `none`, `always`, or `first-k`. With `first-k`, `--adapt-iters` sets the number of adaptive iterations.
- `--init`: initialization type; use `zero` for the origin or `random` for a Gaussian random initialization, optionally with `--seed`.
- `--period`: positive integer controlling how often progress is printed.

### Auxiliary Functions

Convert supported MATLAB data to SDPA format:

```bash
uv run python -m cli convert-mat --input path/to/instance.mat
```

Plot convergence from a saved history CSV:

```bash
uv run python -m cli plot \
  --history results/example-run/history.csv
```

Diagnose a saved solution:

```bash
uv run python -m cli diagnose \
  --solution results/example-run/solution.h5 \
  --check-nd true
```

Optionally rewrite dense solution matrices in sparse HDF5 form:

```bash
uv run python -m cli sparsify-h5 \
  results/example-run/solution.h5
```

Solver runs produce a summary JSON file, solution HDF5 file, iteration history CSV file, and convergence plot PNG file.
