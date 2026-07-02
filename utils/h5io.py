"""HDF5 helpers for dense and sparse SDP run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse

from core.operators import operator_matrix, svec_index


def is_sparse_group(node: h5py.Dataset | h5py.Group) -> bool:
    return isinstance(node, h5py.Group) and node.attrs.get("storage") == "sparse_csr"


def write_csr_group(parent: h5py.File | h5py.Group, name: str, matrix: sparse.spmatrix, attrs: dict[str, Any] | None = None) -> None:
    csr = matrix.tocsr()
    group = parent.create_group(name)
    group.attrs["storage"] = "sparse_csr"
    group.attrs["shape"] = np.asarray(csr.shape, dtype=np.int64)
    if attrs:
        for key, value in attrs.items():
            group.attrs[key] = value
    group.create_dataset("data", data=csr.data)
    group.create_dataset("indices", data=csr.indices)
    group.create_dataset("indptr", data=csr.indptr)


def read_h5_array(node: h5py.Dataset | h5py.Group, *, as_sparse: bool = False) -> np.ndarray | sparse.csr_matrix:
    if is_sparse_group(node):
        shape = tuple(int(v) for v in node.attrs["shape"])
        matrix = sparse.csr_matrix((np.asarray(node["data"]), np.asarray(node["indices"]), np.asarray(node["indptr"])), shape=shape)
        return matrix if as_sparse else matrix.toarray()
    array = np.asarray(node)
    if as_sparse and array.ndim == 2:
        return sparse.csr_matrix(array)
    return array


def sparse_density(value: np.ndarray | sparse.spmatrix) -> float:
    if sparse.issparse(value):
        total = value.shape[0] * value.shape[1]
        return 0.0 if total == 0 else float(value.nnz / total)
    array = np.asarray(value)
    total = array.size
    return 0.0 if total == 0 else float(np.count_nonzero(array) / total)


def write_maybe_sparse(
    parent: h5py.File | h5py.Group,
    name: str,
    value: np.ndarray | sparse.spmatrix,
    *,
    density_threshold: float = 0.5,
) -> None:
    if sparse.issparse(value):
        write_csr_group(parent, name, value)
        return
    array = np.asarray(value)
    if array.ndim == 2 and sparse_density(array) < density_threshold:
        write_csr_group(parent, name, sparse.csr_matrix(array))
        return
    parent.create_dataset(name, data=array)


def write_sdp_operator(parent: h5py.File | h5py.Group, name: str, A: np.ndarray | sparse.spmatrix, *, n: int) -> None:
    matrix = operator_matrix(A)
    write_csr_group(parent, name, sparse.csr_matrix(matrix), attrs={"sdp_operator": "svec", "n": int(n)})


def _dense_operator_dataset_to_csr(dataset: h5py.Dataset) -> sparse.csr_matrix:
    if dataset.ndim == 2:
        return sparse.csr_matrix(np.asarray(dataset))
    if dataset.ndim != 3:
        raise ValueError(f"A must have rank 2 or 3, got shape {dataset.shape}.")
    m, n, n2 = dataset.shape
    if n != n2:
        raise ValueError(f"Dense A tensor must have square slices, got shape {dataset.shape}.")
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    root2 = float(np.sqrt(2.0))
    for matno in range(m):
        matrix = np.asarray(dataset[matno])
        matrix = 0.5 * (matrix + matrix.T)
        nz_rows, nz_cols = np.nonzero(np.tril(matrix))
        if nz_rows.size == 0:
            continue
        coeffs = matrix[nz_rows, nz_cols].astype(float, copy=True)
        coeffs[nz_rows != nz_cols] *= root2
        rows_parts.append(np.full(nz_rows.size, matno, dtype=np.int64))
        cols_parts.append(np.asarray([svec_index(n, int(i), int(j)) for i, j in zip(nz_rows, nz_cols)], dtype=np.int64))
        data_parts.append(coeffs)
    if not data_parts:
        return sparse.csr_matrix((m, n * (n + 1) // 2), dtype=float)
    return sparse.coo_matrix(
        (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
        shape=(m, n * (n + 1) // 2),
        dtype=float,
    ).tocsr()


def read_sdp_operator(node: h5py.Dataset | h5py.Group) -> np.ndarray | sparse.csr_matrix:
    if is_sparse_group(node):
        return read_h5_array(node, as_sparse=True)
    if isinstance(node, h5py.Dataset) and node.ndim == 2:
        return sparse.csr_matrix(np.asarray(node))
    return np.asarray(node)


def rewrite_solution_sparse(path: str | Path, *, density_threshold: float = 0.5) -> Path:
    path_obj = Path(path)
    tmp_path = path_obj.with_name(path_obj.name + ".tmp")
    sparse_matrix_names = {"C", "S_last", "S_prev"}
    with h5py.File(path_obj, "r") as src, h5py.File(tmp_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        n = int(src["X_last"].shape[0])
        for output_key in src.keys():
            node = src[output_key]
            if output_key == "A":
                if is_sparse_group(node):
                    A = read_h5_array(node, as_sparse=True)
                else:
                    A = _dense_operator_dataset_to_csr(node)
                write_csr_group(dst, output_key, A, attrs={"sdp_operator": "svec", "n": n})
            elif output_key in sparse_matrix_names:
                write_maybe_sparse(dst, output_key, read_h5_array(node), density_threshold=density_threshold)
            elif is_sparse_group(node):
                write_csr_group(dst, output_key, read_h5_array(node, as_sparse=True))
            else:
                dst.create_dataset(output_key, data=np.asarray(node))
    with h5py.File(tmp_path, "r") as handle:
        for required in ("X_last", "y_last", "S_last", "A", "b"):
            if required not in handle:
                raise ValueError(f"Converted file is missing {required}.")
        if not is_sparse_group(handle["A"]):
            raise ValueError("Converted A is not stored as sparse CSR.")
    tmp_path.replace(path_obj)
    return path_obj
