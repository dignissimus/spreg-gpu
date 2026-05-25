import cupy as cp
import cupyx.scipy.sparse as cpsp
from cupy import linalg as cpla

def cuspdot(a, b, array_out=True):
    """
    Matrix multiplication function to deal with sparse and dense objects
    """
    a_is_sparse = cpsp.issparse(a)
    b_is_sparse = cpsp.issparse(b)

    if not a_is_sparse and not b_is_sparse:
        ab = cp.dot(a, b)
    elif a_is_sparse or b_is_sparse:
        ab = a @ b
        if array_out and cpsp.issparse(ab):
            ab = ab.toarray()
    else:
        raise Exception(
            f"Invalid format for 'spdot' argument: {type(a).__name__} and {type(b).__name__}"
        )
    return ab


def cuspfill_diagonal(a, val):
    """
    Fill the diagonal of a sparse or dense matrix
    """
    if cpsp.issparse(a):
        a.setdiag(val)
    else:
        cp.fill_diagonal(a, val)
    return a

def spisfinite(a):
    """
    Determine whether an array has nan or inf values
    """
    return cp.isfinite(a.sum()).item()
