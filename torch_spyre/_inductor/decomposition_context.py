import contextvars
from contextlib import contextmanager
import torch._inductor.decomposition as decomp

_spyre_decomposition_table = contextvars.ContextVar(
    "spyre_decomposition_table", default=decomp.decompositions
)


def get_spyre_decomposition_table():
    """Get the current custom decomposition table from context."""
    return _spyre_decomposition_table.get()


@contextmanager
def set_spyre_decomposition_table(decomp_table):
    """Set a custom decomposition table for the current context."""
    token = _spyre_decomposition_table.set(decomp_table)
    try:
        yield
    finally:
        _spyre_decomposition_table.reset(token)
