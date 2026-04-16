"""Scaffold-based diversity analysis module."""

from navidiv.scaffold import Scaffold_scorer

try:
    from navidiv.scaffold import Scaffold_GNN
except ImportError:
    # FORMED_PROP not installed
    pass

__all__ = ["Scaffold_scorer"]
