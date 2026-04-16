"""NaviDiv - Molecular Diversity Analysis Package.

This package provides tools for molecular diversity analysis including:
- Fragment analysis
- Scaffold analysis
- Similarity scoring
- Clustering
- t-SNE visualization
"""

# Configure RDKit logging to suppress warnings throughout the package
try:
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    # RDKit not available - continue without logging configuration
    pass

# Import and export main modules
from navidiv.simlarity import (
    cluster_similarity_scorer,
    orginal_similarity_scorer,
)

__version__ = "0.1.0"
__author__ = "NaviDiv Team"

__all__ = [
    "cluster_similarity_scorer",
    "orginal_similarity_scorer",
]
