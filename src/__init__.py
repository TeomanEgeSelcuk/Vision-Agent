"""
HYGO - AI-Generated Image Error Detection

A Python package for detecting visual errors in AI-generated images
using vision language models and grid-based analysis.
"""

__version__ = "0.1.0"
__author__ = "HYGO Team"

# Package-level imports for convenience
from .config import load_config
from .analyzer import create_openrouter_client, analyze_image_grid, compute_severity
from .image_utils import collect_images, load_image, split_into_grid
from .visualizer import create_annotated_image, print_analysis_summary

__all__ = [
    "load_config",
    "create_openrouter_client",
    "analyze_image_grid",
    "compute_severity",
    "collect_images",
    "load_image",
    "split_into_grid",
    "create_annotated_image",
    "print_analysis_summary",
]
