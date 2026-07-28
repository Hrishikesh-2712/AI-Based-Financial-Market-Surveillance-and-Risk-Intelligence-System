# -*- coding: utf-8 -*-
"""
Root-level models.py wrapper to support imports such as `import models` or `from models import train_model`.
"""

from model.model import (
    FEATURE_COLS,
    train_model,
    predict,
    save_artifacts,
    load_artifacts,
    run_training,
)

__all__ = [
    "FEATURE_COLS",
    "train_model",
    "predict",
    "save_artifacts",
    "load_artifacts",
    "run_training",
]
