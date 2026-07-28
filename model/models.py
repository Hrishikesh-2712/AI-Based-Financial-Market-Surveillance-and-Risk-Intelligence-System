# -*- coding: utf-8 -*-
"""
models.py alias inside model directory for backward-compatibility and alternative import conventions.
"""

from .model import (
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
