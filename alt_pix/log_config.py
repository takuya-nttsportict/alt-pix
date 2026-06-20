"""Logging configuration for alt-pix.

Levels:
  DEBUG   - per-frame raw values (detection counts, raw coords, timing)
  INFO    - pipeline milestones and per-100-frame summaries
  WARNING - recoverable issues (model not found, no detections for N frames)
  ERROR   - fatal / unrecoverable errors

Usage:
  from alt_pix.log_config import setup_logging
  setup_logging(level="INFO")   # or "DEBUG" for verbose
"""

import logging
import sys
from typing import Literal


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: str | None = None,
) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
    # Silence noisy third-party loggers
    for noisy in ("onnxruntime", "PIL", "urllib3", "easyocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
