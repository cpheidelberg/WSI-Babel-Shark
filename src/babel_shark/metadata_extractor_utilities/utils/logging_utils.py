
# -*- coding: utf-8 -*-
"""Lightweight logging setup helper.

Provides a single function `setup_logging` to initialize Python's logging with a
simple formatter. The level can be provided as an argument or via the
`LOG_LEVEL` environment variable (fallback: "INFO").
"""

import logging
import os

def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a concise format and chosen level.

    Resolution order for the level:
      1) `level` argument (if provided and not empty),
      2) `LOG_LEVEL` environment variable,
      3) default "INFO".

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    lvl = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, lvl, logging.INFO),
                        format="%(levelname)s | %(message)s")
