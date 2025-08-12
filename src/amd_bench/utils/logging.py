"""Logging configuration utilities"""

import logging
import sys
from typing import Optional

import structlog
from structlog.stdlib import BoundLogger


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging"""
    level = logging.DEBUG if verbose else logging.INFO

    # Configure standard library logging
    logging.basicConfig(level=level, format="%(message)", stream=sys.stdout)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if verbose else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> BoundLogger:
    """Get a configured logger instance."""
    return structlog.stdlib.get_logger(name)
