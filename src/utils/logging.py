"""Structured logging configuration using loguru."""

import sys
from typing import Any

from loguru import logger as _root_logger

_CONFIGURED = False

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{message}"
)


def _configure_once() -> None:
    """Set up the root loguru logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _root_logger.remove()  # drop the default stderr handler
    _root_logger.add(
        sys.stderr,
        format=_LOG_FORMAT,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=False,  # disable variable values in tracebacks (safer in prod)
    )
    _CONFIGURED = True


def get_logger(name: str) -> Any:
    """Return a loguru logger bound to the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A loguru ``BoundLogger`` instance with the module name in every record.
    """
    _configure_once()
    return _root_logger.bind(name=name)
