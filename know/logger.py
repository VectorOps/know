import json
import logging
from logging.config import dictConfig
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # Human-readable formatter
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        # Machine-readable / structured (JSON) formatter
        "json": {
            "format": (
                '{"timestamp": "%(asctime)s", '
                '"logger": "%(name)s", '
                '"level": "%(levelname)s", '
                '"message": %(message)s}'
            )
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            # Change to `"json"` if you prefer JSON on stdout by default
            "formatter": "default",
        },
    },
    "loggers": {
        # Library-wide root logger
        "know": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

dictConfig(LOGGING_CONFIG)

# --------------------------------------------------------------------------- #
# Public logger instance
# --------------------------------------------------------------------------- #
logger: logging.Logger = logging.getLogger("know")

class KnowLogger:
    """
    Class-based logging interface for the ``know`` library.
    """
    _logger: logging.Logger = logger

    # ------------------------------------------------------------------ #
    # Standard logging wrappers
    # ------------------------------------------------------------------ #
    @classmethod
    def debug(cls, msg: str, *args, **kwargs) -> None:
        cls._logger.debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg: str, *args, **kwargs) -> None:
        cls._logger.info(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args, **kwargs) -> None:
        cls._logger.warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg: str, *args, **kwargs) -> None:
        cls._logger.error(msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg: str, *args, **kwargs) -> None:
        cls._logger.critical(msg, *args, **kwargs)

    # ------------------------------------------------------------------ #
    # Structured event logging
    # ------------------------------------------------------------------ #
    @classmethod
    def log_event(
        cls,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        level: int = logging.INFO,
        log: logging.Logger | None = None,
    ) -> None:
        payload: Dict[str, Any] = {"event": event_type}
        if data:
            payload["data"] = data
        (log or cls._logger).log(level, json.dumps(payload, default=str))
