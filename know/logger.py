import json
import logging
from logging.config import dictConfig
from typing import Any, Dict

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
    def debug(cls, event_type: str, **data: Any) -> None:
        cls._log_event(event_type, level=logging.DEBUG, **data)

    @classmethod
    def info(cls, event_type: str, **data: Any) -> None:
        cls._log_event(event_type, level=logging.INFO, **data)

    @classmethod
    def warning(cls, event_type: str, **data: Any) -> None:
        cls._log_event(event_type, level=logging.WARNING, **data)

    @classmethod
    def error(cls, event_type: str, **data: Any) -> None:
        cls._log_event(event_type, level=logging.ERROR, **data)

    @classmethod
    def critical(cls, event_type: str, **data: Any) -> None:
        cls._log_event(event_type, level=logging.CRITICAL, **data)

    # ------------------------------------------------------------------ #
    # Structured event logging
    # ------------------------------------------------------------------ #
    @classmethod
    def _log_event(
        cls,
        event_type: str,
        *,
        level: int = logging.INFO,
        log: logging.Logger | None = None,
        **data: Any,
    ) -> None:
        payload: Dict[str, Any] = {"event": event_type}
        if data:
            payload["data"] = data
        (log or cls._logger).log(level, json.dumps(payload, default=str))
