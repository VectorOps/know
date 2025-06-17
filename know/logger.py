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
            "level": "INFO",
            "propagate": False,
        },
    },
}

dictConfig(LOGGING_CONFIG)

# --------------------------------------------------------------------------- #
# Public logger instance
# --------------------------------------------------------------------------- #
logger: logging.Logger = logging.getLogger("know")

# --------------------------------------------------------------------------- #
# Structured logging helper
# --------------------------------------------------------------------------- #
def log_event(
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    level: int = logging.INFO,
    log: logging.Logger | None = None,
) -> None:
    """
    Emit a structured log entry.

    Args:
        event_type:  Short identifier of the event, e.g. ``"FILE_PARSED"``.
        data:        Optional JSON-serialisable payload with event details.
        level:       Standard ``logging`` level (default: ``logging.INFO``).
        log:         Custom logger to use; defaults to the shared ``know`` logger.
    """
    payload: Dict[str, Any] = {"event": event_type}
    if data:
        payload["data"] = data

    (log or logger).log(level, json.dumps(payload, default=str))
