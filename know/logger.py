import logging
import sys
from typing import Any

import structlog


# --------------------------------------------------------------------------- #
# Stdlib logging → stdout (kept very small; everything else is handled by
# structlog).  You may change the default level if desired.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(message)s",
)

# --------------------------------------------------------------------------- #
# structlog configuration
# --------------------------------------------------------------------------- #
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
    processors=[
        # enrich
        structlog.processors.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(key="timestamp", fmt="iso"),
        # final render
        structlog.processors.JSONRenderer(),
    ],
)

# --------------------------------------------------------------------------- #
# Public logger instance – import this from everywhere else in the code base.
# --------------------------------------------------------------------------- #
logger: structlog.BoundLogger = structlog.get_logger("know")
__all__: list[str] = ["logger"]
