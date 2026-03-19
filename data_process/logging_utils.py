from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "data_process"


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_log_path = log_path.resolve()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stale_handlers: list[logging.Handler] = []
    has_target_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == resolved_log_path:
                has_target_handler = True
                handler.setFormatter(formatter)
            else:
                stale_handlers.append(handler)

    for handler in stale_handlers:
        logger.removeHandler(handler)
        handler.close()

    if not has_target_handler:
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def close_logger() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
