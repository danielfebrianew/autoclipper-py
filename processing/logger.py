import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to stdout (captured by api.py job drain).
    Format: [HH:MM:SS] LEVEL processing.module — message
    Safe to call multiple times — handlers are only added once.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False

    return logger
