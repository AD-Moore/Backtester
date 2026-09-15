"""Centralized logging setup. basicConfig() only takes effect on its
first call per process, so configuring it here (once) rather than per
module avoids depending on import order."""
import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger(name)
