"""Centralized logging setup.

Call get_logger(__name__) from any module instead of configuring
logging locally. logging.basicConfig() only has an effect the first
time it's called per process, so configuring it in every module that
happens to get imported first is fragile — this makes the intent explicit.
"""
import logging

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger(name)