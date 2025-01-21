import logging

from quantum_pipeline.configs import settings


def get_logger(name: str):
    """Set up and return a logger without duplicate handlers."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(settings.LOG_LEVEL)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
