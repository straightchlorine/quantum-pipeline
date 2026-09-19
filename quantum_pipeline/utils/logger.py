import logging

from quantum_pipeline.configs import settings


def get_logger(name: str):
    """Return a logger, attaching the handler only once so records are not emitted twice."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(settings.LOG_LEVEL)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
