import logging


def get_logger(name: str):
    """Set up and return a logger without duplicate handlers."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
