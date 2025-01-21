from quantum_pipeline.utils.logger import get_logger


class Runner:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
