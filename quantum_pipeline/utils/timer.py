import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError('Timer has not finished yet.')
        return self.end_time - self.start_time
