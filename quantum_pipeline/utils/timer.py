import time


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        # perf_counter, not time(): these durations become ML features, and a wall-clock
        # step (NTP correction on an ephemeral cloud VM) would corrupt them.
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError('Timer has not finished yet.')
        return self.end_time - self.start_time
