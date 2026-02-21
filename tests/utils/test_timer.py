"""Tests for quantum_pipeline.utils.timer.Timer context manager."""

import time

import pytest

from quantum_pipeline.utils.timer import Timer


class TestTimerInit:
    """Test Timer initial state."""

    def test_initial_times_are_none(self):
        timer = Timer()
        assert timer.start_time is None
        assert timer.end_time is None


class TestTimerContextManager:
    """Test Timer __enter__ and __exit__ behaviour."""

    def test_enter_sets_start_time_and_returns_self(self):
        timer = Timer()
        result = timer.__enter__()
        assert timer.start_time is not None
        assert isinstance(timer.start_time, float)
        assert result is timer

    def test_exit_sets_end_time(self):
        with Timer() as t:
            pass
        assert t.end_time is not None
        assert isinstance(t.end_time, float)

    def test_end_time_after_start_time(self):
        with Timer() as t:
            pass
        assert t.end_time >= t.start_time

    def test_context_manager_with_sleep(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed >= 0.04

    def test_exit_called_on_exception(self):
        timer = Timer()
        with pytest.raises(ValueError, match='test error'):
            with timer:
                raise ValueError('test error')
        assert timer.end_time is not None

    def test_exit_does_not_suppress_exception(self):
        with pytest.raises(RuntimeError):
            with Timer():
                raise RuntimeError('should propagate')


class TestTimerElapsed:
    """Test Timer.elapsed property."""

    def test_elapsed_returns_positive_float(self):
        with Timer() as t:
            time.sleep(0.01)
        assert isinstance(t.elapsed, float)
        assert t.elapsed > 0

    @pytest.mark.parametrize('setup', [
        'no_context',
        'inside_context',
        'start_only',
        'end_only',
    ])
    def test_elapsed_raises_when_incomplete(self, setup):
        timer = Timer()
        if setup == 'no_context':
            pass
        elif setup == 'start_only':
            timer.start_time = time.time()
        elif setup == 'end_only':
            timer.end_time = time.time()
        elif setup == 'inside_context':
            timer.__enter__()

        with pytest.raises(ValueError):
            _ = timer.elapsed

    @pytest.mark.parametrize(
        'sleep_duration',
        [0.02, 0.05, 0.1],
        ids=['20ms', '50ms', '100ms'],
    )
    def test_elapsed_approximates_sleep(self, sleep_duration):
        with Timer() as t:
            time.sleep(sleep_duration)
        assert t.elapsed == pytest.approx(sleep_duration, abs=0.05)

    def test_elapsed_is_stable_after_exit(self):
        with Timer() as t:
            pass
        first = t.elapsed
        time.sleep(0.01)
        second = t.elapsed
        assert first == second


class TestTimerNesting:
    """Test nested Timer usage."""

    def test_nested_timers_independent(self):
        with Timer() as outer:
            time.sleep(0.02)
            with Timer() as inner:
                time.sleep(0.02)
        assert inner.elapsed < outer.elapsed

    def test_multiple_sequential_timers(self):
        with Timer() as t1:
            time.sleep(0.01)
        with Timer() as t2:
            time.sleep(0.01)
        assert t1.elapsed > 0
        assert t2.elapsed > 0
        assert t1.start_time != t2.start_time


class TestTimerReuse:
    """Test reusing a Timer object."""

    def test_reuse_overwrites_times(self):
        timer = Timer()
        with timer:
            time.sleep(0.01)
        first_elapsed = timer.elapsed

        with timer:
            time.sleep(0.02)
        second_elapsed = timer.elapsed

        assert second_elapsed != first_elapsed
