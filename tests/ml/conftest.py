"""Skip all ML tests if optional ML dependencies are not installed."""

import pytest

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
except ImportError:
    pytest.skip(
        'ML dependencies not installed (install with: pdm install -G ml)',
        allow_module_level=True,
    )
