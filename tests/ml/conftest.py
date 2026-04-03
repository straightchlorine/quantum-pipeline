"""Skip all ML tests if optional ML dependencies are not installed."""

import pytest

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
except ImportError:
    pytest.skip(
        'ML tests skipped - slow model training tests run locally only '
        '(install with: pdm install -G ml)',
        allow_module_level=True,
    )
