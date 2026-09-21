import pytest

from sib1div.analysis import wilson_interval


def test_wilson_zero_error_is_reported_as_upper_bound():
    low, high = wilson_interval(0, 100, 0.95)
    assert low == 0.0
    assert high == pytest.approx(0.0369935, rel=1e-5)

