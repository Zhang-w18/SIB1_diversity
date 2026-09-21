from pathlib import Path

from sib1div.analysis.monotone_bler import (
    binomial_isotonic_decreasing,
    log_bler_threshold,
    threshold_bracket,
    threshold_estimate,
)
from sib1div.config import load_config


ROOT = Path(__file__).parents[1]


def test_binomial_isotonic_decreasing_pools_upward_jump() -> None:
    fitted = binomial_isotonic_decreasing([12, 8, 9, 2], [100, 100, 100, 100])
    assert fitted == [0.12, 0.085, 0.085, 0.02]
    assert all(left >= right for left, right in zip(fitted[:-1], fitted[1:]))


def test_log_bler_threshold_interpolates_in_log_domain() -> None:
    value = log_bler_threshold([-8.0, -7.0, -6.0], [0.02, 0.01, 0.005], 0.01)
    assert value == -7.0
    assert threshold_bracket([-8.0, -7.0, -6.0], [0.02, 0.01, 0.005], 0.01) == (-8.0, -7.0)
    assert threshold_estimate([-8.0, -7.0, -6.0], [0.02, 0.01, 0.005], 0.01) == (
        -7.0, -8.0, -7.0,
    )


def test_threshold_estimate_returns_none_when_target_is_not_bracketed() -> None:
    assert threshold_estimate([-8.0, -7.0, -6.0], [0.02, 0.017, 0.013], 0.01) is None


def test_plan004_batches_share_grid_and_expected_drop_ranges() -> None:
    expected = {
        "plan-004-threshold-integer-fill.yaml": (500, 1500),
        "plan-004-threshold-half-grid.yaml": (1000, 1000),
        "plan-004-tail-balance.yaml": (1000, 1000),
        "plan-004-low-bler-refine.yaml": (4000, 2000),
    }
    for filename, (drops, start) in expected.items():
        config = load_config(ROOT / "configs" / "experiments" / filename)
        monte = config.data["monte_carlo"]
        assert config.data["run"]["output_root"] == "outputs/plan-004"
        assert monte["fixed_drops_per_snr"] == drops
        assert monte["drop_index_start"] == start
        assert monte["snr_stream_grid_origin_db"] == -18.0
        assert monte["snr_stream_grid_step_db"] == 0.5
        assert {curve["id"] for curve in config.data["receiver"]["estimated_csi_curves"]} == {
            "B-SSB", "P2-SSB", "P6-SSB", "BC-SSB", "CDD-SSB"
        }
