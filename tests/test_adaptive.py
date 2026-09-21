from pathlib import Path

from sib1div.config import load_config
from sib1div.sim.adaptive import (
    bler_plot_lower_limit,
    curve_is_resolved,
    format_duration,
    refinement_snrs,
)
from sib1div.sim.engine import _lmmse_priors


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "plan-001.yaml"
FIXED_CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "plan-002.yaml"
PHASED_CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "plan-003.yaml"


def test_common_drop_curve_stopping_uses_errors_or_below_interest_bound():
    settings = dict(
        min_blocks=200,
        target_errors=100,
        below_interest_bler=1e-2,
        confidence=0.95,
    )
    assert not curve_is_resolved(100, 199, **settings)
    assert curve_is_resolved(100, 200, **settings)
    assert not curve_is_resolved(0, 200, **settings)
    assert curve_is_resolved(0, 400, **settings)


def test_refinement_adds_only_crossing_midpoints():
    estimates = {
        -2.0: {"a": 0.2, "b": 0.08},
        0.0: {"a": 0.08, "b": 0.02},
        2.0: {"a": 0.02, "b": 0.005},
        4.0: {"a": 0.005, "b": 0.001},
    }
    assert refinement_snrs(estimates, ("a", "b"), (0.1, 0.01), 1.0) == [-1.0, 1.0, 3.0]


def test_bler_plot_lower_limit_keeps_one_in_500_visible():
    assert bler_plot_lower_limit([0.078, 0.002]) == 0.001
    assert bler_plot_lower_limit([0.1, 0.05]) == 0.005


def test_progress_duration_formatting():
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661) == "01:01:01"
    assert format_duration(90061) == "1d 01:01:01"


def test_frozen_plan_has_seven_curves_and_expected_prior_counts():
    config = load_config(CONFIG)
    assert len(config.data["receiver"]["estimated_csi_curves"]) == 7
    assert _lmmse_priors("baseline", "plan001") == ("ssb_pdp",)
    assert _lmmse_priors("pol_cycling", "plan001") == ("ssb_pdp",)
    assert len(_lmmse_priors("beam_cycling", "plan001")) == 2
    assert len(_lmmse_priors("beam_cdd", "plan001")) == 3


def test_fixed_coarse_plan_has_isolated_half_db_points_and_bounded_work():
    config = load_config(FIXED_CONFIG)
    monte = config.data["monte_carlo"]
    assert config.data["experiment"]["id"] == "plan-002"
    assert config.data["run"]["output_root"] == "outputs/plan-002"
    assert monte["snr_strategy"] == "fixed_common_drops"
    assert monte["snr_points_db"] == [-11.5, -11, -10.5, -9.5, -8.5]
    assert monte["fixed_drops_per_snr"] == 500
    assert monte["snr_stream_grid_origin_db"] == -12.0
    assert monte["snr_stream_grid_step_db"] == 0.5


def test_plan003_freezes_phase_order_grids_and_pol_prg_variants():
    config = load_config(PHASED_CONFIG)
    monte = config.data["monte_carlo"]
    assert monte["snr_strategy"] == "fixed_phased_common_drops"
    assert monte["phases"]["estimated"]["snr_points_db"] == [-11, -10.5, -10, -9.5, -9, -8.5]
    assert monte["phases"]["ideal"]["snr_points_db"] == [-12, -11.5, -11, -10.5, -10, -9.5, -9]
    assert {curve["id"] for curve in config.data["receiver"]["estimated_csi_curves"]} >= {"P2-SSB", "P6-SSB"}
    assert {curve["id"] for curve in config.data["receiver"]["perfect_csi_curves"]} >= {"PERF-P2", "PERF-P6"}
