"""Configuration loading, derived quantities, and cross-field validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a simulation configuration violates a platform invariant."""


EXPECTED_SCHEMES = ("baseline", "pol_cycling", "beam_cycling", "beam_cdd")
TBS_BY_PRBS = {24: 736, 48: 1480, 96: 2976}


def _require(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ConfigError(f"missing required configuration field: {path}")
        value = value[key]
    return value


@dataclass(frozen=True)
class SimulationConfig:
    """Validated configuration plus quantities derived from the NR grid."""

    data: dict[str, Any]
    source: Path

    @property
    def seed(self) -> int:
        return int(_require(self.data, "run.seed"))

    @property
    def tbs_bits(self) -> int:
        return int(_require(self.data, "nr.tbs_bits"))

    @property
    def pdsch_prbs(self) -> int:
        return int(_require(self.data, "nr.pdsch_prbs"))

    @property
    def active_subcarriers(self) -> int:
        return 12 * self.pdsch_prbs

    @property
    def dmrs_res(self) -> int:
        nr = self.data["nr"]
        return (
            self.pdsch_prbs
            * len(nr["dmrs_symbols"])
            * int(nr["dmrs_res_per_prb_per_symbol"])
        )

    @property
    def data_res(self) -> int:
        nr = self.data["nr"]
        return self.active_subcarriers * int(nr["pdsch_length_symbols"]) - self.dmrs_res

    @property
    def coded_bits(self) -> int:
        bits_per_symbol = {"QPSK": 2}[str(self.data["nr"]["modulation"])]
        return self.data_res * bits_per_symbol * int(self.data["nr"]["rank"])

    def resolved(self) -> dict[str, Any]:
        result = deepcopy(self.data)
        result["derived"] = {
            "active_subcarriers": self.active_subcarriers,
            "dmrs_res": self.dmrs_res,
            "data_res": self.data_res,
            "coded_bits": self.coded_bits,
            "txrus_per_polarization": int(self.data["bs_array"]["vertical_txrus_per_pol"])
            * int(self.data["bs_array"]["horizontal_txrus_per_pol"]),
            "total_txru_ports": int(self.data["bs_array"]["polarizations"])
            * int(self.data["bs_array"]["vertical_txrus_per_pol"])
            * int(self.data["bs_array"]["horizontal_txrus_per_pol"]),
            "total_bs_aes": int(self.data["bs_array"]["polarizations"])
            * int(self.data["bs_array"]["vertical_aes"])
            * int(self.data["bs_array"]["horizontal_aes"]),
        }
        return result


def load_config(path: str | Path) -> SimulationConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    config = SimulationConfig(raw, source)
    validate_config(config)
    return config


def validate_config(config: SimulationConfig) -> None:
    data = config.data
    for section in (
        "run", "scenario", "panel", "bs_array", "ue_array", "nr",
        "codebooks", "schemes", "receiver", "monte_carlo",
    ):
        _require(data, section)

    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if tuple(data["run"]["schemes"]) != EXPECTED_SCHEMES:
        errors.append(f"run.schemes must be exactly {list(EXPECTED_SCHEMES)}")
    link_mode = data["run"]["link_mode"]
    supported_link_modes = {
        "normalized_link",  # Legacy Plan 001/002 behavior, retained for reproducibility.
        "coverage_link_budget",
        "fixed_radius_ls_normalized",
        "fixed_radius_full_channel",
        "fixed_cdl_statistics",
    }
    if link_mode not in supported_link_modes:
        errors.append("run.link_mode is unsupported")
    expected_model = "CDL" if link_mode == "fixed_cdl_statistics" else "UMa"
    if data["scenario"]["model"] != expected_model or int(data["scenario"]["ue_per_drop"]) != 1:
        errors.append(f"link mode {link_mode} requires {expected_model} with one UE per drop")
    if link_mode == "fixed_cdl_statistics":
        fixed = data.get("fixed_cdl_statistics")
        if not isinstance(fixed, Mapping):
            errors.append("fixed_cdl_statistics link mode requires a fixed_cdl_statistics section")
        else:
            if str(fixed.get("profile", "")).upper() not in {"A", "B", "C", "D", "E"}:
                errors.append("fixed CDL profile must be one of A, B, C, D, E")
            if float(fixed.get("delay_spread_s", 0.0)) <= 0.0:
                errors.append("fixed CDL delay_spread_s must be positive")
            if int(fixed.get("covariance_realizations", 0)) < 1:
                errors.append("fixed CDL covariance_realizations must be positive")
            if int(fixed.get("statistics_seed", -1)) < 0 or int(fixed.get("realization_seed", -1)) < 0:
                errors.append("fixed CDL statistics_seed and realization_seed must be non-negative")
            if fixed.get("statistics_seed") == fixed.get("realization_seed"):
                errors.append("fixed CDL covariance and BLER streams must use different seeds")
            steps = int(fixed.get("time_steps_per_slot", 1))
            if steps not in {1, int(data["nr"]["slot_symbols"])}:
                errors.append("fixed CDL time_steps_per_slot must be 1 or nr.slot_symbols")
            if steps > 1 and float(fixed.get("time_step_s", 0.0)) <= 0.0:
                errors.append("fixed CDL time_step_s must be positive for time evolution")
            transform = fixed.get("angle_transform", {})
            if not isinstance(transform, Mapping):
                errors.append("fixed CDL angle_transform must be a mapping")
            elif any(float(transform.get(f"{name}_scale", 1.0)) <= 0.0 for name in ("aod", "aoa", "zod", "zoa")):
                errors.append("fixed CDL angle scales must be positive")
            else:
                spread_fields = {
                    "aod": "target_asd_deg", "aoa": "target_asa_deg",
                    "zod": "target_zsd_deg", "zoa": "target_zsa_deg",
                }
                for angle, spread_field in spread_fields.items():
                    if spread_field in transform and float(transform[spread_field]) <= 0.0:
                        errors.append(f"fixed CDL {spread_field} must be positive")
                    if spread_field in transform and f"{angle}_scale" in transform:
                        errors.append(
                            f"fixed CDL {spread_field} and {angle}_scale are mutually exclusive"
                        )
    if link_mode in {"fixed_radius_ls_normalized", "fixed_radius_full_channel"}:
        radius_min = float(data["scenario"]["ue_distance_min_m"])
        radius_max = float(data["scenario"]["ue_distance_max_m"])
        if radius_min != radius_max:
            errors.append("v4 section 2.8 link modes require a fixed UE radius (distance min=max)")
        normalization = data.get("link_normalization")
        if not isinstance(normalization, Mapping):
            errors.append("v4 section 2.8 link modes require link_normalization")
        else:
            gain = normalization.get("reference_large_scale_power_gain_linear")
            gain_db = normalization.get("reference_large_scale_power_gain_db")
            definition = normalization.get("reference_definition")
            if gain is None or float(gain) <= 0.0:
                errors.append("reference_large_scale_power_gain_linear (G0) must be positive")
            if gain_db is None:
                errors.append("reference_large_scale_power_gain_db must be recorded")
            elif gain is not None and float(gain) > 0.0:
                import math
                if not math.isclose(float(gain_db), 10.0 * math.log10(float(gain)), abs_tol=1e-6):
                    errors.append("linear and dB reference large-scale gains must describe the same G0")
            if not isinstance(definition, str) or not definition.strip():
                errors.append("link_normalization.reference_definition must describe the frozen G0")
            if normalization.get("per_scheme_renormalization") is not False:
                errors.append("v4 section 2.8 forbids per-scheme channel renormalization")

    array = data["bs_array"]
    if int(array["vertical_txrus_per_pol"]) * int(array["txru_subarray_vertical_aes"]) != int(array["vertical_aes"]):
        errors.append("vertical TXRU subarrays must cover every vertical AE exactly once")
    if int(array["horizontal_txrus_per_pol"]) * int(array["txru_subarray_horizontal_aes"]) != int(array["horizontal_aes"]):
        errors.append("horizontal TXRU subarrays must cover every horizontal AE exactly once")
    if float(array["horizontal_spacing_lambda"]) != 0.5 or float(array["vertical_spacing_lambda"]) != 0.8:
        errors.append("7 GHz main configuration requires named horizontal=0.5λ and vertical=0.8λ spacing")

    nr = data["nr"]
    prbs = int(nr["pdsch_prbs"])
    if prbs not in TBS_BY_PRBS or int(nr["tbs_bits"]) != TBS_BY_PRBS[prbs]:
        errors.append(f"TBS does not match the independent first-round reference for {prbs} PRB")
    if int(nr["bwp_prbs"]) < prbs:
        errors.append("BWP must contain the full PDSCH allocation")
    if int(nr["pdsch_start_symbol"]) != 2 or int(nr["pdsch_length_symbols"]) != 12:
        errors.append("first-round PDSCH must occupy symbols 2--13")
    if list(nr["dmrs_symbols"]) != [2, 11]:
        errors.append("first-round DMRS symbols must be [2, 11]")
    if int(nr["prg_size_prbs"]) != 2 or prbs % 2:
        errors.append("PDSCH allocation must divide into 2-PRB PRGs")
    if int(nr["rank"]) != 1 or nr["modulation"] != "QPSK" or int(nr["mcs_index"]) != 0:
        errors.append("first-round SIB1 requires rank 1, QPSK, and MCS 0")
    if config.data_res != 132 * prbs:
        errors.append("data RE count must be 132 per PRB")

    books = data["codebooks"]
    if int(books["ssb_horizontal_beams"]) != 8 or int(books["ssb_vertical_beams"]) != 1:
        errors.append("main SSB codebook must be 8H x 1V")
    if int(books["secondary_horizontal_beams"]) != 16 or int(books["secondary_vertical_beams"]) != 1:
        errors.append("main secondary codebook must be 16H x 1V")
    if int(books["secondary_per_selected_ssb"]) != int(data["schemes"]["beam_cdd_beams"]):
        errors.append("first-round Beam CDD must use both secondary beams of the selected SSB")
    if books.get("review_status") == "frozen":
        digest = str(books.get("weights_sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            errors.append("a frozen codebook requires a 64-digit SHA-256 digest")
    if data["schemes"]["normalization"] != "per_active_subcarrier":
        errors.append("all schemes must use per-active-subcarrier unit-norm precoding")
    if int(data["receiver"]["rx_branches"]) != 4 or data["receiver"]["combining"] != "MRC":
        errors.append("the first-round receiver must use four branches and MRC")

    receiver = data["receiver"]
    if "estimated_csi_curves" in receiver:
        expected_curves = {
            ("B-SSB", "baseline", "ssb_pdp", 2),
            ("P-SSB", "pol_cycling", "ssb_pdp", 2),
            ("BC-SSB", "beam_cycling", "ssb_pdp", 2),
            ("BC-BEAM", "beam_cycling", "per_beam_pdp", 2),
            ("CDD-SSB", "beam_cdd", "cdd_ssb_pdp", 48),
            ("CDD-BEAM", "beam_cdd", "cdd_per_beam_pdp", 48),
            ("CDD-IDEAL", "beam_cdd", "ideal_covariance", 48),
        }
        actual_curves = {
            (str(curve["id"]), str(curve["scheme"]), str(curve["covariance"]), int(curve["window_prbs"]))
            for curve in receiver["estimated_csi_curves"]
        }
        experiment_id = str(data.get("experiment", {}).get("id", ""))
        if experiment_id in {"plan-001", "plan-002"} and (
            actual_curves != expected_curves or len(receiver["estimated_csi_curves"]) != 7
        ):
            errors.append("Plan 001 estimated_csi_curves must contain the frozen seven curve combinations")
        curve_ids: set[str] = set()
        supported_covariances = {
            "ssb_pdp", "per_beam_pdp", "cdd_ssb_pdp",
            "cdd_per_beam_pdp", "ideal_covariance",
        }
        for curve in receiver["estimated_csi_curves"]:
            curve_id = str(curve.get("id", ""))
            scheme = str(curve.get("scheme", ""))
            covariance = str(curve.get("covariance", ""))
            window = int(curve.get("window_prbs", 0))
            curve_prg = int(curve.get("prg_size_prbs", nr["prg_size_prbs"]))
            if not curve_id or curve_id in curve_ids:
                errors.append("estimated-CSI curve IDs must be non-empty and unique")
            curve_ids.add(curve_id)
            if scheme not in EXPECTED_SCHEMES or covariance not in supported_covariances:
                errors.append(f"unsupported estimated-CSI curve definition: {curve_id}")
            if window < 1 or prbs % window or curve_prg < 1 or prbs % curve_prg:
                errors.append(f"curve {curve_id} window/PRG must divide the PDSCH allocation")

    perfect_curves = receiver.get("perfect_csi_curves", [])
    perfect_ids: set[str] = set()
    for curve in perfect_curves:
        curve_id = str(curve.get("id", ""))
        scheme = str(curve.get("scheme", ""))
        curve_prg = int(curve.get("prg_size_prbs", nr["prg_size_prbs"]))
        if not curve_id or curve_id in perfect_ids:
            errors.append("Perfect-CSI curve IDs must be non-empty and unique")
        perfect_ids.add(curve_id)
        if scheme not in EXPECTED_SCHEMES or curve_prg < 1 or prbs % curve_prg:
            errors.append(f"unsupported Perfect-CSI curve definition: {curve_id}")

    monte = data["monte_carlo"]
    if monte.get("snr_strategy") == "adaptive_bracket_then_refine":
        search = list(monte.get("search_range_db", []))
        interest = list(monte.get("bler_interest_range", []))
        if len(search) != 2 or float(search[0]) >= float(search[1]):
            errors.append("adaptive SNR search_range_db must be increasing")
        if interest != [0.01, 0.1]:
            errors.append("Plan 001 BLER interest range must be [1e-2, 1e-1]")
        if int(monte.get("min_drops_per_snr", 0)) < 1:
            errors.append("adaptive runner requires a positive minimum drop count")
        if int(monte.get("target_block_errors", 0)) < 1:
            errors.append("adaptive runner requires a positive target error count")
        if int(monte.get("max_drops_per_snr_safety_cap", 0)) < int(monte.get("min_drops_per_snr", 0)):
            errors.append("adaptive runner safety cap must not be below the minimum drop count")
    elif monte.get("snr_strategy") in {"fixed_common_drops", "fixed_phased_common_drops"}:
        if monte.get("snr_strategy") == "fixed_phased_common_drops":
            phases = monte.get("phases", {})
            if set(phases) != {"estimated", "ideal"}:
                errors.append("phased fixed runner requires estimated and ideal phases")
            snrs = [
                value for phase in phases.values()
                for value in phase.get("snr_points_db", [])
            ]
            if phases and not bool(phases.get("estimated", {}).get("estimated_csi", False)):
                errors.append("estimated phase must enable estimated_csi")
            if phases and not bool(phases.get("ideal", {}).get("perfect_csi", False)):
                errors.append("ideal phase must enable perfect_csi")
        else:
            snrs = list(monte.get("snr_points_db", []))
        if not snrs or len(set(map(float, snrs))) != len(snrs):
            if monte.get("snr_strategy") == "fixed_common_drops":
                errors.append("fixed runner requires unique snr_points_db values")
        if int(monte.get("fixed_drops_per_snr", 0)) < 1:
            errors.append("fixed runner requires a positive fixed_drops_per_snr")
        if int(monte.get("progress_interval_drops", 0)) < 1:
            errors.append("fixed runner requires a positive progress interval")
        stream_step = float(monte.get("snr_stream_grid_step_db", 1.0))
        stream_origin = float(monte.get("snr_stream_grid_origin_db", min(map(float, snrs), default=0.0)))
        if stream_step <= 0:
            errors.append("fixed runner SNR stream grid step must be positive")
        elif snrs:
            stream_indices = [round((float(snr) - stream_origin) / stream_step) for snr in snrs]
            if any(
                index < 0 or abs(stream_origin + index * stream_step - float(snr)) > 1e-9
                for snr, index in zip(snrs, stream_indices)
            ):
                errors.append("fixed SNR points must lie on the configured random-stream grid")
            elif monte.get("snr_strategy") == "fixed_common_drops" and len(set(stream_indices)) != len(stream_indices):
                errors.append("fixed SNR points must use unique random-stream indices")
            elif monte.get("snr_strategy") == "fixed_phased_common_drops" and any(
                len({float(value) for value in phase.get("snr_points_db", [])})
                != len(phase.get("snr_points_db", []))
                for phase in phases.values()
            ):
                errors.append("each fixed phase requires unique SNR points")

    if errors:
        raise ConfigError("invalid simulation configuration:\n- " + "\n- ".join(errors))
