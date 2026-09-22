"""Paired end-to-end frequency-domain link simulation."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sib1div.analysis import wilson_interval
from sib1div.channel import FixedCDLChannel, UMaChannel, UMaDrop
from sib1div.config import SimulationConfig
from sib1div.nr import SIB1Codec, map_pdsch, qpsk_llr, qpsk_modulate
from sib1div.receiver import (
    beam_domain_path_covariances,
    frequency_covariance,
    independent_cdd_frequency_covariance,
    joint_cdd_frequency_covariance,
    linear_ls_interpolate,
    lmmse_interpolate,
    ls_at_pilots,
    mrc_equalize,
    nmse,
    projected_path_powers,
    windowed_lmmse_interpolate,
)
from sib1div.schemes import SCHEMES, assert_unit_norm, build_precoder, dual_polarized

from .randomness import common_samples


@dataclass(frozen=True)
class LinkResult:
    scheme: str
    estimator: str
    crc_ok: bool
    bit_errors: int
    nmse: float
    selected_ssb: int
    selected_ssb_power_before_scaling: float
    selected_ssb_power_after_scaling: float
    large_scale_power_gain: float
    channel_power_scale: float
    noise_variance: float
    curve_id: str | None = None


def load_codebooks(
    path: str | Path,
    expected_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path)
    if expected_sha256 is not None:
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"frozen codebook SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
    with np.load(source, allow_pickle=False) as archive:
        return archive["ssb_txru_weights"], archive["secondary_txru_weights"]


def select_ssb(
    response: np.ndarray,
    ssb_txru_weights: np.ndarray,
) -> tuple[int, np.ndarray]:
    dual = np.column_stack([
        dual_polarized(ssb_txru_weights[:, index])
        for index in range(ssb_txru_weights.shape[1])
    ])
    effective = np.einsum("rkt,tb->rkb", response, dual, optimize=True)
    powers = np.mean(np.abs(effective) ** 2, axis=(0, 1))
    return int(np.argmax(powers)), powers


def link_mode_scaling(
    config: SimulationConfig,
    large_scale_power_gain: float,
    selected_ssb_power: float,
    snr_db: float,
) -> tuple[float, float]:
    """Return channel power scale and per-branch complex-noise variance.

    The two v4 section 2.8 modes share the frozen reference gain ``G0`` and
    therefore the same nominal-SNR axis. ``selected_ssb_power`` is consulted
    only by the legacy Plan 001/002 normalization mode.
    """
    link_mode = str(config.data["run"]["link_mode"])
    if not np.isfinite(large_scale_power_gain) or large_scale_power_gain <= 0.0:
        raise ValueError("large-scale power gain Gd must be finite and positive")
    if link_mode == "fixed_cdl_statistics":
        reference_power = float(selected_ssb_power)
        if not np.isfinite(reference_power) or reference_power <= 0.0:
            raise ValueError("fixed CDL reference receive power must be finite and positive")
        return 1.0, reference_power * 10.0 ** (-snr_db / 10.0)
    if link_mode in {"fixed_radius_ls_normalized", "fixed_radius_full_channel"}:
        reference_gain = float(
            config.data["link_normalization"]["reference_large_scale_power_gain_linear"]
        )
        channel_power_scale = (
            reference_gain / large_scale_power_gain
            if link_mode == "fixed_radius_ls_normalized" else 1.0
        )
        noise_variance = reference_gain * 10.0 ** (-snr_db / 10.0)
        return channel_power_scale, noise_variance
    if link_mode == "normalized_link":
        reference_power = max(float(selected_ssb_power), 1e-30)
        return 1.0 / reference_power, 10.0 ** (-snr_db / 10.0)
    raise ValueError(f"link mode {link_mode!r} is not implemented by the link simulator")


def _prior_covariances(
    config: SimulationConfig,
    drop: UMaDrop,
    selected_ssb: int,
    ssb_weights: np.ndarray,
    secondary_weights: np.ndarray,
    channel_power_scale: float,
    scheme: str,
    precoder,
    prior: str,
) -> list[np.ndarray]:
    n_sc = config.active_subcarriers
    spacing = float(config.data["nr"]["subcarrier_spacing_hz"])
    frequencies = (np.arange(n_sc) - 0.5 * (n_sc - 1)) * spacing
    # Apply exactly the same deterministic power scaling as the instantaneous
    # channel. This keeps the LMMSE prior and the simulated channel in one
    # physical power domain for every link mode.
    path_covariances = drop.path_spatial_covariances * channel_power_scale
    delays = drop.path_delays_s
    ssb = dual_polarized(ssb_weights[:, selected_ssb])[:, None]

    ssb_powers = projected_path_powers(path_covariances, ssb)[..., 0]
    if prior == "ssb_pdp":
        return [frequency_covariance(delays, powers, frequencies) for powers in ssb_powers]

    children = (2 * selected_ssb, 2 * selected_ssb + 1)
    branches = np.column_stack([
        dual_polarized(np.asarray(secondary_weights)[:, child]) for child in children
    ])
    branch_powers = projected_path_powers(path_covariances, branches)

    if prior == "ideal_covariance":
        if scheme == "beam_cdd":
            settings = config.data["schemes"]
            indices = np.asarray(settings["beam_cdd_delay_grid_indices"], dtype=float)
            denominator = float(settings["beam_cdd_phase_denominator"])
            q = np.arange(n_sc)
            coefficients = np.exp(
                -1j * 2 * np.pi * q[None, :] * indices[:, None] / denominator
            ) / np.sqrt(indices.size)
            if settings["normalization"] == "per_active_subcarrier":
                coefficients /= np.sqrt(precoder.raw_power)[None, :]
            joint = beam_domain_path_covariances(path_covariances, branches)
            return [
                joint_cdd_frequency_covariance(
                    delays, joint[rx], frequencies, coefficients,
                )
                for rx in range(path_covariances.shape[0])
            ]
        # Every non-CDD precoder is constant inside its configured LMMSE
        # window. Construct its matched PDP separately in each window.
        window_sc = 12 * int(config.data["receiver"][f"{scheme}_lmmse_window_prbs"])
        result = []
        for rx in range(path_covariances.shape[0]):
            covariance = np.zeros((n_sc, n_sc), dtype=np.complex128)
            for start in range(0, n_sc, window_sc):
                weight = precoder.weights[:, start:start + 1]
                powers = projected_path_powers(
                    path_covariances[rx:rx + 1], weight,
                )[0, :, 0]
                covariance[start:start + window_sc, start:start + window_sc] = frequency_covariance(
                    delays, powers, frequencies[start:start + window_sc],
                )
            result.append(covariance)
        return result

    if prior == "per_beam_pdp":
        # Beam cycling is estimated independently inside every PRG.  Fill only
        # those diagonal blocks; cross-PRG entries are deliberately unused.
        prg_sc = int(config.data["nr"]["prg_size_prbs"]) * 12
        result = []
        for rx in range(path_covariances.shape[0]):
            covariance = np.zeros((n_sc, n_sc), dtype=np.complex128)
            for start in range(0, n_sc, prg_sc):
                beam = ((start // prg_sc) // int(
                    config.data["schemes"].get("beam_cycling_hold_prgs", 1)
                )) % branches.shape[1]
                local = frequency_covariance(
                    delays, branch_powers[rx, :, beam], frequencies[start:start + prg_sc],
                )
                covariance[start:start + prg_sc, start:start + prg_sc] = local
            result.append(covariance)
        return result

    if prior in {"cdd_ssb_pdp", "cdd_per_beam_pdp"}:
        artificial = np.asarray(
            config.data["schemes"]["beam_cdd_delay_seconds"], dtype=float,
        )
        include_normalization = bool(config.data["receiver"].get(
            "beam_cdd_include_known_per_subcarrier_normalization", True,
        ))
        raw_power = precoder.raw_power if include_normalization else None
        result = []
        for rx in range(path_covariances.shape[0]):
            if prior == "cdd_ssb_pdp":
                powers = np.tile(ssb_powers[rx][None, :], (artificial.size, 1))
            else:
                powers = branch_powers[rx].T
            result.append(independent_cdd_frequency_covariance(
                delays, powers, frequencies, artificial, raw_power,
            ))
        return result
    raise ValueError(f"unsupported covariance prior {prior!r} for scheme {scheme!r}")


def _lmmse_priors(scheme: str, estimator: str) -> tuple[str, ...]:
    primary = "cdd_ssb_pdp" if scheme == "beam_cdd" else "ssb_pdp"
    if estimator == "lmmse":
        return (primary,)
    if estimator == "lmmse_all":
        if scheme == "beam_cdd":
            return ("cdd_ssb_pdp", "cdd_per_beam_pdp", "ideal_covariance")
        if scheme == "beam_cycling":
            return ("ssb_pdp", "per_beam_pdp", "ideal_covariance")
        return ("ssb_pdp", "ideal_covariance")
    if estimator == "plan001":
        if scheme == "beam_cdd":
            return ("cdd_ssb_pdp", "cdd_per_beam_pdp", "ideal_covariance")
        if scheme == "beam_cycling":
            return ("ssb_pdp", "per_beam_pdp")
        return ("ssb_pdp",)
    requested = {
        "lmmse_ssb": primary,
        "lmmse_per_beam": (
            "cdd_per_beam_pdp" if scheme == "beam_cdd" else
            "per_beam_pdp" if scheme == "beam_cycling" else primary
        ),
        "lmmse_ideal": "ideal_covariance",
    }
    return (requested[estimator],)


def simulate_one_drop(
    config: SimulationConfig,
    drop: UMaDrop,
    ssb_weights: np.ndarray,
    secondary_weights: np.ndarray,
    snr_db: float,
    snr_index: int,
    estimator: str = "lmmse",
    curve_specs: list[dict] | None = None,
    perfect_specs: list[dict] | None = None,
) -> list[LinkResult]:
    if estimator not in {
        "perfect", "ls", "lmmse", "lmmse_all", "lmmse_ssb",
        "lmmse_per_beam", "lmmse_ideal", "plan001",
    }:
        raise ValueError("unsupported estimator selection")
    codec = SIB1Codec(config)
    samples = common_samples(
        config, snr_index, drop.drop_index,
        (int(config.data["nr"]["slot_symbols"]), config.active_subcarriers, 4),
    )
    coded = codec.encode(samples.transport_block)
    grid = map_pdsch(config, qpsk_modulate(coded))
    spacing = float(config.data["nr"]["subcarrier_spacing_hz"])
    response = drop.frequency_response(config.active_subcarriers, spacing)
    selection_response = response if response.ndim == 3 else response.reshape(
        -1, response.shape[-2], response.shape[-1]
    )
    # P_b is used only for SSB selection and diagnostics in the v4 section 2.8
    # modes. Because their large-scale scaling is common to all beams, the
    # selected index is identical before and after that scaling.
    instantaneous_selected, ssb_powers = select_ssb(selection_response, ssb_weights)
    selected = instantaneous_selected if drop.fixed_selected_ssb is None else int(drop.fixed_selected_ssb)
    selected_power_before = float(ssb_powers[selected])
    large_scale_power_gain = float(drop.large_scale_power_gain)
    channel_power_scale, noise_variance = link_mode_scaling(
        config, large_scale_power_gain,
        (selected_power_before if drop.reference_receive_power is None else drop.reference_receive_power),
        snr_db,
    )
    response = response * np.sqrt(channel_power_scale)
    scaled_selection = response if response.ndim == 3 else response.reshape(
        -1, response.shape[-2], response.shape[-1]
    )
    selected_after_instantaneous, scaled_ssb_powers = select_ssb(scaled_selection, ssb_weights)
    selected_after = selected_after_instantaneous if drop.fixed_selected_ssb is None else selected
    if selected_after != selected:
        raise RuntimeError("common scalar link-mode scaling changed selected SSB")
    selected_power_after = float(scaled_ssb_powers[selected])
    noise = np.sqrt(noise_variance) * samples.unit_variance_noise
    data_l, data_k = np.nonzero(grid.data_mask)
    results: list[LinkResult] = []

    configured_curves = curve_specs is not None or perfect_specs is not None
    if configured_curves:
        estimated = [] if curve_specs is None else curve_specs
        perfect = [] if perfect_specs is None else perfect_specs
        profiles: dict[tuple[str, int], dict[str, list[dict]]] = {}
        default_prg = int(config.data["nr"]["prg_size_prbs"])
        for spec in estimated:
            profile = (str(spec["scheme"]), int(spec.get("prg_size_prbs", default_prg)))
            profiles.setdefault(profile, {"estimated": [], "perfect": []})["estimated"].append(spec)
        for spec in perfect:
            profile = (str(spec["scheme"]), int(spec.get("prg_size_prbs", default_prg)))
            profiles.setdefault(profile, {"estimated": [], "perfect": []})["perfect"].append(spec)
        work = [(scheme, prg, specs) for (scheme, prg), specs in profiles.items()]
    else:
        work = [(scheme, int(config.data["nr"]["prg_size_prbs"]), None) for scheme in SCHEMES]

    for scheme, prg_size_prbs, specs in work:
        precoder = build_precoder(
            config, scheme, selected, ssb_weights, secondary_weights,
            prg_size_prbs=prg_size_prbs,
        )
        assert_unit_norm(precoder)
        if response.ndim == 3:
            true_h = np.einsum("rkt,tk->rk", response, precoder.weights, optimize=True)
            true_h_grid = np.broadcast_to(true_h.T[None, :, :], noise.shape)
        else:
            sampled = response
            if sampled.shape[0] != grid.symbols.shape[0]:
                indices = np.rint(np.linspace(0, sampled.shape[0] - 1, grid.symbols.shape[0])).astype(int)
                sampled = sampled[indices]
            true_h_grid = np.einsum("nrkt,tk->nkr", sampled, precoder.weights, optimize=True)
            true_h = np.mean(true_h_grid, axis=0).T
        received = grid.symbols[:, :, None] * true_h_grid + noise
        pilot_estimates, pilot_k = ls_at_pilots(received, grid.symbols, grid.dmrs_mask)
        if configured_curves:
            estimates_to_run = [
                (str(spec["id"]), "perfect", true_h_grid)
                for spec in specs["perfect"]
            ]
            for spec in specs["estimated"]:
                prior = str(spec["covariance"])
                covariances = _prior_covariances(
                    config, drop, selected, ssb_weights, secondary_weights,
                    channel_power_scale, scheme, precoder, prior,
                )
                window_sc = 12 * int(spec["window_prbs"])
                estimated_h = np.vstack([
                    windowed_lmmse_interpolate(
                        pilot_estimates[rx:rx + 1], pilot_k, covariances[rx],
                        noise_variance, window_sc,
                        pilot_repetitions=len(grid.dmrs_symbols),
                    )[0]
                    for rx in range(true_h.shape[0])
                ])
                estimates_to_run.append((str(spec["id"]), f"lmmse_{prior}", estimated_h))
        elif estimator in {"perfect", "ls"}:
            estimates_to_run = [(estimator, true_h_grid if estimator == "perfect" else
                linear_ls_interpolate(pilot_estimates, pilot_k, config.active_subcarriers))]
        else:
            estimates_to_run = [("perfect", true_h_grid)] if estimator == "plan001" else []
            for prior in _lmmse_priors(scheme, estimator):
                covariances = _prior_covariances(
                    config, drop, selected, ssb_weights, secondary_weights,
                    channel_power_scale, scheme, precoder, prior,
                )
                window_prbs = int(config.data["receiver"][f"{scheme}_lmmse_window_prbs"])
                window_sc = 12 * window_prbs
                estimated_h = np.vstack([
                    windowed_lmmse_interpolate(
                        pilot_estimates[rx:rx + 1], pilot_k, covariances[rx],
                        noise_variance, window_sc,
                        pilot_repetitions=len(grid.dmrs_symbols),
                    )[0]
                    for rx in range(true_h.shape[0])
                ])
                estimates_to_run.append((f"lmmse_{prior}", estimated_h))
        observations = received[data_l, data_k, :]
        for estimate in estimates_to_run:
            if configured_curves:
                curve_id, estimator_name, estimated_h = estimate
            else:
                estimator_name, estimated_h = estimate
                curve_id = None
            if estimated_h.ndim == 3:
                estimates = estimated_h[data_l, data_k, :]
            else:
                estimates = estimated_h[:, data_k].T
            equalized, effective_variance = mrc_equalize(observations, estimates, noise_variance)
            decoded = codec.decode(qpsk_llr(equalized, effective_variance))
            bit_errors = int(np.count_nonzero(decoded.payload != samples.transport_block))
            results.append(LinkResult(
                scheme, estimator_name, decoded.crc_ok, bit_errors,
                nmse(
                    estimated_h if estimated_h.ndim == 3 else np.broadcast_to(estimated_h.T[None, :, :], true_h_grid.shape),
                    true_h_grid,
                ), selected, selected_power_before,
                selected_power_after, large_scale_power_gain,
                channel_power_scale, noise_variance,
                curve_id,
            ))
    return results


def run_simulation(
    config: SimulationConfig,
    codebook_path: str | Path,
    output_dir: str | Path,
    snr_values: list[float],
    max_drops: int,
    estimator: str,
) -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_hash = config.data["codebooks"].get("weights_sha256")
    ssb, secondary = load_codebooks(codebook_path, expected_hash)
    channel = (
        FixedCDLChannel(config, ssb)
        if config.data["run"]["link_mode"] == "fixed_cdl_statistics"
        else UMaChannel(config)
    )
    rows = []
    cdd_rows = []
    for snr_index, snr_db in enumerate(snr_values):
        for drop_index in range(max_drops):
            drop = channel.generate(drop_index)
            for result in simulate_one_drop(
                config, drop, ssb, secondary, snr_db, snr_index, estimator
            ):
                rows.append({
                    "snr_db": snr_db,
                    "drop_index": drop_index,
                    "radius_m": drop.radius_m,
                    "azimuth_deg": drop.azimuth_deg,
                    "los": drop.los,
                    **result.__dict__,
                })
            selected = int(rows[-1]["selected_ssb"])
            cdd = build_precoder(config, "beam_cdd", selected, ssb, secondary)
            norms = np.sum(np.abs(cdd.weights) ** 2, axis=0)
            for k in range(config.active_subcarriers):
                cdd_rows.append({
                    "snr_db": snr_db,
                    "drop_index": drop_index,
                    "selected_ssb": selected,
                    "subcarrier": k,
                    "raw_power": cdd.raw_power[k],
                    "normalized_power": norms[k],
                })
    path = output / "link_results.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "cdd_precoder_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cdd_rows[0]))
        writer.writeheader()
        writer.writerows(cdd_rows)
    summary = []
    confidence = float(config.data["monte_carlo"]["confidence_level"])
    for snr_db in snr_values:
        estimators = sorted({
            str(row["estimator"]) for row in rows if row["snr_db"] == snr_db
        })
        for scheme in SCHEMES:
            for estimator_name in estimators:
                selected_rows = [
                    row for row in rows
                    if row["snr_db"] == snr_db and row["scheme"] == scheme
                    and row["estimator"] == estimator_name
                ]
                if not selected_rows:
                    continue
                errors = sum(not bool(row["crc_ok"]) for row in selected_rows)
                total = len(selected_rows)
                low, high = wilson_interval(errors, total, confidence)
                summary.append({
                    "snr_db": snr_db,
                    "scheme": scheme,
                    "estimator": estimator_name,
                    "block_errors": errors,
                    "blocks": total,
                    "bler": errors / total,
                    "wilson_low": low,
                    "wilson_high": high,
                    "mean_nmse": float(np.mean([row["nmse"] for row in selected_rows])),
                })
    with (output / "bler_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return path
