"""Command-line entry points for platform validation and later simulations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .codebook.report import write_codebook_report
from .config import load_config
from .runtime import initialize_run
from .sim import run_adaptive_simulation, run_fixed_curve_simulation, run_simulation
from .analysis.fixed_cdl_diagnostics import write_fixed_cdl_beam_diagnostics
from .analysis.rsrp_cdf import write_rsrp_cdfs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sib1div")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config", help="validate and expand a YAML configuration")
    validate.add_argument("config", type=Path)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--overwrite", action="store_true")
    validate.add_argument("--skip-tensorflow-check", action="store_true")
    codebook = commands.add_parser("generate-codebook", help="generate codebooks and an audit report")
    codebook.add_argument("config", type=Path)
    codebook.add_argument("--output", type=Path, required=True)
    diagnose = commands.add_parser(
        "diagnose-fixed-cdl", help="write long-term SSB and secondary-beam RSRP diagnostics"
    )
    diagnose.add_argument("config", type=Path)
    diagnose.add_argument("--codebook", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    rsrp = commands.add_parser(
        "fixed-cdl-rsrp-cdf", help="write per-Rx effective-port RSRP distributions"
    )
    rsrp.add_argument("config", type=Path)
    rsrp.add_argument("--codebook", type=Path, required=True)
    rsrp.add_argument("--output", type=Path, required=True)
    rsrp.add_argument("--drop-start", type=int, default=2000)
    rsrp.add_argument("--drops", type=int, default=10000)
    simulate = commands.add_parser("simulate", help="run a paired four-scheme link simulation")
    simulate.add_argument("config", type=Path)
    simulate.add_argument("--codebook", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--phase", choices=["estimated", "ideal"], default=None)
    simulate.add_argument("--snr", type=float, nargs="+", default=None)
    simulate.add_argument("--max-drops", type=int, default=None)
    simulate.add_argument(
        "--estimator",
        choices=[
            "perfect", "ls", "lmmse", "lmmse_all", "lmmse_ssb",
            "lmmse_per_beam", "lmmse_ideal",
        ],
        default="lmmse",
    )
    simulate.add_argument("--smoke", action="store_true", help="allow a bounded run before formal preflight")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        output = initialize_run(
            config,
            args.output,
            overwrite=args.overwrite,
            check_tensorflow=not args.skip_tensorflow_check,
        )
        print(f"configuration valid; metadata written to {output}")
        return 0
    if args.command == "generate-codebook":
        report = write_codebook_report(load_config(args.config), args.output)
        print(f"codebooks generated; audit report written to {report}")
        return 0
    if args.command == "diagnose-fixed-cdl":
        result = write_fixed_cdl_beam_diagnostics(
            load_config(args.config), args.codebook, args.output,
        )
        print(f"fixed-CDL diagnostics written to {result}")
        return 0
    if args.command == "fixed-cdl-rsrp-cdf":
        result = write_rsrp_cdfs(
            load_config(args.config), args.codebook, args.output,
            drop_start=args.drop_start, drops=args.drops,
        )
        print(f"fixed-CDL RSRP CDF written to {result}")
        return 0
    if args.command == "simulate":
        config = load_config(args.config)
        if not args.smoke and not bool(config.data.get("preflight", {}).get("formal_run_allowed", False)):
            raise RuntimeError("formal simulation is blocked by the experiment preflight; use --smoke only for bounded debugging")
        strategy = config.data["monte_carlo"].get("snr_strategy")
        if not args.smoke and strategy == "adaptive_bracket_then_refine":
            if args.snr is not None or args.max_drops is not None:
                raise ValueError("formal adaptive runs read SNR/stopping from YAML; --snr and --max-drops are smoke-only")
            result = run_adaptive_simulation(config, args.codebook, args.output)
        elif not args.smoke and strategy in {"fixed_common_drops", "fixed_phased_common_drops"}:
            if args.snr is not None or args.max_drops is not None:
                raise ValueError("formal fixed runs read SNR/drop count from YAML; command-line overrides are smoke-only")
            result = run_fixed_curve_simulation(config, args.codebook, args.output, phase=args.phase)
        else:
            snr = [20.0] if args.snr is None else args.snr
            max_drops = 1 if args.max_drops is None else args.max_drops
            if max_drops < 1 or (args.smoke and max_drops > 10):
                raise ValueError("smoke runs require 1--10 drops")
            result = run_simulation(
                config, args.codebook, args.output, snr,
                max_drops, args.estimator,
            )
        print(f"simulation complete; results written to {result}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
