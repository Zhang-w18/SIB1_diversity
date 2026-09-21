#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed 7.5-degree AOD BLER sweep for all 8 SIB1 schemes.

This script imports the target module and replicates its BLER sweep logic, but
with a FIXED angle_gap_deg=7.5 instead of the per-trial random AOD rotation.

Key differences from the original script's __main__:
  - angle_gap_deg is fixed at 7.5 for every trial (not derived from trial seed)
  - n_trials_per_target = 1000 (configurable via --trials)
  - Output files are suffixed with _fixed7p5deg to avoid overwriting random-angle results

Everything else (scheme configs, SNR points, channel model, target mapping, etc.)
is identical to the original script.
"""
import argparse
import csv
import os
import sys
import time
import concurrent.futures
from dataclasses import asdict
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# Import the target script as a module
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "sib1_2v4h_coarse_2v8h_fine_comb_re_dmrs_with_prg_random_angle.py")
spec = importlib.util.spec_from_file_location("sib1_target", TARGET)
mod = importlib.util.module_from_spec(spec)
sys.modules["sib1_target"] = mod
spec.loader.exec_module(mod)

# Re-use the same names from the module
SchemeConfig = mod.SchemeConfig
make_target_mappings_8ssb_16fine = mod.make_target_mappings_8ssb_16fine
resolve_target_mapping = mod.resolve_target_mapping
one_shot_sib_min_combined = mod.one_shot_sib_min_combined
deterministic_angle_gap_from_trial_seed = mod.deterministic_angle_gap_from_trial_seed


def run_single_trial_fixed_angle(args):
    """Like run_single_trial_global, but with a FIXED angle_gap_deg."""
    snr_db, target_mapping, trial_seed, fixed_angle_deg, static_cfg, scheme_dict = args
    scheme_cfg = SchemeConfig(**scheme_dict)
    qv, qh_a, qh_b, qh_center = resolve_target_mapping(target_mapping, Mh=mod.HORIZONTAL_ARRAY_SIZE)

    ok, _ = one_shot_sib_min_combined(
        snr_db=snr_db,
        ncellid=static_cfg["ncellid"],
        scheme_cfg=scheme_cfg,
        qv=qv,
        qh_a=qh_a,
        qh_b=qh_b,
        qh_center=qh_center,
        fs=static_cfg["fs"],
        scs_khz=static_cfg["scs_khz"],
        nrb_sib=static_cfg["nrb_sib"],
        nsym_sib=static_cfg["nsym_sib"],
        rng_seed=trial_seed,
        N_comb=static_cfg["N_comb"],
        payload=static_cfg["payload"],
        dmrs_step=2,
        dmrs_loc=(0, 6, 9),
        phase_table=tuple(static_cfg["phase_table"]),
        chan_seed_base=trial_seed,
        channel_model=static_cfg["channel_model"],
        cdl_type=static_cfg["cdl_type"],
        tdl_profile=static_cfg["tdl_profile"],
        tdl_directional=static_cfg["tdl_directional"],
        tdl_normalize=static_cfg["tdl_normalize"],
        angle_gap_deg=fixed_angle_deg,  # <-- FIXED, not from seed
    )
    return ok


def simulate_bler_fixed_angle(
    snr_db_list,
    scheme_cfg: SchemeConfig,
    target_mappings,
    n_trials_per_target=1000,
    ncellid=208,
    fs=61.44e6,
    scs_khz=30,
    seed=2029,
    nrb_sib=48,
    nsym_sib=12,
    N_comb=1,
    payload=1200,
    max_workers=4,
    target_errors=200,
    snr_early_stop_th=None,
    channel_model="CDL",
    cdl_type="CDLC",
    tdl_profile="TDL-C",
    tdl_directional=False,
    tdl_normalize=False,
    phase_table=(0, np.pi / 2),
    fixed_angle_deg=7.5,
):
    bler_avg_list = []
    rng = np.random.RandomState(seed)

    static_cfg = {
        "ncellid": ncellid,
        "fs": fs,
        "scs_khz": scs_khz,
        "nrb_sib": nrb_sib,
        "nsym_sib": nsym_sib,
        "N_comb": N_comb,
        "payload": payload,
        "channel_model": channel_model,
        "cdl_type": cdl_type,
        "tdl_profile": tdl_profile,
        "tdl_directional": tdl_directional,
        "tdl_normalize": tdl_normalize,
        "phase_table": list(phase_table),
    }

    scheme_dict = asdict(scheme_cfg)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for snr_db in snr_db_list:
            target_blers = []

            for mapping in target_mappings:
                err = 0
                run = 0
                tasks = []
                for _ in range(n_trials_per_target):
                    trial_seed = int(rng.randint(1 << 31))
                    # Fixed angle: same 7.5 deg for every trial
                    tasks.append((snr_db, mapping, trial_seed, fixed_angle_deg, static_cfg, scheme_dict))
                futures = [executor.submit(run_single_trial_fixed_angle, t) for t in tasks]

                desc = f"{scheme_cfg.label} | SNR={snr_db:>5.1f} | {mapping['name']} | ang={fixed_angle_deg}deg"
                with tqdm(total=n_trials_per_target, desc=desc, ncols=120, leave=False) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        ok = future.result()
                        run += 1
                        if not ok:
                            err += 1
                        pbar.update(1)

                        if err >= target_errors:
                            pbar.set_postfix_str(f"Early Stop: {err} errs")
                            for f in futures:
                                f.cancel()
                            break

                target_bler = err / max(run, 1)
                target_blers.append(target_bler)

            avg_bler = float(np.mean(target_blers))
            bler_avg_list.append(avg_bler)
            print(
                f"[{scheme_cfg.label}] SNR={snr_db:>5.1f} dB | "
                f"Avg BLER={avg_bler:.4e} | per-target={['%.4f' % b for b in target_blers]} | "
                f"ang={fixed_angle_deg}deg"
            )

            if snr_early_stop_th is not None and avg_bler <= snr_early_stop_th:
                print(f"  -> Avg BLER <= {snr_early_stop_th}. Padding remaining SNRs.")
                remaining = len(snr_db_list) - len(bler_avg_list)
                bler_avg_list.extend([0.0] * remaining)
                break

    return np.array(bler_avg_list)


def main():
    parser = argparse.ArgumentParser(description="Fixed-angle AOD SIB1 BLER sweep")
    parser.add_argument("--trials", type=int, default=1000,
                        help="trials per (scheme, SNR, target). Default 1000")
    parser.add_argument("--angle", type=float, default=7.5,
                        help="fixed AOD angle in degrees. Default 7.5")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel workers. Default 4")
    parser.add_argument("--schemes", type=int, default=0,
                        help="run only first N schemes (0=all). Default 0")
    parser.add_argument("--quick", action="store_true",
                        help="quick mode: 2 schemes, 3 SNR points, 50 trials (for timing)")
    parser.add_argument("--ssb-compare", type=int, nargs="+",
                        help="compare center_ssb_idx values, runs baseline only")
    args = parser.parse_args()

    # ---- Configuration (mirrors the original script's __main__) ----
    nrb_sib = 48
    fs = 61.44e6
    channel_model = "CDL"
    cdl_type = "CDLC"
    snr_points = [-17, -15, -13, -11, -9, -7, -5, -3, -1]
    target_mappings = make_target_mappings_8ssb_16fine(run_all_ssb=False, center_ssb_idx=3)

    scheme_configs = [
        SchemeConfig(
            label="Baseline: coarse 4H beam, 24RB CE, legacy DMRS",
            family="array_based", scheme="array_coarse_baseline",
            prg_size_rb=24, prg_precoding_mode="traditional",
            ce_mode="fullband", dmrs_pattern="legacy", td_win_len_ratio=0.12,
        ),
        SchemeConfig(
            label="Baseline: coarse 4H beam, 8RB CE, legacy DMRS",
            family="array_based", scheme="array_coarse_baseline",
            prg_size_rb=8, prg_precoding_mode="traditional",
            ce_mode="fullband", dmrs_pattern="legacy", td_win_len_ratio=0.12,
        ),
        SchemeConfig(
            label="Comb-DMRS-array: RE-level qh_a/qh_b fine-beam states",
            family="array_based", scheme="array_comb_re_dmrs",
            prg_size_rb=48, prg_precoding_mode="traditional",
            ce_mode="fullband", dmrs_pattern="comb2_re", td_win_len_ratio=0.24,
        ),
        SchemeConfig(
            label="Comb-DMRS-precoding: RE-level dual-pol [1,1]/[1,j] states",
            family="precoding_based", scheme="precoding_comb_re_dmrs",
            prg_size_rb=48, prg_precoding_mode="cyclic",
            ce_mode="fullband", precoding_basis="polarization",
            dmrs_pattern="comb2_re", td_win_len_ratio=0.24,
        ),
        SchemeConfig(
            label="PRG-array: 8H fine pair qh_a/qh_b, PRG=24RB, legacy DMRS",
            family="array_based", scheme="array_fdm_fine",
            prg_size_rb=24, prg_precoding_mode="traditional",
            ce_mode="fullband", dmrs_pattern="legacy", td_win_len_ratio=0.12,
        ),
        SchemeConfig(
            label="PRG-array: 8H fine pair qh_a/qh_b, PRG=8RB, legacy DMRS",
            family="array_based", scheme="array_fdm_fine",
            prg_size_rb=8, prg_precoding_mode="traditional",
            ce_mode="fullband", dmrs_pattern="legacy", td_win_len_ratio=0.12,
        ),
        SchemeConfig(
            label="PRG-precoding: centered spatial beam, dual-pol [1,1]/[1,j], PRG=24RB",
            family="precoding_based", scheme="precoding_cyclic",
            prg_size_rb=24, prg_precoding_mode="cyclic",
            ce_mode="fullband", precoding_basis="polarization",
            dmrs_pattern="legacy", td_win_len_ratio=0.12,
        ),
        SchemeConfig(
            label="PRG-precoding: centered spatial beam, dual-pol [1,1]/[1,j], PRG=8RB",
            family="precoding_based", scheme="precoding_cyclic",
            prg_size_rb=8, prg_precoding_mode="cyclic",
            ce_mode="fullband", precoding_basis="polarization",
            dmrs_pattern="legacy", td_win_len_ratio=0.12,
        ),
    ]

    if args.quick:
        snr_points = [-11, -9, -7]
        scheme_configs = scheme_configs[:2]
        n_trials = 50
        print("[QUICK MODE] 2 schemes, 3 SNR points, 50 trials — for timing estimation")
    else:
        n_trials = args.trials
        if args.schemes > 0:
            scheme_configs = scheme_configs[:args.schemes]

    # ---- SSB comparison mode ----
    if args.ssb_compare is not None:
        print(f"[SSB COMPARE] Comparing center_ssb_idx = {args.ssb_compare}")
        print(f"[SSB COMPARE] Baseline only, {n_trials} trials, fixed AOD = {args.angle} deg")
        baseline_cfgs = scheme_configs[:2]  # only baseline 24RB and 8RB
        all_results = {}
        t_total = time.time()
        for ssb_idx in args.ssb_compare:
            tm = make_target_mappings_8ssb_16fine(run_all_ssb=False, center_ssb_idx=ssb_idx)
            print(f"\n{'='*60}")
            print(f"Running center_ssb_idx={ssb_idx} | mapping={tm[0]['name']}")
            print(f"{'='*60}")
            ssb_results = {}
            for cfg in baseline_cfgs:
                bler_curve = simulate_bler_fixed_angle(
                    snr_db_list=snr_points, scheme_cfg=cfg,
                    target_mappings=tm, n_trials_per_target=n_trials,
                    ncellid=208, fs=fs, scs_khz=30, seed=2029,
                    nrb_sib=nrb_sib, nsym_sib=12, N_comb=1, payload=1200,
                    max_workers=args.workers, target_errors=200,
                    snr_early_stop_th=0.03, channel_model=channel_model,
                    cdl_type=cdl_type, tdl_profile="TDL-C",
                    tdl_directional=False, tdl_normalize=False,
                    fixed_angle_deg=args.angle,
                )
                ssb_results[cfg.label] = bler_curve
                print(f"  -> {cfg.label} done")
            all_results[ssb_idx] = ssb_results

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"SSB COMPARE RESULT (fixed AOD={args.angle} deg, {n_trials} trials)")
        print(f"{'='*70}")
        for cfg in baseline_cfgs:
            print(f"\n  --- {cfg.label} ---")
        for cfg in baseline_cfgs:
            print(f"  {'SNR(dB)':>8}", end='')
            for ssb_idx in args.ssb_compare:
                print(f"  {'SSB'+str(ssb_idx):>8}", end='')
            print(f"  {'差距(3-1)':>10}")
            print('  ' + '-'*48)
            for i, snr in enumerate(snr_points):
                row = f"  {snr:>8}"
                vals = []
                for ssb_idx in args.ssb_compare:
                    b = all_results[ssb_idx][cfg.label][i] if i < len(all_results[ssb_idx][cfg.label]) else 0.0
                    vals.append(b)
                    row += f"  {b:>8.4f}"
                gap = vals[0] - vals[1] if len(vals) >= 2 else 0
                row += f"  {gap:>+10.4f}"
                if gap > 0.1: row += '  << SSB1 好很多'
                elif gap > 0.05: row += '  << SSB1 略好'
                elif gap < -0.05: row += '  << SSB3 略好'
                print(row)

        dt_total = time.time() - t_total
        print(f"\n[Total elapsed] {dt_total:.1f}s ({dt_total/60:.1f} min)")
        sys.exit(0)

    print(f"[Config] Fixed AOD = {args.angle} deg")
    print(f"[Config] Trials per (scheme, SNR, target) = {n_trials}")
    print(f"[Config] Schemes = {len(scheme_configs)}")
    print(f"[Config] SNR points = {snr_points}")
    print(f"[Config] Workers = {args.workers}")
    print(f"[Config] Target mapping = {target_mappings[0]['name']}")
    print(f"[Config] Channel = {channel_model}/{cdl_type}")
    print()

    curves: Dict[str, np.ndarray] = {}
    out_csv = f"sib1_fixed{args.angle}deg_bler.csv"
    out_png = f"sib1_fixed{args.angle}deg_bler.png"

    t_total = time.time()
    for cfg in scheme_configs:
        t_scheme = time.time()
        bler_curve = simulate_bler_fixed_angle(
            snr_db_list=snr_points,
            scheme_cfg=cfg,
            target_mappings=target_mappings,
            n_trials_per_target=n_trials,
            ncellid=208,
            fs=fs,
            scs_khz=30,
            seed=2029,
            nrb_sib=nrb_sib,
            nsym_sib=12,
            N_comb=1,
            payload=1200,
            max_workers=args.workers,
            target_errors=200,
            snr_early_stop_th=0.03,
            channel_model=channel_model,
            cdl_type=cdl_type,
            tdl_profile="TDL-C",
            tdl_directional=False,
            tdl_normalize=False,
            fixed_angle_deg=args.angle,
        )
        curves[cfg.label] = bler_curve
        dt_scheme = time.time() - t_scheme
        print(f"  -> {cfg.label} done in {dt_scheme:.1f}s")

    dt_total = time.time() - t_total
    print(f"\n[Total elapsed] {dt_total:.1f}s ({dt_total/60:.1f} min)")

    # ---- Save CSV ----
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Config", "Family", "Scheme", "PRG_Size_RB", "Precoding_Mode",
            "CE_Mode", "Precoding_Basis", "DMRS_Pattern",
            "N_Coarse_SSB", "N_Fine_Beams",
            "Channel_Model", "CDL_Type",
            "Fixed_AOD_Deg", "SNR(dB)", "BLER",
        ])
        for cfg in scheme_configs:
            for snr, bler_val in zip(snr_points, curves[cfg.label]):
                writer.writerow([
                    cfg.label, cfg.family, cfg.scheme,
                    cfg.prg_size_rb, cfg.prg_precoding_mode,
                    cfg.ce_mode, cfg.precoding_basis, cfg.dmrs_pattern,
                    mod.N_COARSE_SSB, mod.N_FINE_BEAMS,
                    channel_model, cdl_type,
                    args.angle, snr, bler_val,
                ])

    # ---- Save PNG ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    for label, bler_curve in curves.items():
        plt.semilogy(snr_points, np.maximum(bler_curve, 1e-4), marker='o', label=label)
    plt.grid(True, which="both")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average BLER")
    plt.title(f"SIB1 BLER: Fixed AOD={args.angle}deg, {n_trials} trials, CDLC, SSB03")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
