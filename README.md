# SIB1 transmission-diversity simulator

This directory contains the new modular simulator. The historical `py3GPP/`
tree is deliberately not added to `PYTHONPATH`; selected NR helpers may only be
adopted later behind tests.

Validate the first-round configuration and record the runtime environment:

```powershell
py -3.11 -m pip install -e .
py -3.11 -m sib1div.cli validate-config configs/uma_7ghz_48prb.yaml --output outputs/scaffold_validation
```

Run the scaffold tests:

```powershell
py -3.11 -m pytest
```

Generate the offline main codebooks, plots, metrics, and researcher audit report:

```powershell
py -3.11 -m sib1div.cli generate-codebook configs/uma_7ghz_48prb.yaml --output outputs/codebook_7ghz_8h1v
```

Run a bounded preflight smoke simulation (formal Plan 001 execution remains
guarded until every preflight item is cleared):

```powershell
py -3.11 -m sib1div.cli simulate configs/experiments/plan-001.yaml --codebook outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_weights.npz --output outputs/platform_acceptance/smoke --snr 30 --max-drops 1 --estimator lmmse --smoke
```

The package layout follows the development plan. Implementations are added to
the corresponding subpackage as each platform acceptance check is developed.

## Formal simulation records

Every formal scheme simulation must be documented by a matching pair:

```text
research/plan-XXX.md
research/result-XXX.md
```

Create the plan before the run and record the exact system, codebook, receiver,
CDD cyclic-delay, SNR, seed, and stopping configuration. After the run, create
the result document with numerical statistics, output paths, and embedded
curves. See `AGENTS.md` and `research/README.md`; templates are under
`research/templates/`.
