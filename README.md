> **Proprietary - All Rights Reserved.** Copyright (c) 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train machine-learning models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Drug-Repurposing Pipeline (consensus + validation controls)

A reproducible, consensus-based computational drug-repurposing pipeline with retrospective validation controls. It narrows a library of FDA-approved small molecules down to a defensible shortlist against a disease-associated protein target, and checks the scoring approach against a retrospective benchmark before anyone trusts it on true unknowns.

## Overview

A shortlist built from a single docking tool's raw score is fragile, because every docking program has systematic blind spots. This pipeline reduces that risk by requiring candidates to survive a multi-stage funnel of independent methods, and by validating the whole scoring approach against a labelled control (known actives versus decoys) before it is applied to unknowns.

The repository runs end to end today in a self-contained **demo mode** that needs only `numpy`, `pandas`, `scikit-learn`, and `pyyaml`. The heavy scientific engines (AlphaFold3, DiffDock, AutoDock Vina, GROMACS, an ADMET predictor) plug in behind adapter functions. Demo mode substitutes deterministic synthetic scores so the pipeline plumbing, the consensus logic, and the validation maths are all runnable and testable on any machine without a GPU or a docking setup.

## What is inside (the funnel)

```
  library of approved molecules
              |
   [1] prepare    SMILES -> 3D descriptors            (src/prepare.py)
              |
   [2] dock       DiffDock (ML pose) + Vina (physics)  (src/dock.py)   two independent signals
              |
   [3] consensus  rank aggregation; neither tool decides alone  (src/consensus.py)
              |
   [4] ADMET      Lipinski Ro5, Veber, PAINS-like alerts  (src/admet.py)
              |
   [5] MD/MM-PBSA re-score top poses for stability      (src/md.py)
              |
        SHORTLIST  ->  results/shortlist.csv

   [6] validation  runs on known actives vs decoys using the SAME consensus score
                    EF@1%, EF@5%, ROC-AUC, permutation null  (src/validate.py)
                    ->  results/validation_report.json
```

How single-tool bias is reduced, in four ways:

1. **Consensus docking across complementary methods.** An ML pose predictor (DiffDock) is combined with a physics-based scorer (AutoDock Vina). Aggregation is done on ranks rather than raw scores, so the two very different score scales (an ML confidence versus a kcal/mol energy) combine fairly. The code computes a weighted mean-rank (Borda-style) consensus and, in parallel, a weighted z-score consensus that preserves score magnitude (`src/consensus.py`).
2. **Physics re-scoring of top poses.** The top consensus candidates are re-scored with a short molecular-dynamics run followed by an MM-PBSA free-energy estimate, which accounts for flexibility and solvation that rigid docking ignores (`src/md.py`).
3. **Medicinal-chemistry filters.** Independent of any binding score, candidates are checked against Lipinski's rule of five, the Veber oral-bioavailability rules, and a lightweight PAINS-style alert set (`src/admet.py`).
4. **Retrospective validation with a null control.** Before trusting the funnel on unknowns, it is run on a labelled benchmark of known actives and decoys to measure whether the consensus score actually enriches for actives. The harness computes the Enrichment Factor at the top 1% and 5% (`EF@1%`, `EF@5%`), ROC-AUC (implemented via the Mann-Whitney relationship with average-rank tie handling), and a permutation null control (labels shuffled many times, empirical p-value for the observed AUC). See `src/validate.py`.

The validation and consensus maths is unit-tested in `tests/test_validate.py`.

## Tech stack

- **Language:** Python (100% of the repo by GitHub language bytes).
- **Demo-mode dependencies (in `requirements.txt`):** `numpy` (>=1.23), `pandas` (>=1.5), `scikit-learn` (>=1.1), `pyyaml` (>=6.0).
- **Optional heavy engines (intentionally NOT in `requirements.txt`, wired via adapters in `src/`):** RDKit (exact descriptors and a real PAINS catalog), AutoDock Vina (`vina` binary), DiffDock (inference environment plus weights), GROMACS (`gmx`) and `gmx_MMPBSA`, and an ADMET predictor of your choice.
- **Testing:** `pytest`.
- **Config:** a single `config.yaml` drives every stage.

## Repository structure

```
.
|- main.py            end-to-end CLI orchestrator (argparse)
|- config.yaml        target id, consensus weights, ADMET thresholds, seed, output paths
|- requirements.txt   lightweight demo dependencies only
|- src/
|  |- prepare.py      SMILES -> descriptors (RDKit with a documented RDKit-free fallback)
|  |- dock.py         DiffDock + Vina adapters (real and demo paths)
|  |- consensus.py    rank aggregation (mean-rank / Borda plus z-consensus)
|  |- md.py           GROMACS / MM-PBSA re-scoring adapter
|  |- admet.py        Lipinski Ro5, Veber, PAINS-like filters
|  |- validate.py     EF, ROC-AUC, permutation null control
|- data/
|  |- demo_actives.smi   approved-drug SMILES (demo actives)
|  |- demo_decoys.smi    simple-molecule SMILES (demo decoys)
|  |- target_meta.yaml   placeholder target plus structure adapter point
|- tests/
|  |- test_validate.py   correctness tests for the validation and consensus maths
|- LICENSE, NOTICE
```

Observed: 19 paths in the repo tree, including 7 Python files under `src/` and `main.py`, plus `tests/test_validate.py`.

## Quickstart

```bash
pip install -r requirements.txt
python main.py --demo
```

That runs the full funnel on the bundled demo data and prints a per-stage report. It writes two artifacts:

- `results/shortlist.csv` : the ranked candidate shortlist after ADMET filtering, with per-tool scores, the consensus rank, the MM-PBSA delta for re-scored poses, molecular descriptors, and ADMET pass/fail reasons.
- `results/validation_report.json` : the retrospective validation metrics (EF@1%, EF@5%, ROC-AUC, permutation-null p-value).

The run is deterministic. The seed is set in `config.yaml` (default 42) and can be overridden:

```bash
python main.py --demo --seed 7
python main.py --config config.yaml --seed 42   # example
```

Run the tests:

```bash
pytest -q
```

### Real-run mode

Demo mode exists so the methodology is runnable without a GPU or a docking box. For a real screen, provide a real target structure and wire in the engines. Each heavy tool is optional and lives behind a single adapter function in `src/`. To run against real engines: install the tools you need, point `data/target_meta.yaml` at your prepared receptor and binding site, and invoke the adapters with `demo=False` (the README in the repo documents a `python main.py --real` entry point; treat the exact flag as an example and confirm against `main.py`). The adapters are written to raise a clear error if a required binary is missing rather than silently faking a number.

## Outputs

Running `python main.py --demo` generates (into the `results/` directory, which is gitignored) a ranked `shortlist.csv` and a `validation_report.json` as described above. The repository does not ship a committed benchmark result. Demo-mode scores are synthetic and deterministic: they exercise the pipeline, they are not a binding prediction and should not be quoted as a benchmark. The validation maths (EF, ROC-AUC, permutation control) is real and unit-tested, and is the reusable core independent of any single docking engine.

## Scope and honesty

This is a methodology scaffold with pluggable engine adapters, not a trained or benchmarked predictor. The bundled demo decoys are simple molecules; a defensible benchmark uses property-matched decoys (for example DUD-E or DEKOIS), which drop into `data/demo_decoys.smi` unchanged.

## Author

**Dr. Sandeep Grover** - [github.com/Sandyyy123](https://github.com/Sandyyy123)

## License

Proprietary - All Rights Reserved. Copyright (c) 2026 Sandeep Grover. See [LICENSE](LICENSE) and [NOTICE](NOTICE). No permission is granted to use, run, copy, modify, distribute, or train models on this code without prior written permission.
