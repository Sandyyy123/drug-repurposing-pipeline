#!/usr/bin/env python3
"""Consensus drug-repurposing pipeline - end-to-end orchestrator.

Runs the full multi-stage funnel and prints a clean per-stage report:

  Stage 0  Load config + target metadata
  Stage 1  Prepare ligands (SMILES -> descriptors)          [src/prepare.py]
  Stage 2  Dock with two engines (DiffDock + Vina)          [src/dock.py]
  Stage 3  Consensus rank aggregation                        [src/consensus.py]
  Stage 4  ADMET / medicinal-chemistry filters               [src/admet.py]
  Stage 5  MD / MM-PBSA re-scoring of top consensus poses    [src/md.py]
  Stage 6  Retrospective validation (EF, ROC-AUC, permute)   [src/validate.py]

Outputs:
  results/shortlist.csv          ranked candidate shortlist (post-ADMET)
  results/validation_report.json retrospective validation metrics

Usage:
  python main.py --demo
  python main.py --demo --seed 7
  python main.py --config config.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

# Make ``src`` importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import prepare, dock, consensus as consensus_mod, admet as admet_mod, md as md_mod, validate as validate_mod


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def banner(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def run_pipeline(config: dict, demo: bool, seed: int) -> dict:
    receptor_id = config["target"]["receptor_id"]
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------------------------------------------- Stage 0
    banner("Stage 0  Configuration")
    print(f"  receptor          : {receptor_id}")
    print(f"  mode              : {'DEMO (synthetic deterministic scores)' if demo else 'REAL (engine adapters)'}")
    print(f"  seed              : {seed}")
    print(f"  descriptor backend: {'RDKit (exact)' if prepare.rdkit_available() else 'RDKit-free approximation'}")

    # ----------------------------------------------------------------- Stage 1
    banner("Stage 1  Ligand preparation (SMILES -> descriptors)")
    actives_raw = prepare.read_smiles_file(config["data"]["actives"])
    decoys_raw = prepare.read_smiles_file(config["data"]["decoys"])
    actives = prepare.prepare_ligands(actives_raw, label=1)
    decoys = prepare.prepare_ligands(decoys_raw, label=0)
    library = actives + decoys
    print(f"  actives loaded    : {len(actives)}")
    print(f"  decoys loaded     : {len(decoys)}")
    print(f"  library size      : {len(library)} candidates")
    print(f"  example descriptor: {actives[0]['name']} "
          f"MW={actives[0]['mw']} logP={actives[0]['logp']} "
          f"HBD={actives[0]['hbd']} HBA={actives[0]['hba']}")

    # ----------------------------------------------------------------- Stage 2
    banner("Stage 2  Docking (two independent engines)")
    scores_by_tool = dock.dock_library(library, receptor_id, seed=seed, demo=demo)
    for tool, scores in scores_by_tool.items():
        arr = np.asarray(scores)
        direction = dock.TOOL_DIRECTION[tool]
        print(f"  {tool:8s} ({'higher better' if direction == 'higher' else 'lower better ' }): "
              f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")

    # ----------------------------------------------------------------- Stage 3
    banner("Stage 3  Consensus rank aggregation")
    weights = config["consensus"]["weights"]
    cons = consensus_mod.consensus_ranking(scores_by_tool, dock.TOOL_DIRECTION, weights=weights)
    order = cons["order"]
    print(f"  weights           : {weights}")
    print(f"  top 5 by consensus:")
    for rank, idx in enumerate(list(order)[:5], start=1):
        idx = int(idx)
        lig = library[idx]
        print(f"    {rank}. {lig['name']:14s} mean_rank={cons['mean_rank'][idx]:.2f} "
              f"z_consensus={cons['z_consensus'][idx]:+.3f} "
              f"(label={'active' if lig['label'] == 1 else 'decoy'})")

    # ----------------------------------------------------------------- Stage 4
    banner("Stage 4  ADMET / medicinal-chemistry filters")
    verdicts = admet_mod.filter_library(library, config)
    n_pass = sum(1 for v in verdicts if v["passed"])
    print(f"  filters           : Lipinski={config['admet']['lipinski']} "
          f"Veber={config['admet']['veber']} PAINS={config['admet']['pains']}")
    print(f"  passed ADMET      : {n_pass} / {len(library)}")
    failed_examples = [v for v in verdicts if not v["passed"]][:3]
    for v in failed_examples:
        print(f"    rejected {v['name']:14s}: {v['reasons'][0]}")

    # ----------------------------------------------------------------- Stage 5
    banner("Stage 5  MD / MM-PBSA re-scoring of top consensus poses")
    top_n = config["run"]["top_n_md"]
    md_deltas = md_mod.rescore_top_poses(library, order, receptor_id,
                                         top_n=top_n, seed=seed, demo=demo)
    print(f"  re-scored top     : {len(md_deltas)} poses")
    for idx, delta in sorted(md_deltas.items(), key=lambda kv: kv[1]):
        print(f"    {library[idx]['name']:14s} MM-PBSA stability delta = {delta:+.3f} kcal/mol")

    # ----------------------------------------------------------- Build shortlist
    passed_flag = {v["name"]: v["passed"] for v in verdicts}
    reasons_map = {v["name"]: "; ".join(v["reasons"]) for v in verdicts}
    rows = []
    for idx in order:
        idx = int(idx)
        lig = library[idx]
        rows.append({
            "rank": None,  # filled after filtering
            "name": lig["name"],
            "smiles": lig["smiles"],
            "label": "active" if lig["label"] == 1 else "decoy",
            "diffdock_score": scores_by_tool["diffdock"][idx],
            "vina_score": scores_by_tool["vina"][idx],
            "consensus_mean_rank": round(float(cons["mean_rank"][idx]), 4),
            "consensus_z": round(float(cons["z_consensus"][idx]), 4),
            "md_mmpbsa_delta": md_deltas.get(idx, ""),
            "mw": lig["mw"], "logp": lig["logp"], "hbd": lig["hbd"], "hba": lig["hba"],
            "tpsa": lig["tpsa"], "rotatable_bonds": lig["rotatable_bonds"],
            "admet_pass": passed_flag[lig["name"]],
            "admet_reasons": reasons_map[lig["name"]],
        })
    df = pd.DataFrame(rows)
    shortlist = df[df["admet_pass"]].copy().reset_index(drop=True)
    shortlist_size = config["output"]["shortlist_size"]
    shortlist = shortlist.head(shortlist_size)
    shortlist["rank"] = range(1, len(shortlist) + 1)
    shortlist_path = config["output"]["shortlist_csv"]
    shortlist.to_csv(shortlist_path, index=False)

    # ----------------------------------------------------------------- Stage 6
    banner("Stage 6  Retrospective validation (rigor control)")
    labels = [lig["label"] for lig in library]
    # Use the consensus z-score (higher = more likely active) as the ranking score.
    val_scores = [float(cons["z_consensus"][i]) for i in range(len(library))]
    report = validate_mod.validate(
        labels, val_scores,
        ef_fractions=config["validation"]["ef_fractions"],
        n_permutations=config["validation"]["n_permutations"],
        seed=seed,
    )
    report["mode"] = "demo" if demo else "real"
    report["receptor"] = receptor_id
    validation_path = config["output"]["validation_json"]
    with open(validation_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"  compounds         : {report['n_compounds']} "
          f"({report['n_actives']} actives / {report['n_decoys']} decoys)")
    print(f"  ROC-AUC           : {report['roc_auc']}")
    for key, val in report["enrichment_factor"].items():
        print(f"  {key:16s}: {val}")
    perm = report["permutation_control"]
    print(f"  permutation p     : {perm['p_value']} "
          f"(null AUC {perm['null_auc_mean']} +/- {perm['null_auc_std']}, "
          f"n={perm['n_permutations']})")
    print(f"  beats null @0.05  : {perm['significant_at_0.05']}")

    # --------------------------------------------------------------- Final report
    banner("PIPELINE COMPLETE")
    print(f"  shortlist written : {shortlist_path} ({len(shortlist)} candidates)")
    print(f"  validation written: {validation_path}")
    print(f"  top candidate     : {shortlist.iloc[0]['name']} "
          f"(consensus_mean_rank={shortlist.iloc[0]['consensus_mean_rank']})")
    print(f"  ROC-AUC={report['roc_auc']}  "
          f"{'  '.join(f'{k}={v}' for k, v in report['enrichment_factor'].items())}  "
          f"perm_p={perm['p_value']}")
    if demo:
        print("  NOTE: demo mode - engine scores are synthetic and deterministic. "
              "Validation numbers are illustrative, not a benchmark result.")

    return {"shortlist": shortlist, "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Consensus drug-repurposing pipeline with retrospective validation controls.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (synthetic deterministic engine scores).")
    parser.add_argument("--real", action="store_true",
                        help="Run in real mode (requires DiffDock/Vina/GROMACS adapters wired).")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2

    config = load_config(args.config)

    # Resolve mode: --real overrides; else --demo or config default.
    if args.real:
        demo = False
    elif args.demo:
        demo = True
    else:
        demo = bool(config["run"].get("demo", True))

    seed = args.seed if args.seed is not None else int(config["run"].get("seed", 42))

    # Determinism for any numpy-based steps outside the seeded permutation test.
    np.random.seed(seed)

    run_pipeline(config, demo=demo, seed=seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
