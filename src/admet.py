"""Medicinal-chemistry / ADMET filters.

Genuine, exact rule logic computed from the descriptors produced in ``prepare``:

* **Lipinski rule of five (Ro5)** - MW <= 500, logP <= 5, H-bond donors <= 5,
  H-bond acceptors <= 10. The classic rule permits at most one violation.
* **Veber** - rotatable bonds <= 10 AND topological polar surface area <= 140.
* **PAINS-like alerts** - a small, transparent set of substructure patterns that
  frequently flag pan-assay interference / promiscuous binders (catechol, quinone,
  nitro, azo, rhodanine-like, Michael-acceptor enone). This is a lightweight
  heuristic, deliberately labelled as such: it is NOT a substitute for a curated
  PAINS filter (e.g. RDKit FilterCatalog), which is the real-mode plug-in point.

``admet_filter`` returns a structured verdict with per-rule detail and a plain
list of human-readable reasons, so a reviewer can see exactly why a compound
passed or failed.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Lightweight PAINS-like SMARTS-ish patterns expressed as regexes over SMILES.
# These are approximate string matches, not full substructure searches. When
# RDKit is available a real FilterCatalog(PAINS) run should replace this.
_PAINS_PATTERNS = {
    "nitro": re.compile(r"\[N\+\]\(=O\)\[O-\]|N\(=O\)=O"),
    "azo": re.compile(r"N=N"),
    "quinone": re.compile(r"O=C1C=CC\(=O\)C=C1|O=C1C=CC\(=O\)"),
    "michael_acceptor_enone": re.compile(r"C=CC\(=O\)"),
    "rhodanine_like": re.compile(r"C1=CSC\(=S\)N1|N1C\(=S\)SC"),
    "catechol": re.compile(r"c1cc\(O\)c\(O\)cc1|Oc1ccccc1O"),
}


def lipinski_ro5(desc: Dict[str, float], max_violations: int = 1) -> Dict[str, object]:
    """Lipinski Ro5 with the standard "at most one violation" pass rule."""
    checks = {
        "mw<=500": float(desc.get("mw", 0.0)) <= 500.0,
        "logp<=5": float(desc.get("logp", 0.0)) <= 5.0,
        "hbd<=5": float(desc.get("hbd", 0.0)) <= 5.0,
        "hba<=10": float(desc.get("hba", 0.0)) <= 10.0,
    }
    violations = [name for name, ok in checks.items() if not ok]
    return {
        "checks": checks,
        "violations": violations,
        "n_violations": len(violations),
        "passed": len(violations) <= max_violations,
    }


def veber(desc: Dict[str, float]) -> Dict[str, object]:
    """Veber oral-bioavailability rule: rotatable bonds <=10 and TPSA <=140."""
    rot_ok = float(desc.get("rotatable_bonds", 0.0)) <= 10.0
    tpsa_ok = float(desc.get("tpsa", 0.0)) <= 140.0
    return {
        "rotatable_bonds<=10": rot_ok,
        "tpsa<=140": tpsa_ok,
        "passed": bool(rot_ok and tpsa_ok),
    }


def pains_alerts(smiles: str) -> Dict[str, object]:
    """Return which lightweight PAINS-like patterns match this SMILES."""
    hits = [name for name, pat in _PAINS_PATTERNS.items() if pat.search(smiles)]
    return {"alerts": hits, "flagged": len(hits) > 0}


def admet_filter(entry: Dict[str, object],
                 use_lipinski: bool = True,
                 use_veber: bool = True,
                 use_pains: bool = True,
                 lipinski_max_violations: int = 1) -> Dict[str, object]:
    """Apply the enabled medicinal-chemistry filters to one prepared ligand.

    ``entry`` must carry the descriptor keys produced by ``prepare`` plus
    ``smiles``. Returns a verdict dict with ``passed`` and ``reasons``.
    """
    reasons: List[str] = []
    result: Dict[str, object] = {"name": entry.get("name"), "passed": True, "reasons": reasons}

    if use_lipinski:
        lip = lipinski_ro5(entry, max_violations=lipinski_max_violations)
        result["lipinski"] = lip
        if not lip["passed"]:
            result["passed"] = False
            reasons.append(f"Lipinski Ro5 failed ({lip['n_violations']} violations: "
                           f"{', '.join(lip['violations'])})")

    if use_veber:
        veb = veber(entry)
        result["veber"] = veb
        if not veb["passed"]:
            result["passed"] = False
            failed = [k for k in ("rotatable_bonds<=10", "tpsa<=140") if not veb[k]]
            reasons.append(f"Veber failed ({', '.join(failed)})")

    if use_pains:
        pns = pains_alerts(str(entry.get("smiles", "")))
        result["pains"] = pns
        if pns["flagged"]:
            result["passed"] = False
            reasons.append(f"PAINS-like alert(s): {', '.join(pns['alerts'])}")

    if result["passed"]:
        reasons.append("passed all enabled medicinal-chemistry filters")
    return result


def filter_library(entries: List[Dict[str, object]], config: Dict[str, object]) -> List[Dict[str, object]]:
    """Run ``admet_filter`` across a library using thresholds from config."""
    admet_cfg = (config or {}).get("admet", {})
    verdicts = []
    for entry in entries:
        verdicts.append(admet_filter(
            entry,
            use_lipinski=admet_cfg.get("lipinski", True),
            use_veber=admet_cfg.get("veber", True),
            use_pains=admet_cfg.get("pains", True),
            lipinski_max_violations=admet_cfg.get("lipinski_max_violations", 1),
        ))
    return verdicts
