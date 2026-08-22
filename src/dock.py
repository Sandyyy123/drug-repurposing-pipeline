"""Docking-engine adapters.

The pipeline is engine-agnostic. Two adapters are provided:

* ``diffdock_dock`` - a machine-learning pose predictor (DiffDock). Reports a
  *confidence* score where HIGHER is better.
* ``vina_dock``     - a physics-based scorer (AutoDock Vina). Reports a predicted
  binding free energy in kcal/mol where LOWER (more negative) is better.

Each adapter has the same two-path shape:

1. **Real mode** (``demo=False``): build the engine command, run it via
   ``subprocess``, parse the score from the output. This path is guarded and
   documented; it only fires when the binary is available and inputs are real.
2. **Demo mode** (``demo=True``, the default here): return a DETERMINISTIC,
   reproducible pseudo-score so the whole funnel runs end-to-end and is testable
   on any machine with no scientific binaries installed.

Honesty note on the demo score
-------------------------------
The demo score is synthetic. It is built from:
  (a) a deterministic hash of (ligand, receptor, tool, seed) that stands in for
      tool-specific pose noise, so the two engines disagree in a realistic way and
      the consensus step has something non-trivial to reconcile; plus
  (b) a small drug-likeness term derived from the ligand's own descriptors, so
      the bundled real-drug actives tend to rank above the deliberately simple
      decoys. This lets the retrospective validation harness demonstrate a
      non-trivial (illustrative, not benchmarked) enrichment on the demo set.
It is NOT a real binding prediction. Swap in the real ``subprocess`` calls to get
real scores; the rest of the funnel does not change.

The direction of each tool ("higher"/"lower" is better) is exported via
``TOOL_DIRECTION`` so downstream rank aggregation orients scores correctly.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from typing import Dict, List

# Which direction is "better" for each tool's raw score.
TOOL_DIRECTION: Dict[str, str] = {
    "diffdock": "higher",  # ML confidence: higher is better
    "vina": "lower",       # binding energy kcal/mol: more negative is better
}


def _stable_unit(ligand: str, receptor: str, tool: str, seed: int) -> float:
    """Deterministic float in [0, 1) from the (ligand, receptor, tool, seed) key.

    Uses an MD5 digest so the value is stable across machines and Python runs
    (Python's builtin ``hash`` is salted per process and must not be used here).
    """
    key = f"{tool}|{receptor}|{ligand}|{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def _drug_likeness(descriptors: Dict[str, float]) -> float:
    """A transparent 0..1 drug-likeness proxy from descriptors.

    Rewards Ro5-compatible ranges (MW ~<=500, logP ~<=5, HBD<=5, HBA<=10) and a
    little molecular complexity (heavy atoms, rings). Deliberately simple and
    fully deterministic. Used ONLY to shape the synthetic demo score.
    """
    mw = float(descriptors.get("mw", 0.0))
    logp = float(descriptors.get("logp", 0.0))
    hbd = float(descriptors.get("hbd", 0.0))
    hba = float(descriptors.get("hba", 0.0))
    heavy = float(descriptors.get("heavy_atoms", 0.0))
    rings = float(descriptors.get("rings", 0.0))

    def window(value, lo, hi):
        # 1.0 inside [lo, hi], decaying linearly outside it.
        if lo <= value <= hi:
            return 1.0
        span = (hi - lo) or 1.0
        dist = (lo - value) if value < lo else (value - hi)
        return max(0.0, 1.0 - dist / span)

    score = (
        0.30 * window(mw, 250.0, 500.0)
        + 0.20 * window(logp, 0.0, 5.0)
        + 0.15 * window(hbd, 0.0, 5.0)
        + 0.15 * window(hba, 1.0, 10.0)
        + 0.10 * min(1.0, heavy / 25.0)
        + 0.10 * min(1.0, rings / 3.0)
    )
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# DiffDock adapter
# --------------------------------------------------------------------------- #
def diffdock_dock(ligand: Dict[str, object], receptor: str, seed: int = 42,
                  demo: bool = True) -> float:
    """Return a DiffDock-style confidence score (higher is better)."""
    if not demo:
        return _diffdock_real(ligand, receptor, seed)

    noise = _stable_unit(str(ligand.get("smiles", "")), receptor, "diffdock", seed)
    likeness = _drug_likeness(ligand)
    # Confidence in roughly [-1, 1.4]: drug-likeness pulls it up, noise spreads it.
    score = 0.9 * likeness + 0.9 * noise - 0.5
    return round(score, 4)


def _diffdock_real(ligand: Dict[str, object], receptor: str, seed: int) -> float:  # pragma: no cover
    """Real DiffDock invocation (guarded adapter).

    A real integration writes the prepared receptor + ligand, calls the DiffDock
    inference script, and parses the top pose's confidence from its output. The
    binary is not assumed present in this demo repo, so we fail loudly rather
    than silently faking a number.
    """
    if shutil.which("diffdock") is None:
        raise RuntimeError(
            "diffdock binary not found. Install DiffDock and expose it on PATH, "
            "or run in demo mode. This function is the single integration point."
        )
    cmd = [
        "diffdock", "--protein_path", receptor,
        "--ligand", str(ligand.get("sdf_path", ligand.get("smiles", ""))),
        "--samples_per_complex", "10", "--seed", str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if "confidence" in line.lower():
            return float(line.split()[-1])
    raise RuntimeError("Could not parse DiffDock confidence from output.")


# --------------------------------------------------------------------------- #
# AutoDock Vina adapter
# --------------------------------------------------------------------------- #
def vina_dock(ligand: Dict[str, object], receptor: str, seed: int = 42,
              demo: bool = True) -> float:
    """Return an AutoDock Vina-style binding energy in kcal/mol (lower is better)."""
    if not demo:
        return _vina_real(ligand, receptor, seed)

    noise = _stable_unit(str(ligand.get("smiles", "")), receptor, "vina", seed)
    likeness = _drug_likeness(ligand)
    # Energy in roughly [-13, -5] kcal/mol: more drug-like -> more negative.
    score = -(5.0 + 5.0 * likeness + 3.0 * noise)
    return round(score, 4)


def _vina_real(ligand: Dict[str, object], receptor: str, seed: int) -> float:  # pragma: no cover
    """Real AutoDock Vina invocation (guarded adapter).

    A real integration supplies a PDBQT receptor + ligand and a search box
    (centre and size from ``config.yaml``), runs Vina, and parses the best mode
    affinity. The binary is not assumed present here.
    """
    vina_bin = shutil.which("vina") or shutil.which("vina_1.2.5")
    if vina_bin is None:
        raise RuntimeError(
            "vina binary not found. Install AutoDock Vina and expose it on PATH, "
            "or run in demo mode. This function is the single integration point."
        )
    cmd = [
        vina_bin, "--receptor", receptor,
        "--ligand", str(ligand.get("pdbqt_path", "")),
        "--seed", str(seed), "--exhaustiveness", "8",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("1 "):  # first (best) docking mode row
            return float(stripped.split()[1])
    raise RuntimeError("Could not parse Vina affinity from output.")


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def dock_library(ligands: List[Dict[str, object]], receptor: str, seed: int = 42,
                 demo: bool = True) -> Dict[str, List[float]]:
    """Dock every ligand with both engines.

    Returns ``{"diffdock": [...], "vina": [...]}`` with one score per ligand,
    aligned to the input order.
    """
    scores: Dict[str, List[float]] = {"diffdock": [], "vina": []}
    for lig in ligands:
        scores["diffdock"].append(diffdock_dock(lig, receptor, seed=seed, demo=demo))
        scores["vina"].append(vina_dock(lig, receptor, seed=seed, demo=demo))
    return scores
