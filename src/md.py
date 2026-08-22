"""Molecular-dynamics / MM-PBSA re-scoring adapter.

Docking scores are cheap but noisy. A rigorous funnel re-scores the top consensus
poses with a short molecular-dynamics run and an MM-PBSA (or MM-GBSA) free-energy
estimate, which accounts for flexibility and solvation that rigid docking misses.
This is the stability filter: a pose that docks well but falls apart in MD is
demoted.

Real orchestration (what the guarded adapter would run per complex):
  1. Solvate + add ions, then ENERGY MINIMISATION (GROMACS ``gmx grompp``/``mdrun``).
  2. Short equilibration + a brief production MD (a few ns).
  3. MM-PBSA / MM-GBSA re-scoring over the trajectory frames (e.g. ``gmx_MMPBSA``)
     to estimate the binding free energy delta.

Demo mode returns a DETERMINISTIC "stability delta" (kcal/mol; more negative = more
stable) derived from a hash of the complex plus a small drug-likeness term, so the
re-scoring step actually re-orders the top-N in a reproducible way. It is clearly
synthetic, exactly like the docking demo scores.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from typing import Dict, List


def _stable_unit(ligand: str, receptor: str, seed: int) -> float:
    key = f"mmpbsa|{receptor}|{ligand}|{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def mmpbsa_rescore(ligand: Dict[str, object], receptor: str, seed: int = 42,
                   demo: bool = True) -> float:
    """Return an MM-PBSA stability delta in kcal/mol (more negative = more stable)."""
    if not demo:
        return _mmpbsa_real(ligand, receptor, seed)

    noise = _stable_unit(str(ligand.get("smiles", "")), receptor, seed)
    # Small refinement term derived from ligand size (heavier ligands bury more
    # surface and tend to gain a little in MM-PBSA); purely illustrative.
    heavy = float(ligand.get("heavy_atoms", 0.0))
    size_term = min(1.0, heavy / 40.0)
    delta = -(1.5 + 2.5 * noise + 1.0 * size_term)
    return round(delta, 4)


def _mmpbsa_real(ligand: Dict[str, object], receptor: str, seed: int) -> float:  # pragma: no cover
    """Real GROMACS + MM-PBSA re-scoring (guarded adapter).

    Fails loudly when GROMACS is not installed rather than faking a number. The
    body sketches the real command sequence; wire in a project-specific .mdp set
    and topology builder to activate it.
    """
    if shutil.which("gmx") is None and shutil.which("gmx_mpi") is None:
        raise RuntimeError(
            "GROMACS (gmx) not found. Install GROMACS + gmx_MMPBSA, or run in demo "
            "mode. This function is the single integration point for MD re-scoring."
        )
    # Illustrative real sequence (paths/.mdp/topology supplied by the caller):
    #   gmx grompp -f min.mdp   -c solv_ions.gro -p topol.top -o em.tpr
    #   gmx mdrun  -deffnm em
    #   gmx grompp -f md.mdp    -c em.gro        -p topol.top -o md.tpr
    #   gmx mdrun  -deffnm md
    #   gmx_MMPBSA -O -i mmpbsa.in -cs md.tpr -ct md.xtc -cp topol.top ...
    steps = [
        ["gmx", "grompp", "-f", "min.mdp", "-c", "solv_ions.gro", "-p", "topol.top", "-o", "em.tpr"],
        ["gmx", "mdrun", "-deffnm", "em"],
    ]
    for cmd in steps:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    raise RuntimeError(
        "MM-PBSA parsing not wired for this environment; supply mmpbsa.in and parse "
        "the FINAL RESULTS delta-G from gmx_MMPBSA output here."
    )


def rescore_top_poses(ligands: List[Dict[str, object]], order, receptor: str,
                      top_n: int = 5, seed: int = 42, demo: bool = True) -> Dict[int, float]:
    """Re-score the top-N consensus poses.

    Parameters
    ----------
    ligands : the prepared ligand list.
    order   : indices sorted best-to-worst (from ``consensus_ranking``).
    top_n   : how many top poses to re-score.
    Returns ``{ligand_index: stability_delta}`` for the re-scored subset.
    """
    deltas: Dict[int, float] = {}
    for idx in list(order)[:top_n]:
        idx = int(idx)
        deltas[idx] = mmpbsa_rescore(ligands[idx], receptor, seed=seed, demo=demo)
    return deltas
