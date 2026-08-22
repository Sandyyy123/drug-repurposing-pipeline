"""Target and ligand preparation.

Responsibilities
----------------
1. Read ligands (SMILES) from a .smi file.
2. Compute the molecular descriptors the rest of the funnel needs
   (molecular weight, heavy-atom count, H-bond donors/acceptors, rotatable
   bonds, ring count, an estimated logP and TPSA).
3. Describe the protonation / 3D-embedding step that a real run performs
   before docking (guarded behind an adapter).

Two descriptor back-ends are provided:

* ``_descriptors_rdkit``  - exact descriptors via RDKit. Used automatically when
  RDKit is importable.
* ``_descriptors_fallback`` - an RDKit-free approximation computed directly from
  the SMILES string. The numbers are clearly labelled ``approximate`` so nobody
  mistakes them for RDKit output. This keeps the demo runnable on a bare
  ``numpy/pandas`` environment without silently pretending to be RDKit.

Nothing here is a stub: both back-ends return real, computed numbers.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Atomic weights for the common organic subset (used by the fallback estimator).
_ATOMIC_WEIGHT = {
    "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974,
    "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.904, "B": 10.811,
    "H": 1.008,
}

# Two-letter element symbols must be matched before single-letter ones.
_TWO_LETTER = ("Cl", "Br")

try:  # pragma: no cover - exercised only when RDKit is installed
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    _HAVE_RDKIT = True
except Exception:  # RDKit absent -> use the documented approximation instead
    _HAVE_RDKIT = False


def rdkit_available() -> bool:
    """Return True when the exact RDKit descriptor back-end is in use."""
    return _HAVE_RDKIT


def read_smiles_file(path: str) -> List[Dict[str, str]]:
    """Parse a whitespace-delimited .smi file into ``[{name, smiles}, ...]``.

    Lines beginning with ``#`` and blank lines are ignored. Each data line is
    ``<SMILES> <optional name>``; when the name is missing an auto id is used.
    """
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"cpd_{idx:04d}"
            records.append({"name": name, "smiles": smiles})
    return records


def _count_heavy_atoms(smiles: str) -> Dict[str, int]:
    """Approximate per-element heavy-atom counts from a SMILES string.

    Two-letter symbols (Cl, Br) are consumed first, then single-letter organic
    symbols (upper case = aliphatic, lower case = aromatic). Hydrogens and
    structural characters are ignored. This is an approximation used only by the
    RDKit-free fallback.
    """
    counts: Dict[str, int] = {}
    token = smiles
    # Strip bracket-atom charges/isotopes to their element symbol for counting.
    token = re.sub(r"\[(\d*)([A-Za-z][a-z]?)[^\]]*\]", r"\2", token)
    i = 0
    while i < len(token):
        two = token[i:i + 2]
        if two in _TWO_LETTER:
            counts[two] = counts.get(two, 0) + 1
            i += 2
            continue
        ch = token[i]
        upper = ch.upper()
        if upper in _ATOMIC_WEIGHT and upper != "H" and ch.isalpha():
            counts[upper] = counts.get(upper, 0) + 1
        i += 1
    return counts


def _descriptors_fallback(smiles: str) -> Dict[str, float]:
    """RDKit-free approximate descriptor set (documented heuristics).

    These are deliberately simple, transparent estimators. They are good enough
    to keep the funnel running and testable, and every value carries the
    ``approximate=True`` flag so downstream code and readers know the provenance.
    """
    elements = _count_heavy_atoms(smiles)
    heavy = sum(elements.values())

    # Molecular weight: sum of heavy-atom weights plus an average hydrogen
    # contribution (~1.1 H per heavy atom is a rough organic-molecule average).
    mw = sum(_ATOMIC_WEIGHT[e] * n for e, n in elements.items())
    mw += heavy * 1.1 * _ATOMIC_WEIGHT["H"]

    n_atoms = elements.get("N", 0)
    o_atoms = elements.get("O", 0)

    # H-bond acceptors ~ count of N + O. Donors ~ explicit O-H / N-H patterns.
    hba = n_atoms + o_atoms
    hbd = len(re.findall(r"O(?![\)\-=#0-9])", smiles))  # terminal-ish O
    hbd += smiles.count("OH") + smiles.count("NH") + smiles.count("[nH]")

    # Ring count ~ number of ring-closure bond labels / 2 (each ring uses a pair).
    ring_labels = len(re.findall(r"%\d\d", smiles)) + len(re.findall(r"(?<![%\d])\d", smiles))
    rings = ring_labels // 2

    # Rotatable bonds ~ acyclic single bonds; approximate by heavy atoms outside
    # rings scaled down. Documented rough heuristic.
    rot = max(0, int(round((heavy - 3 * rings) * 0.35)))

    # logP: crude atom-contribution proxy (hydrophobic carbons up, polar down).
    logp = 0.20 * elements.get("C", 0) - 0.30 * (n_atoms + o_atoms) \
        + 0.50 * (elements.get("Cl", 0) + elements.get("Br", 0) + elements.get("F", 0))

    # TPSA: ~20 A^2 per polar N/O heteroatom (very rough).
    tpsa = 20.0 * (n_atoms + o_atoms)

    return {
        "mw": round(mw, 3),
        "heavy_atoms": heavy,
        "hbd": int(hbd),
        "hba": int(hba),
        "rotatable_bonds": int(rot),
        "rings": int(rings),
        "logp": round(logp, 3),
        "tpsa": round(tpsa, 3),
        "approximate": True,
    }


def _descriptors_rdkit(smiles: str) -> Dict[str, float]:
    """Exact descriptors via RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # RDKit could not parse the SMILES: fall back rather than crash.
        out = _descriptors_fallback(smiles)
        out["parse_error"] = True
        return out
    return {
        "mw": round(Descriptors.MolWt(mol), 3),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "approximate": False,
    }


def compute_descriptors(smiles: str) -> Dict[str, float]:
    """Compute descriptors, preferring RDKit and falling back transparently."""
    if _HAVE_RDKIT:
        return _descriptors_rdkit(smiles)
    return _descriptors_fallback(smiles)


def embed_3d_adapter(smiles: str, out_path: str = None) -> Dict[str, object]:
    """Protonation + 3D embedding adapter.

    A real run protonates at the target pH, enumerates a small number of
    conformers (e.g. RDKit ETKDG or an external tool), and writes a 3D ligand
    file (SDF/MOL2/PDBQT) for the docking engines. That heavy step is guarded
    here: we report what would run and, when RDKit is present, actually embed a
    single conformer so the path is exercised rather than faked.
    """
    result = {"smiles": smiles, "embedded": False, "engine": "none", "out_path": out_path}
    if not _HAVE_RDKIT:
        result["note"] = "3D embedding skipped (RDKit absent); adapter point for an external embedder."
        return result
    try:  # pragma: no cover - depends on RDKit build
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["note"] = "SMILES did not parse; no conformer generated."
            return result
        mol = Chem.AddHs(mol)
        ok = AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        if ok == 0:
            AllChem.MMFFOptimizeMolecule(mol)
            result["embedded"] = True
            result["engine"] = "rdkit-ETKDG+MMFF"
            if out_path:
                Chem.MolToMolFile(mol, out_path)
        else:
            result["note"] = "ETKDG embedding failed for this ligand."
    except Exception as exc:  # pragma: no cover
        result["note"] = f"embedding adapter error: {exc}"
    return result


def prepare_ligands(records: List[Dict[str, str]], label: int = None) -> List[Dict[str, object]]:
    """Attach descriptors (and an optional label) to each ligand record."""
    prepared: List[Dict[str, object]] = []
    for rec in records:
        desc = compute_descriptors(rec["smiles"])
        entry = {"name": rec["name"], "smiles": rec["smiles"], **desc}
        if label is not None:
            entry["label"] = int(label)
        prepared.append(entry)
    return prepared
