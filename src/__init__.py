"""Consensus drug-repurposing pipeline package.

Modules
-------
prepare   : target and ligand preparation, molecular descriptors (RDKit + fallback)
dock      : docking-engine adapters (DiffDock, AutoDock Vina) with demo scoring
consensus : rank-aggregation across tools (mean-rank / Borda + z-score consensus)
md        : GROMACS / MM-PBSA re-scoring adapter for top consensus poses
admet     : medicinal-chemistry filters (Lipinski Ro5, Veber, PAINS-like alerts)
validate  : retrospective validation harness (EF, ROC-AUC, permutation control)

The demo mode is fully self-contained and deterministic. Real scientific engines
(AlphaFold3, DiffDock, AutoDock Vina, GROMACS, an ADMET predictor) plug in behind
the adapter functions marked in each module.
"""

__version__ = "0.1.0"
