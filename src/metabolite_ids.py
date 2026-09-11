"""Canonical extracellular metabolite identifiers shared across GEM sources."""

from __future__ import annotations

import re


_ENCODED_COMPARTMENTS = {
    "_LSQBKTe_RSQBKT": "_e",
    "_LSQBKTc_RSQBKT": "_c",
    "_LSQBKTp_RSQBKT": "_p",
}


def canonical_metabolite_id(value: str) -> str:
    """Normalize SBML/MATLAB encodings without merging stereoisomers.

    The consortium combines CarveMe/BiGG-style models with WCFS1 models that
    encode ``glc-D[e]`` as either ``glc-D_e`` or
    ``glc__D_LSQBKTe_RSQBKT``.  A shared-medium calculation must recognize
    these as the same extracellular pool while retaining D/L and R/S labels.
    """

    result = str(value)
    while result.startswith("M_"):
        result = result[2:]
    for encoded, suffix in _ENCODED_COMPARTMENTS.items():
        result = result.replace(encoded, suffix)
    # COBRA/BiGG convention uses two underscores before stereochemistry.
    # Restrict the rewrite to a stereochemical token followed by a separator
    # or string end so ordinary hyphenated chemical names are not collapsed.
    result = re.sub(r"-(RR|SS|RS|SR|L|D|R|S)(?=_[a-z]$|$)", r"__\1", result)
    return result

