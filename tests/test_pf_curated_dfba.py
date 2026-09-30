"""Opt-in Pf exchange curation in the dFBA entry points (PF_CURATED / --pf-curated).

The switch is OFF by default; OFF must leave loading and CLI options unchanged.
With the switch ON, Pf metabolites secreted into shared pools must become
shared-medium inventory that OR16/NS21 can take up (audited reference).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import main
from main import cultivation_options, load_requested_models, pf_curation_requested

ROOT = Path(__file__).resolve().parents[1]
OR = "Actinoplanes_sp_OR16_lcp"
NS = "Rhizobacter_gummiphilus_NS21"
PF = "Propionibacterium_freudenreichii_shermanii"
OLD_SINKS = {"Ex_S_cpd00239_ext", "Ex_S_cpd00531_ext", "Ex_S_cpd01012_ext", "Ex_S_cpd04097_ext"}


@pytest.mark.parametrize("raw,expected", [(None, False), ("", False), ("0", False), ("off", False),
                                          ("1", True), ("true", True), ("ON", True)])
def test_env_switch_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PF_CURATED", raising=False)
    else:
        monkeypatch.setenv("PF_CURATED", raw)
    assert pf_curation_requested() is expected
    assert pf_curation_requested(True) is True  # explicit flag wins
    assert pf_curation_requested(False) is False


def test_env_switch_rejects_garbage(monkeypatch):
    monkeypatch.setenv("PF_CURATED", "maybe")
    with pytest.raises(ValueError):
        pf_curation_requested()


def test_cli_options_unchanged_when_off(monkeypatch):
    monkeypatch.delenv("PF_CURATED", raising=False)
    off = cultivation_options(argparse.Namespace(dynamics="audited", consortium_profile="pf-helper3"))
    assert "pf_curated" not in off
    on = cultivation_options(argparse.Namespace(dynamics="audited", consortium_profile="pf-helper3", pf_curated=True))
    assert on.pop("pf_curated") is True and on == off
    monkeypatch.setenv("PF_CURATED", "1")
    assert cultivation_options(argparse.Namespace(dynamics="audited", consortium_profile="pf-helper3"))["pf_curated"] is True


@pytest.fixture(scope="module")
def loaded():
    return dict(off=load_requested_models(None, "pf-helper3", pf_curated=False),
                on=load_requested_models(None, "pf-helper3", pf_curated=True))


def test_loader_off_is_uncurated_and_on_is_curated(loaded):
    off, on = loaded["off"][PF], loaded["on"][PF]
    assert {r.id for r in off.sinks} == OLD_SINKS and len(off.exchanges) == 23
    assert len(on.sinks) == 0 and len(on.exchanges) == 195 and [r.id for r in on.demands] == ["Ex_S_biomass_ext"]
    for rid in ["EX_for_e", "EX_succ_e", "EX_co2_e", "EX_pyr_e"]:
        assert on.reactions.get_by_id(rid).lower_bound == 0.0  # secretion-only preserved
    for rid in ["EX_cpd00531_e", "EX_cd2_e", "EX_cpd04097_e"]:
        assert on.reactions.get_by_id(rid).lower_bound == 0.0  # toxic uptake closed
    assert on.reactions.get_by_id("EX_h2s_e").bounds == (-1000.0, 1000.0)  # medium-controlled
    # The other two species are not touched.
    for name in (OR, NS):
        assert len(loaded["on"][name].reactions) == len(loaded["off"][name].reactions)


def test_loader_fails_closed_without_pf():
    with pytest.raises(ValueError, match="no single Pf model"):
        load_requested_models(None, "or16-ns21", pf_curated=True)


def test_curated_pf_secretion_reaches_shared_medium(loaded):
    """One audited substep: formate/succinate/CO2 secreted by Pf enter the
    shared inventory, and NS21 then gets a positive uptake limit for them."""
    from src.audited_dfba import AuditedDFBASimulator
    from src.cultivation_numerics import compute_uptake_limits
    models = loaded["on"]
    design = json.loads((ROOT / "results/equal_budget_comparison_20260908/design.json").read_text())
    medium = dict(design["common_initial_medium"], lac__L_e=1.0)
    sim = AuditedDFBASimulator(models, {OR: .5, NS: .1, PF: .03}, medium, initial_rubber=10.,
                               dt=.025, max_internal_dt=.025, ph_control_target=7.)
    pf_map = sim.exchange_reactions[PF]
    assert {"for_e", "succ_e", "co2_e", "pyr_e", "cpd00531_e"} <= set(pf_map)
    # Uptake of new pools is closed unless the medium provides them.
    limits = compute_uptake_limits(sim.state.metabolites, {n: s.biomass for n, s in sim.state.species.items()},
                                   sim.exchange_reactions, sim.original_bounds, .025, sim.max_uptake_rate)
    assert limits[PF]["for_e"] == 0.0 and limits[PF]["cpd00531_e"] == 0.0
    before = dict(sim.state.metabolites)
    sim.step({}, {}, dynamic_kla=10.)
    solution = sim.last_fba_solutions[PF]
    secreted = {p: float(solution.fluxes[pf_map[p]]) for p in ("for_e", "succ_e")}
    assert max(secreted.values()) > 0.0
    assert all(solution.fluxes[r.id] >= -1e-12 for r in sim.models[PF].exchanges
               if r.id in {"EX_cpd00531_e", "EX_cd2_e", "EX_cpd04097_e"})
    for pool, flux in secreted.items():
        if flux > 0:
            assert sim.state.metabolites.get(pool, 0.) > before.get(pool, 0.)
    limits = compute_uptake_limits(sim.state.metabolites, {n: s.biomass for n, s in sim.state.species.items()},
                                   sim.exchange_reactions, sim.original_bounds, .025, sim.max_uptake_rate)
    produced = [p for p, f in secreted.items() if f > 0]
    assert any(limits[NS][p] > 0.0 for p in produced)
