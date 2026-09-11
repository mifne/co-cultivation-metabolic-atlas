"""Mutation coverage for the opt-in COBRA exchange-classification cache."""

from copy import deepcopy
import gc
import weakref

import numpy as np
import pytest
from cobra import Metabolite, Model, Reaction
import cobra.core.model as cobra_model_module

from src.exchange_classification_cache import (
    ExchangeClassificationCache,
    exchange_classification_fingerprint,
)


def _reaction(reaction_id, metabolites, sbo=None):
    reaction = Reaction(reaction_id)
    reaction.add_metabolites(metabolites)
    reaction.bounds = (-10., 1000.)
    if sbo is not None:
        reaction.annotation['sbo'] = sbo
    return reaction


def _model():
    model = Model('classification')
    external = Metabolite('carbon_e', compartment='e')
    external_2 = Metabolite('factor_e', compartment='e')
    internal = Metabolite('carbon_c', compartment='c')
    exchange = _reaction('EX_carbon_e', {external: -1.})
    transport = _reaction('transport', {external: -1., internal: 1.})
    demand = _reaction('DM_carbon_c', {internal: -1.})
    # SBO annotations dominate compartment and ID heuristics in COBRA 0.31.1.
    annotated_exchange = _reaction(
        'odd_internal_boundary', {internal: -1.}, 'SBO:0000627'
    )
    annotated_sink = _reaction(
        'EX_but_sink', {external_2: -1.}, 'SBO:0000632'
    )
    model.add_reactions(
        [exchange, transport, demand, annotated_exchange, annotated_sink]
    )
    model.compartments = {'e': 'external medium', 'c': 'cytoplasm'}
    return model


def _ids(reactions):
    return [reaction.id for reaction in reactions]


@pytest.fixture
def official_counter(monkeypatch):
    original = cobra_model_module.find_boundary_types
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cobra_model_module, 'find_boundary_types', tracked)
    return calls


def test_stable_hit_skips_official_query_returns_fresh_exact_ordered_list(official_counter):
    model = _model()
    cache = ExchangeClassificationCache()
    first = cache.exchanges(model)
    assert _ids(first) == ['EX_carbon_e', 'odd_internal_boundary']
    assert len(official_counter) == 1
    first.clear()
    second = cache.exchanges(model)
    assert _ids(second) == ['EX_carbon_e', 'odd_internal_boundary']
    assert first is not second
    assert len(official_counter) == 1
    assert cache.hits == 1 and cache.misses == 1


@pytest.mark.parametrize('mutation', [
    'add_reaction', 'remove_reaction', 'stoichiometric_value',
    'stoichiometric_structure', 'metabolite_compartment', 'sbo',
    'sbo_list_inplace', 'reaction_id', 'metabolite_add',
    'compartment_display',
])
def test_every_required_model_mutation_invalidates_and_requeries(mutation, official_counter):
    model = _model()
    cache = ExchangeClassificationCache()
    before = _ids(cache.exchanges(model))
    exchange = model.reactions.EX_carbon_e
    external = model.metabolites.carbon_e
    if mutation == 'add_reaction':
        new_metabolite = Metabolite('new_e', compartment='e')
        model.add_reactions([_reaction('EX_new_e', {new_metabolite: -1.})])
    elif mutation == 'remove_reaction':
        model.remove_reactions([exchange])
    elif mutation == 'stoichiometric_value':
        exchange.add_metabolites({external: np.float32(-2.)}, combine=False)
    elif mutation == 'stoichiometric_structure':
        exchange.add_metabolites({model.metabolites.carbon_c: 1.})
    elif mutation == 'metabolite_compartment':
        external.compartment = 'c'
    elif mutation == 'sbo':
        exchange.annotation['sbo'] = 'SBO:0000632'
    elif mutation == 'sbo_list_inplace':
        exchange.annotation['sbo'] = ['SBO:0000627', 'first']
        cache.exchanges(model)
        exchange.annotation['sbo'][0] = 'SBO:0000632'
    elif mutation == 'reaction_id':
        exchange.id = 'sink_carbon_e'
    elif mutation == 'metabolite_add':
        model.add_metabolites([Metabolite('unconnected_x', compartment='x')])
    else:
        model.compartments = {'e': 'changed display name'}
    calls_before = len(official_counter)
    result = cache.exchanges(model)
    assert len(official_counter) == calls_before + 1
    # A direct official query confirms identity and ordering after the edit.
    expected = model.exchanges
    assert result == expected
    if mutation in {'stoichiometric_value', 'metabolite_add', 'compartment_display'}:
        assert _ids(result) == before


def test_bound_and_objective_changes_do_not_invalidate_exchange_classification(official_counter):
    model = _model()
    cache = ExchangeClassificationCache()
    expected = cache.exchanges(model)
    exchange = model.reactions.EX_carbon_e
    exchange.bounds = (-.25, 17.)
    model.objective = model.reactions.transport
    model.objective_direction = 'min'
    assert cache.exchanges(model) == expected
    assert len(official_counter) == 1


def test_model_context_mutation_and_restoration_are_each_observed(official_counter):
    model = _model()
    cache = ExchangeClassificationCache()
    baseline = _ids(cache.exchanges(model))
    with model:
        temporary = Metabolite('temporary_e', compartment='e')
        model.add_reactions([_reaction('EX_temporary_e', {temporary: -1.})])
        assert 'EX_temporary_e' in _ids(cache.exchanges(model))
    assert _ids(cache.exchanges(model)) == baseline
    assert len(official_counter) == 3


def test_deepcopy_is_a_distinct_cache_entry_with_cloned_reactions(official_counter):
    model = _model()
    cache = ExchangeClassificationCache()
    original = cache.exchanges(model)
    clone = deepcopy(model)
    copied = cache.exchanges(clone)
    assert _ids(copied) == _ids(original)
    assert all(a is not b for a, b in zip(original, copied))
    assert len(cache) == 2 and len(official_counter) == 2


def test_cached_value_does_not_keep_weak_model_key_alive():
    cache = ExchangeClassificationCache()
    model = _model()
    cache.exchanges(model)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None
    assert len(cache) == 0


def test_change_during_official_query_retries_until_fingerprint_is_stable(monkeypatch):
    model = _model()
    cache = ExchangeClassificationCache()
    original = cobra_model_module.find_boundary_types
    calls = []

    def mutate_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            model.reactions.EX_carbon_e.id = 'sink_carbon_e'
        return original(*args, **kwargs)

    monkeypatch.setattr(cobra_model_module, 'find_boundary_types', mutate_once)
    result = cache.exchanges(model)
    assert _ids(result) == _ids(model.exchanges)
    assert cache.stability_retries == 1
    assert cache.official_queries == 2


def test_repeated_change_during_query_fails_closed_without_cache_entry(monkeypatch):
    model = _model()
    cache = ExchangeClassificationCache(maximum_stability_retries=2)
    original = cobra_model_module.find_boundary_types
    exchange = model.reactions.EX_carbon_e
    count = 0

    def always_mutate(*args, **kwargs):
        nonlocal count
        count += 1
        exchange.id = (
            'EX_carbon_e' if count % 2 == 0 else 'sink_carbon_e'
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(cobra_model_module, 'find_boundary_types', always_mutate)
    with pytest.raises(RuntimeError, match='changed repeatedly'):
        cache.exchanges(model)
    assert len(cache) == 0


def test_clear_one_or_all_forces_new_official_query(official_counter):
    first, second = _model(), deepcopy(_model())
    cache = ExchangeClassificationCache()
    cache.exchanges(first); cache.exchanges(second)
    cache.clear(first)
    assert len(cache) == 1
    cache.exchanges(first)
    cache.clear()
    assert len(cache) == 0
    cache.exchanges(first)
    assert len(official_counter) == 4


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_stability_retry_count_rejected(value):
    with pytest.raises(ValueError):
        ExchangeClassificationCache(maximum_stability_retries=value)


def test_fingerprint_is_nan_stable_and_detects_exact_scalar_representation():
    model = _model()
    reaction = model.reactions.EX_carbon_e
    metabolite = model.metabolites.carbon_e
    reaction._metabolites[metabolite] = float('nan')
    first = exchange_classification_fingerprint(model)
    assert first == exchange_classification_fingerprint(model)
    reaction._metabolites[metabolite] = np.float32(np.nan)
    second = exchange_classification_fingerprint(model)
    assert first != second
