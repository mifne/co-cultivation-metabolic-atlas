"""Experimental cache for COBRApy's expensive ``Model.exchanges`` query.

Supported scope is deliberately narrow: stock COBRApy 0.31.1, structurally
valid standard ``Model``/``Reaction``/``Metabolite`` objects, and unchanged
COBRApy exchange-classifier functions and global classification policy. The
model must be owned by one thread and must not mutate concurrently with an
``exchanges`` call. ``RLock`` protects cache bookkeeping only; it does not make
the COBRA model thread-safe. Retry logic can detect observed sequential edits,
but cannot make arbitrary concurrent mutation (including ABA changes) safe.

This prototype is not production-ready and remains unintegrated. Its mutation
tests cover the standard classifier inputs listed below; they are not a claim
that every possible custom object, monkeypatch, or in-place edit is supported.

Bounds and objectives deliberately do not enter the fingerprint: COBRApy
0.31.1 does not consult reaction reversibility when classifying *exchange*
boundaries. For the supported stock object model, the fingerprint scans the
exchange/external-compartment classifier inputs, plus stoichiometric values and
compartment display metadata, to invalidate common sequential model edits.

The fingerprint is still O(metabolites + reactions + stoichiometric entries)
and allocates nested tuple tokens in one explicit traversal. A cache miss delegates to the
official property, which additionally repeats boundary scans, builds Pandas
objects for external-compartment inference, and queries every reaction. Thus a
stable hit removes those repeated passes without assuming that a GEM is
immutable, but its own allocation and comparison cost may outweigh the saving.
This helper is not integrated into the simulator by itself.
"""

from __future__ import annotations

import threading
import weakref


def _value_token(value):
    """Return a comparison-stable token, including NaN and scalar type."""
    value_type = type(value)
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value_type, value
    if isinstance(value, float):
        return value_type, value.hex()
    if isinstance(value, (list, tuple)):
        return value_type, tuple(_value_token(item) for item in value)
    if isinstance(value, dict):
        return value_type, tuple(
            (_value_token(key), _value_token(item))
            for key, item in value.items()
        )
    if isinstance(value, (set, frozenset)):
        tokens = [_value_token(item) for item in value]
        return value_type, tuple(sorted(tokens, key=repr))
    # COBRA stoichiometry commonly contains NumPy scalar floats. Preserve
    # their dtype and exact bit representation, including signed zero/NaN.
    dtype = getattr(value, 'dtype', None)
    tobytes = getattr(value, 'tobytes', None)
    if dtype is not None and callable(tobytes) and getattr(value, 'ndim', 0) == 0:
        return value_type, str(dtype), tobytes()
    return value_type, repr(value)


def exchange_classification_fingerprint(model):
    """Snapshot supported stock-COBRApy exchange-classifier inputs.

    Reaction/metabolite identities are included as integers, not strong
    references, so storing the result in a weak-key cache cannot keep a model
    alive. Bounds and objective coefficients are intentionally absent. This is
    an experimental COBRApy-0.31.1 fingerprint, not a general contract for
    custom classes or replaced classifier functions.
    """
    metabolites = tuple(
        (
            id(metabolite),
            _value_token(metabolite.id),
            _value_token(metabolite.compartment),
        )
        for metabolite in model.metabolites
    )
    compartment_descriptions = tuple(
        (_value_token(key), _value_token(value))
        for key, value in model._compartments.items()
    )
    reactions = tuple(
        (
            id(reaction),
            _value_token(reaction.id),
            _value_token(reaction.annotation.get('sbo', '')),
            tuple(
                (id(metabolite), _value_token(coefficient))
                for metabolite, coefficient in reaction._metabolites.items()
            ),
        )
        for reaction in model.reactions
    )
    return metabolites, compartment_descriptions, reactions


class ExchangeClassificationCache:
    """Experimentally cache official COBRA exchange indices.

    Returned lists are new list objects, matching ``Model.exchanges``. Cached
    values contain integer positions rather than reactions, avoiding a strong
    value-to-weak-key reference cycle. If a model changes while COBRA performs
    its official query and that sequential change is observed, the query is
    retried against a stable fingerprint.

    Supported use requires stock COBRApy 0.31.1 objects/classifiers and a model
    exclusively owned by one thread during each call. The internal lock guards
    only cache entries and counters; it does not synchronize model access, and
    retry does not guarantee safety under concurrent or ABA mutation.
    """

    def __init__(self, *, maximum_stability_retries=3):
        if (
            isinstance(maximum_stability_retries, bool)
            or not isinstance(maximum_stability_retries, int)
            or maximum_stability_retries < 1
        ):
            raise ValueError('maximum_stability_retries must be a positive integer')
        self.maximum_stability_retries = maximum_stability_retries
        self._entries = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.official_queries = 0
        self.stability_retries = 0

    def exchanges(self, model):
        fingerprint = exchange_classification_fingerprint(model)
        with self._lock:
            entry = self._entries.get(model)
            if entry is not None and entry[0] == fingerprint:
                self.hits += 1
                return [model.reactions[index] for index in entry[1]]
            self.misses += 1

        for attempt in range(self.maximum_stability_retries):
            official = model.exchanges
            with self._lock:
                self.official_queries += 1
            after = exchange_classification_fingerprint(model)
            if after == fingerprint:
                reaction_positions = {reaction: index for index, reaction in enumerate(model.reactions)}
                indices = tuple(reaction_positions[reaction] for reaction in official)
                with self._lock:
                    self._entries[model] = (after, indices)
                return list(official)
            fingerprint = after
            with self._lock:
                self.stability_retries += 1
        raise RuntimeError('Model changed repeatedly during exchange classification')

    def clear(self, model=None):
        """Forget one model or all models without changing the models."""
        with self._lock:
            if model is None:
                self._entries.clear()
            else:
                self._entries.pop(model, None)

    def __len__(self):
        with self._lock:
            return len(self._entries)
