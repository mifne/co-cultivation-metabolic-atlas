"""Carbon-flow decomposition of one species' pFBA solution.

For every carbon source taken up from the medium this module estimates where
its carbon goes, as a conserved table ``source pool -> entry category -> fate``
(see :func:`trace`).

Method
------
PROPORTIONAL CARBON ALLOCATION ON THE FLUX NETWORK -- NOT ATOM MAPPING.
Nodes are carbon-carrying metabolites that carry flux.  In every active
reaction the carbon consumed from each traced substrate is split over the
traced products in proportion to the carbon each product receives
(``C_p * |s_p * v|``).  Per node this defines transition probabilities
(share of that metabolite's total consumption going through each reaction x
product share), i.e. an absorbing Markov chain whose absorbing states are the
sinks (biomass objective, secretion exchanges, other sinks).  Absorption
probabilities ``B = (I - Q)^-1 R`` are obtained with a sparse LU factorisation
(cycles such as the TCA cycle are solved exactly, not iterated).

The entry label (category of the first non-transport reaction after uptake)
comes from a second "pre-entry" chain in which only transport reactions
(category ``'輸送'``) are transient steps; any non-transport step is absorbing
and remembers its category and the product node it lands on, from which the
fate distribution ``B`` is applied.

What this can NOT tell: which atoms really go where.  Proportional allocation
assumes the carbon of all substrates of a reaction is well mixed over its
products; for reactions that combine two carbon skeletons (transaminations,
carboxylations, citrate synthase, ...) the split is an estimate.  It also only
describes this particular pFBA optimum (alternative optima may route
differently), and carbon that enters through untraced cofactors (e.g. the
ribose of ATP used in histidine synthesis) is not attributed to any source.

Handling choices (all reported in ``checks``)
---------------------------------------------
* Currency/cofactor metabolites (``CURRENCY_BIGG`` / ``CURRENCY_MODELSEED``)
  are not nodes.  CO2/HCO3 are NOT currency here (real carbon fates).
  ``ACP`` is added to the task list because it is the acyl-carrier analogue of
  CoA (NS21 ``ACP_c`` is the whole protein, C384).
* Carrier-bound acyl groups (acyl-CoA, acyl-ACP) with ``acyl='strip'``
  (default): their tracing carbon is formula C minus the carrier's C (acetyl-CoA
  C23 -> 2), so e.g. pyruvate dehydrogenase sends 2/3 of pyruvate carbon to
  acetyl-CoA and 1/3 to CO2 instead of 23/24 vs 1/24.  ``acyl='full'`` uses the
  formula carbon (for sensitivity analysis).
* A cofactor product receives a share only when the reaction consumes no
  cofactor/acyl-carrier with at least as much carbon (so ATP->ADP, NADH->NAD,
  acyl-CoA->CoA release carrier carbon without receiving traced carbon, while
  true de-novo synthesis such as adenylosuccinate -> AMP + fumarate does).
  That share is absorbed as ``other`` (reason ``cofactor_synthesis``).
* Formula-less metabolites (Pf lumped macromolecules ``cpdnew*``) get their
  carbon inferred from the carbon balance of a reaction in which they are the
  only unknown (iterated); remaining unknowns count as 0 C.
* Single-metabolite non-exchange boundary reactions consuming an extracellular
  metabolite (Pf ``Ex_S_*_ext``) are secretions not declared as exchanges;
  they become fate ``secr:<metabolite id>`` (``exchange: False`` in ``fates``).
  Other single-metabolite sinks/demands are ``other``.
"""
import re
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

METHOD = 'proportional carbon allocation on the flux network (not atom mapping)'
UNIT = 'C-mmol/gDW/h'
TRANSPORT = '輸送'
TRANSPORT_ONLY = '輸送のみ'
FLUX_EPS = 1e-9
FLOW_EPS = 1e-9

_CARBON = re.compile(r'C(?![a-z])(\d*)')
# BiGG base ids given for this module (co2/hco3/o2 of web/js/core.js deliberately
# NOT excluded: CO2 is a real carbon fate) + ACP (acyl carrier, see docstring).
CURRENCY_BIGG = ('h', 'h2o', 'atp', 'adp', 'amp', 'gtp', 'gdp', 'nad', 'nadh', 'nadp', 'nadph',
                 'coa', 'pi', 'ppi', 'fad', 'fadh2', 'q8', 'q8h2', 'mqn8', 'mql8', '2dmmq8',
                 '2dmmql8', 'fdxox', 'fdxrd', 'ACP')
# ModelSEED list copied from the `currency` regex in web/js/core.js, minus
# cpd00011 (CO2) for consistency with the BiGG list above.
CURRENCY_MODELSEED = ('00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                      '00009', '00010', '00012', '00015', '00018', '00031', '00038', '00067',
                      '00982', '11620', '11621', '15499', '15500', '15560', '15561', '11493')
_CURRENCY = re.compile(r'^(?:(?:%s)_[cep]\d*|(?:S_)?cpd(?:%s)_\w+)$' % (
    '|'.join(map(re.escape, CURRENCY_BIGG)), '|'.join(CURRENCY_MODELSEED)))
_COA_IDS = ('coa_c', 'S_cpd00010_c0')
_ACP_IDS = ('ACP_c', 'S_cpd11493_c0')
_EXTRACELLULAR = re.compile(r'(?:_e\d*|_ext)$')


def carbon_count(formula):
    """Carbon atoms in a formula string; None when the formula is unknown."""
    if not formula:
        return None
    return sum(int(n or 1) for n in _CARBON.findall(str(formula)))


def is_currency(mid):
    return bool(_CURRENCY.match(mid))


def carbon_table(species, acyl='strip'):
    """Return (full, eff, info): formula carbon, tracing carbon and notes."""
    mets = species['metabolites']
    objective = set(species.get('objective') or {})
    full = {m: carbon_count(rec.get('formula')) for m, rec in mets.items()}
    for r in species['reactions']:
        for m in r['stoich']:
            full.setdefault(m, None)
    inferred = {}
    changed = True
    while changed:  # infer lumped-metabolite carbon from single-unknown reactions
        changed = False
        for r in species['reactions']:
            st = r['stoich']
            if r['id'] in objective or len(st) < 2:
                continue
            unknown = [m for m in st if full[m] is None]
            if len(unknown) != 1:
                continue
            u = unknown[0]
            value = -sum(c * full[m] for m, c in st.items() if m != u) / st[u]
            full[u] = max(0.0, float(value))
            inferred[u] = {'carbon': full[u], 'from_reaction': r['id']}
            changed = True
    unknown = sorted(m for m, c in full.items() if c is None)
    for m in unknown:
        full[m] = 0
    eff = dict(full)
    stripped = {}
    carriers = {'CoA': next((full[m] for m in _COA_IDS if full.get(m)), None),
                'ACP': next((full[m] for m in _ACP_IDS if full.get(m)), None)}
    for m, rec in mets.items():
        if is_currency(m):
            continue
        name, base = rec.get('name') or '', m.rsplit('_', 1)[0]
        kind = None
        if base.endswith('coa') or re.search(r'CoA\b', name):
            kind = 'CoA'
        elif 'ACP' in m or re.search(r'\bACP\b|-\[acyl-carrier', name):
            kind = 'ACP'
        carrier = carriers.get(kind) if kind else None
        if carrier and full[m] > carrier:
            stripped[m] = {'carrier': kind, 'formula_c': full[m], 'acyl_c': full[m] - carrier}
            if acyl == 'strip':
                eff[m] = full[m] - carrier
    info = {'inferred_carbon': inferred, 'unknown_carbon': unknown, 'acyl_carrier_metabolites': stripped,
            'carrier_carbon': carriers, 'acyl_mode': acyl}
    return full, eff, info


def _category_map(species, categories):
    if categories is None:
        return {}
    short = species.get('short')
    if short in categories and isinstance(categories[short], dict):
        return categories[short]
    return categories


def trace(species, fluxes, categories, acyl='strip', cofactor_mode='sink', flux_eps=FLUX_EPS):
    """Decompose uptaken carbon into (source pool, entry category, fate) flows.

    species: one ``model_data.json`` species record; fluxes: {reaction id: flux};
    categories: {reaction id: category} (or the whole categories.json).
    acyl: 'strip' (acyl-CoA/ACP carry only acyl carbon) or 'full' (formula C).
    cofactor_mode: how a cofactor produced by de-novo synthesis is handled --
      'sink' (default): the carbon share it would receive is absorbed as
      other/cofactor_synthesis; 'none': it receives no share at all (carbon
      stays on the other products); 'passthrough': the cofactor becomes a node
      in synthesis/degradation reactions only (never in carrier reactions such
      as ATP->ADP, NADH->NAD, acyl-CoA->CoA), for sensitivity analysis.
    """
    if acyl not in ('strip', 'full') or cofactor_mode not in ('sink', 'none', 'passthrough'):
        raise ValueError('invalid acyl/cofactor_mode')
    t0 = time.perf_counter()
    cats = _category_map(species, categories)
    mets = species['metabolites']
    objective = set(species.get('objective') or {})
    full, eff, cinfo = carbon_table(species, acyl=acyl)
    traced = lambda m: eff.get(m, 0) > 0 and not is_currency(m)
    carrier_like = lambda m: is_currency(m) or m in cinfo['acyl_carrier_metabolites']
    passthrough = cofactor_mode == 'passthrough'

    def cofactor_free(m, partners):
        # a cofactor is synthesised/degraded (not carried) when no cofactor or
        # carrier-bound acyl with at least as much carbon is on the other side
        return is_currency(m) and full[m] > 0 and not any(full[p] >= full[m] for p in partners if carrier_like(p))

    def sub_traced(m, st, v):
        if traced(m):
            return True
        return passthrough and cofactor_free(m, [p for p, c in st.items() if c * v > 0])
    name = lambda m: (mets.get(m) or {}).get('name') or m

    node = {}
    def nid(m):
        if m not in node:
            node[m] = len(node)
        return node[m]

    sink = {}       # key -> column
    sink_meta = {}  # key -> (fate_id, reason, reaction)
    def sid(key, fate, reason=None, rid=None):
        if key not in sink:
            sink[key] = len(sink)
            sink_meta[key] = (fate, reason, rid)
        return sink[key]

    fates = {}
    consumption = {}       # node -> total consumed amount (mol)
    edges = []             # (node, target node, amount, is_transport, category)
    sink_edges = []        # (node, sink column, amount, pre-entry label)
    sources, excluded_uptake, nonexchange_inflow = {}, {}, {}
    ind = {'uptake_c': 0.0, 'secretion_c': 0.0, 'nonexchange_secretion_c': 0.0, 'biomass_c': 0.0,
           'other_sink_c': 0.0}
    secretion_flux_c = {}
    active = 0

    for r in species['reactions']:
        rid = r['id']
        v = float(fluxes.get(rid, 0.0) or 0.0)
        if abs(v) <= flux_eps:
            continue
        active += 1
        st = r['stoich']
        cat = cats.get(rid) or 'その他・未分類'
        is_tr = cat == TRANSPORT
        if r.get('exchange'):
            m = next(iter(st))
            pool = r.get('pool') or m
            amount = -st[m] * v  # >0 means the exchange consumes m (secretion)
            if amount < 0:
                c = -amount * full[m]
                if c <= 0:
                    continue
                ind['uptake_c'] += c
                if traced(m):
                    src = sources.setdefault(pool, {'label': name(m), 'value': 0.0, 'metabolite': m})
                    src['value'] += -amount * eff[m]
                    nid(m)
                else:
                    excluded_uptake[pool] = excluded_uptake.get(pool, 0.0) + c
            else:
                ind['secretion_c'] += amount * full[m]
                if traced(m):
                    fate = 'secr:' + pool
                    fates.setdefault(fate, {'label': name(m), 'kind': 'secretion', 'exchange': True, 'pool': pool})
                    sink_edges.append((nid(m), sid(fate, fate), amount, TRANSPORT_ONLY))
                    consumption[node[m]] = consumption.get(node[m], 0.0) + amount
                    secretion_flux_c[fate] = secretion_flux_c.get(fate, 0.0) + amount * eff[m]
            continue
        if rid in objective:
            for m, s in st.items():
                if s * v < 0:
                    ind['biomass_c'] += -s * v * full[m]
                elif s * v > 0:
                    ind['biomass_c'] -= s * v * full[m]
                if s * v < 0 and sub_traced(m, st, v):
                    fates.setdefault('biomass', {'label': 'バイオマス', 'kind': 'biomass'})
                    sink_edges.append((nid(m), sid('biomass', 'biomass'), -s * v, cat))
                    consumption[node[m]] = consumption.get(node[m], 0.0) + (-s * v)
            continue
        if len(st) == 1:
            m, s = next(iter(st.items()))
            if s * v > 0:
                if full[m] > 0:
                    nonexchange_inflow[rid] = s * v * full[m]
                continue
            c = -s * v * full[m]
            ok = traced(m) or (passthrough and is_currency(m) and full[m] > 0)
            if _EXTRACELLULAR.search(m) or (mets.get(m) or {}).get('compartment') in ('e', 'C_e', 'e0'):
                ind['nonexchange_secretion_c'] += c
                if ok:
                    fate = 'secr:' + m
                    fates.setdefault(fate, {'label': name(m), 'kind': 'secretion', 'exchange': False,
                                            'pool': m, 'reaction': rid})
                    sink_edges.append((nid(m), sid(fate, fate), -s * v, TRANSPORT_ONLY))
                    secretion_flux_c[fate] = secretion_flux_c.get(fate, 0.0) - s * v * eff[m]
            else:
                ind['other_sink_c'] += c
                if ok:
                    fates.setdefault('other', {'label': 'その他（シンク・追跡不能）', 'kind': 'other'})
                    sink_edges.append((nid(m), sid(('demand_sink', rid), 'other', 'demand_sink', rid), -s * v, TRANSPORT_ONLY))
            if ok:
                consumption[node[m]] = consumption.get(node[m], 0.0) + (-s * v)
            continue
        consumed = [(m, -s * v) for m, s in st.items() if s * v < 0 and sub_traced(m, st, v)]
        if not consumed:
            for m, s in st.items():  # register produced nodes (e.g. carbon from untraced input)
                if s * v > 0 and traced(m):
                    nid(m)
            continue
        substrates = [m for m, s in st.items() if s * v < 0]
        shares = []
        for m, s in st.items():
            if s * v <= 0:
                continue
            if traced(m):
                shares.append((m, eff[m] * s * v))
            elif cofactor_mode != 'none' and cofactor_free(m, substrates):
                shares.append((m if passthrough else ('cofactor', rid, m), full[m] * s * v))
        total = sum(w for _, w in shares)
        label = TRANSPORT_ONLY if is_tr else cat
        for m, a in consumed:
            i = nid(m)
            consumption[i] = consumption.get(i, 0.0) + a
            if total <= 0:
                fates.setdefault('other', {'label': 'その他（シンク・追跡不能）', 'kind': 'other'})
                sink_edges.append((i, sid(('no_traced_product', rid), 'other', 'no_traced_product', rid), a, label))
                continue
            for p, w in shares:
                if isinstance(p, tuple):
                    fates.setdefault('other', {'label': 'その他（シンク・追跡不能）', 'kind': 'other'})
                    sink_edges.append((i, sid(('cofactor_synthesis', rid), 'other', 'cofactor_synthesis', rid), a * w / total, label))
                else:
                    edges.append((i, nid(p), a * w / total, is_tr, cat))

    n = len(node)
    # Nodes without consumers or unable to reach any sink -> 'other'.
    has_out = np.zeros(n, bool)
    for i, *_ in sink_edges:
        has_out[i] = True
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for i, j, a, *_ in edges:
        succ[i].append(j)
        pred[j].append(i)
    reach = has_out.copy()
    stack = [i for i in range(n) if reach[i]]
    while stack:
        j = stack.pop()
        for i in pred[j]:
            if not reach[i]:
                reach[i] = True
                stack.append(i)
    dead = [i for i in range(n) if not reach[i]]
    inv = {i: m for m, i in node.items()}
    dead_info = {}
    for i in dead:
        reason = 'trapped' if consumption.get(i, 0) > 0 else 'no_consumer'
        fates.setdefault('other', {'label': 'その他（シンク・追跡不能）', 'kind': 'other'})
        sink_edges.append((i, sid((reason, inv[i]), 'other', reason, inv[i]), 1.0, TRANSPORT_ONLY))
        consumption[i] = 1.0
        dead_info[inv[i]] = reason
    dead_set = set(dead)
    edges = [e for e in edges if e[0] not in dead_set]
    denom = np.array([consumption.get(i, 0.0) for i in range(n)])

    K = len(sink)
    qi = [e[0] for e in edges]; qj = [e[1] for e in edges]
    qv = [e[2] / denom[e[0]] for e in edges]
    Q = sp.csc_matrix((qv, (qi, qj)), shape=(n, n))
    R = sp.csc_matrix(([e[2] / denom[e[0]] for e in sink_edges], ([e[0] for e in sink_edges], [e[1] for e in sink_edges])), shape=(n, K))
    eye = sp.identity(n, format='csc')
    B = splu((eye - Q).tocsc()).solve(R.toarray()) if n else np.zeros((0, K))

    # Pre-entry chain: transport steps transient, non-transport steps absorbing.
    tr = [k for k, e in enumerate(edges) if e[3]]
    QT = sp.csc_matrix(([qv[k] for k in tr], ([qi[k] for k in tr], [qj[k] for k in tr])), shape=(n, n))
    entry_mats = {}
    for k, e in enumerate(edges):
        if not e[3]:
            entry_mats.setdefault(e[4], []).append((e[0], e[1], qv[k]))
    entry_mats = {c: sp.csr_matrix(([x[2] for x in lst], ([x[0] for x in lst], [x[1] for x in lst])), shape=(n, n))
                  for c, lst in entry_mats.items()}
    direct = {}
    for i, k, a, lab in sink_edges:
        direct.setdefault(lab, []).append((i, k, a / denom[i]))
    direct = {c: sp.csr_matrix(([x[2] for x in lst], ([x[0] for x in lst], [x[1] for x in lst])), shape=(n, K))
              for c, lst in direct.items()}
    luT = splu((eye - QT).tocsc()) if n else None

    col_fate = [None] * K
    for key, col in sink.items():
        col_fate[col] = sink_meta[key]
    flows = {}
    other_detail = {}
    per_source = {}
    for pool, src in sources.items():
        i = node[src['metabolite']]
        e = np.zeros(n); e[i] = 1.0
        x = luT.solve(e, trans='T')
        by_label = {}
        for c, E in entry_mats.items():
            by_label[c] = (E.T @ x) @ B
        for c, D in direct.items():
            by_label[c] = by_label.get(c, 0) + D.T @ x
        total = 0.0
        for c, vec in by_label.items():
            vec = np.asarray(vec).ravel() * src['value']
            for col in np.nonzero(np.abs(vec) > 0)[0]:
                fate, reason, rid = col_fate[col]
                val = float(vec[col])
                flows[(pool, c, fate)] = flows.get((pool, c, fate), 0.0) + val
                total += val
                if reason:
                    d = other_detail.setdefault(reason, {'value': 0.0, 'by': {}})
                    d['value'] += val
                    d['by'][rid] = d['by'].get(rid, 0.0) + val
        per_source[pool] = total
    out_flows = sorted(([p, c, f, v] for (p, c, f), v in flows.items() if v > FLOW_EPS), key=lambda x: -x[3])
    traced_c = sum(x[3] for x in out_flows)
    total_uptake = ind['uptake_c']
    source_c = sum(s['value'] for s in sources.values())
    fate_traced = {}
    for p, c, f, v in out_flows:
        fate_traced[f] = fate_traced.get(f, 0.0) + v
    used = {x[2] for x in out_flows}
    fates = {k: v for k, v in fates.items() if k in used}
    rel = {p: abs(per_source[p] - s['value']) / s['value'] for p, s in sources.items() if s['value'] > 0}
    balance_out = ind['secretion_c'] + ind['nonexchange_secretion_c'] + ind['biomass_c'] + ind['other_sink_c']
    for d in other_detail.values():
        d['by'] = dict(sorted(((k, v) for k, v in d['by'].items() if v > FLOW_EPS), key=lambda kv: -kv[1])[:8])
    checks = {
        'per_source_max_rel_error': max(rel.values(), default=0.0),
        'source_carbon_c': source_c,
        'excluded_cofactor_uptake_c': excluded_uptake,
        'independent_balance': dict(ind, residual=total_uptake - balance_out + sum(nonexchange_inflow.values()),
                                    nonexchange_inflow_c=sum(nonexchange_inflow.values()),
                                    note='formula carbon of exchange/boundary fluxes and net objective stoichiometry x flux; residual = uptake + non-exchange inflow - outputs'),
        'traced_vs_exchange_flux': {f: {'traced': fate_traced.get(f, 0.0), 'flux_c': secretion_flux_c.get(f, 0.0)} for f in sorted(set(fate_traced) | set(secretion_flux_c)) if f.startswith('secr:')},
        'biomass_traced_c': fate_traced.get('biomass', 0.0),
        'other_breakdown': other_detail,
        'dead_nodes': dead_info,
        'nonexchange_inflow': nonexchange_inflow,
        'inferred_carbon': cinfo['inferred_carbon'],
        'unknown_carbon': cinfo['unknown_carbon'],
        'acyl_mode': acyl,
        'acyl_carrier_metabolites': len(cinfo['acyl_carrier_metabolites']),
        'cofactor_mode': cofactor_mode,
        'active_reactions': active, 'nodes': n, 'edges': len(edges), 'sinks': K,
        'runtime_s': time.perf_counter() - t0,
    }
    return {
        'method': METHOD,
        'unit': UNIT,
        'total_uptake_c': total_uptake,
        'traced_c': traced_c,
        'untraced_c': total_uptake - traced_c,
        'flows': out_flows,
        'fates': fates,
        'sources': {p: {'label': s['label'], 'value': s['value']} for p, s in sources.items()},
        'checks': checks,
        'assumptions': [
            '反応内で基質炭素を生成物の炭素量に比例配分する推定（原子マッピングではない）',
            '補酵素・通貨代謝物（ATP, NAD(P)H, CoA, ACP, キノン等）は炭素を運ばない。CO2/HCO3は追跡する',
            'アシルCoA/アシルACPは担体炭素を除いたアシル部分のみを追跡（acyl_mode）',
            'このpFBA解1つに対する分解。別の最適解では経路が変わり得る',
        ],
    }
