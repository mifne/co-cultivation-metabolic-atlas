"""Carbon-flow decomposition: synthetic model with known answer + real-model smoke test."""
import re
from pathlib import Path

import pytest

import carbon_flow as cf

HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE.parents[2] / 'outputs/metabolic_map_20260922'


def met(mid, formula, name=None):
    return {'id': mid, 'name': name or mid, 'formula': formula, 'compartment': mid.rsplit('_', 1)[-1]}


def rxn(rid, stoich, exchange=False):
    r = {'id': rid, 'stoich': stoich, 'exchange': exchange, 'bounds': [-1000, 1000]}
    if exchange:
        r['pool'] = next(iter(stoich))
    return r


def synthetic():
    """a_e -(transport)-> a_c -(X, +ATP)-> b_c; b_c -> biomass (4) or TCA-like cycle (6):
    b + d -> e -> d + 3 CO2 (cycle), CO2 transported and secreted; g_e is taken up
    and only transported before a demand sink."""
    mets = [met('a_e', 'C3H6O3'), met('a_c', 'C3H6O3'), met('b_c', 'C3H4O3'), met('d_c', 'C2H4O2'),
            met('e_c', 'C5H8O5'), met('co2_c', 'CO2'), met('co2_e', 'CO2'), met('atp_c', 'C10H12N5O13P3'),
            met('adp_c', 'C10H12N5O10P2'), met('g_e', 'CH4O'), met('g_c', 'CH4O')]
    reactions = [
        rxn('EX_a', {'a_e': -1}, True), rxn('Ta', {'a_e': -1, 'a_c': 1}),
        rxn('R1', {'a_c': -1, 'atp_c': -1, 'b_c': 1, 'adp_c': 1}),
        rxn('R2', {'b_c': -1, 'd_c': -1, 'e_c': 1}), rxn('R3', {'e_c': -1, 'd_c': 1, 'co2_c': 3}),
        rxn('BIO', {'b_c': -1}), rxn('Tco2', {'co2_c': -1, 'co2_e': 1}), rxn('EX_co2', {'co2_e': -1}, True),
        rxn('EX_g', {'g_e': -1}, True), rxn('Tg', {'g_e': -1, 'g_c': 1}), rxn('DM_g', {'g_c': -1}),
    ]
    species = {'short': 'toy', 'objective': {'BIO': 1.0}, 'metabolites': {m['id']: m for m in mets},
               'reactions': reactions}
    fluxes = {'EX_a': -10, 'Ta': 10, 'R1': 10, 'R2': 6, 'R3': 6, 'BIO': 4, 'Tco2': 18, 'EX_co2': 18,
              'EX_g': -1, 'Tg': 1, 'DM_g': 1}
    cats = {'EX_a': '交換・境界', 'EX_co2': '交換・境界', 'EX_g': '交換・境界', 'Ta': '輸送', 'Tco2': '輸送',
            'Tg': '輸送', 'R1': 'X', 'R2': 'Y', 'R3': 'Y', 'BIO': '増殖・維持', 'DM_g': '交換・境界'}
    return species, fluxes, cats


def test_synthetic_known_answer():
    species, fluxes, cats = synthetic()
    res = cf.trace(species, fluxes, cats)
    flows = {(p, c, f): v for p, c, f, v in res['flows']}
    assert res['total_uptake_c'] == pytest.approx(31.0)
    assert res['sources']['a_e']['value'] == pytest.approx(30.0)
    # entry = first non-transport reaction (R1, category X); ATP carbon not traced
    assert flows[('a_e', 'X', 'biomass')] == pytest.approx(12.0, rel=1e-12)
    assert flows[('a_e', 'X', 'secr:co2_e')] == pytest.approx(18.0, rel=1e-12)  # through the cycle
    assert flows[('g_e', cf.TRANSPORT_ONLY, 'other')] == pytest.approx(1.0, rel=1e-12)
    assert len(flows) == 3
    assert res['traced_c'] == pytest.approx(31.0) and abs(res['untraced_c']) < 1e-9
    assert res['fates']['secr:co2_e']['kind'] == 'secretion' and res['fates']['biomass']['kind'] == 'biomass'
    assert res['checks']['independent_balance']['residual'] == pytest.approx(0.0, abs=1e-9)
    assert res['checks']['other_breakdown']['demand_sink']['by'] == {'DM_g': pytest.approx(1.0)}
    assert 'not atom mapping' in res['method']


def test_carbon_count_and_currency():
    assert cf.carbon_count('C6H12O6') == 6 and cf.carbon_count('CO2') == 1
    assert cf.carbon_count('CaCl2') == 0 and cf.carbon_count('C61H86CoN13O14PR') == 61
    assert cf.carbon_count(None) is None
    for mid in ('atp_c', 'nadh_p', 'coa_c', 'h_e', 'S_cpd00002_c0', 'cpd00010_c0'):
        assert cf.is_currency(mid), mid
    for mid in ('co2_c', 'hco3_c', 'accoa_c', 'S_cpd00011_c0', 'atpx_c'):
        assert not cf.is_currency(mid), mid


def test_modelseed_list_matches_core_js():
    core = HERE / 'web/js/core.js'
    if not core.exists():
        pytest.skip('core.js missing')
    m = re.search(r'cpd\(\?:([0-9|]+)\)', core.read_text())
    js = set(m.group(1).split('|'))
    # CO2 (cpd00011) is traced here; ACP (cpd11493) added as acyl carrier
    assert set(cf.CURRENCY_MODELSEED) == (js - {'00011'}) | {'11493'}


def test_acyl_coa_strip():
    species = {'objective': {}, 'reactions': [], 'metabolites': {
        'coa_c': met('coa_c', 'C21H32N7O16P3S', 'Coenzyme A'),
        'accoa_c': met('accoa_c', 'C23H34N7O17P3S', 'Acetyl-CoA'),
        'dpcoa_c': met('dpcoa_c', 'C21H33N7O13P2S', 'Dephospho-CoA')}}
    full, eff, info = cf.carbon_table(species)
    assert eff['accoa_c'] == 2 and full['accoa_c'] == 23 and eff['dpcoa_c'] == 21
    assert cf.carbon_table(species, acyl='full')[1]['accoa_c'] == 23


def test_pdh_split_uses_acyl_carbon():
    species, fluxes, cats = synthetic()
    m = species['metabolites']
    m['coa_c'] = met('coa_c', 'C21H32N7O16P3S', 'Coenzyme A')
    m['accoa_c'] = met('accoa_c', 'C23H34N7O17P3S', 'Acetyl-CoA')
    species['reactions'] += [rxn('PDH', {'b_c': -1, 'coa_c': -1, 'accoa_c': 1, 'co2_c': 1}),
                             rxn('BIO2', {'accoa_c': -1, 'coa_c': 1})]
    species['objective'] = {'BIO': 1.0, 'BIO2': 1.0}
    fluxes.update(BIO=2, PDH=2, BIO2=2, Tco2=20, EX_co2=20)
    flows = {(p, c, f): v for p, c, f, v in cf.trace(species, fluxes, cats)['flows']}
    # b carbon: 2 biomass(BIO) x3 + PDH 2 x (2 acetyl -> biomass, 1 -> CO2) + cycle 18
    assert flows[('a_e', 'X', 'biomass')] == pytest.approx(6 + 4)
    assert flows[('a_e', 'X', 'secr:co2_e')] == pytest.approx(18 + 2)


@pytest.mark.skipif(not (DATA_DIR / 'model_data.json').exists() or not (DATA_DIR / 'categories.json').exists(),
                    reason='atlas model_data.json / categories.json missing')
def test_real_model_conservation():
    import fba_service
    d = fba_service.read_model_data()[0]
    medium = dict(d['medium'], **{'lac__L_e': 10.0})
    for short in ('NS21', 'Pf'):
        res = fba_service.check(short, '', 1, ranking=True, medium_override=medium)
        assert res['status'] == 'optimal', res
        cfl = res['carbon_flows']
        assert 'error' not in cfl, cfl
        per = {}
        for p, c, f, v in cfl['flows']:
            per[p] = per.get(p, 0.0) + v
        for p, s in cfl['sources'].items():
            assert abs(per.get(p, 0.0) - s['value']) <= 1e-6 * s['value'] + 1e-8, (short, p)
        assert abs(cfl['untraced_c']) < 1e-6 * cfl['total_uptake_c']
        assert cfl['checks']['runtime_s'] < 2.0
    lac = sum(v for p, c, f, v in cfl['flows'] if p == 'lac__L_e' and f == 'secr:ppa_e')
    ppa = sum(v for p, c, f, v in cfl['flows'] if f == 'secr:ppa_e')
    assert ppa > 1 and lac / ppa > 0.95  # Pf: propionate carbon comes from lactate
