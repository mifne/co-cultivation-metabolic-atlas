"""Browser audit of the core map (needs playwright + firefox and a built index.html; skipped otherwise).

1. Every drawn reaction connects exactly the metabolites its stoichiometry (in the drawn direction) says,
   ignoring currency/cofactor metabolites that are hidden by default.
2. No two reactions draw lines on top of each other (collinear overlap > 25 px), also after nutrients are added.
"""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[4]
INDEX = ROOT / 'outputs/metabolic_map_20260922/index.html'
pw = pytest.importorskip('playwright.sync_api')
if not INDEX.exists():
    pytest.skip('index.html not built', allow_module_level=True)

EDGE_AUDIT = """() => {
  const st = flowState, bad = [];
  mapCy.nodes('[kind="fr"]').forEach(rn => {
    const d = rn.data(); if (!d.coreReaction) return;
    const r = st.rxnMap.get(d.reaction), dir = d.direction === -1 ? -1 : 1;
    const keep = id => !id.startsWith('water_instance') && !CoreMetabolism.currency.test(id.replace(/^m_/, ''));
    const sub = new Set(), prod = new Set();
    for (const [m, c] of Object.entries(r.stoich)) if (keep(m)) (c * dir < 0 ? sub : prod).add('m_' + m);
    const gotS = new Set(), gotP = new Set();
    rn.connectedEdges().forEach(e => { const o = e.target().id() === rn.id() ? e.source() : e.target(); if (!o.data('mid') || !keep(o.id().replace(/^m_/, '')) || o.id().startsWith('water_instance')) return; (e.target().id() === rn.id() ? gotS : gotP).add(o.id()); });
    const diff = (a, b) => [...a].filter(x => !b.has(x));
    const miss = [...diff(sub, gotS), ...diff(prod, gotP)], extra = [...diff(gotS, sub), ...diff(gotP, prod)];
    if (miss.length || extra.length) bad.push({reaction: d.reaction, miss, extra});
  });
  return bad;
}"""


@pytest.fixture(scope='module')
def page():
    with pw.sync_playwright() as p:
        try:
            browser = p.firefox.launch()
        except Exception as exc:  # browser binaries not installed
            pytest.skip('firefox for playwright unavailable: %s' % exc)
        pg = browser.new_page(viewport={'width': 1440, 'height': 950})
        pg.goto(INDEX.as_uri())
        pg.wait_for_timeout(800)
        yield pg
        browser.close()


@pytest.mark.parametrize('species', ['OR16', 'NS21', 'Pf'])
def test_drawn_reactions_match_stoichiometry(page, species):
    page.evaluate("sp => openFlow(sp, 'glc__D_e')", species)
    page.wait_for_timeout(1200)
    assert page.evaluate(EDGE_AUDIT) == []


@pytest.mark.parametrize('species', ['OR16', 'NS21', 'Pf'])
def test_no_overlapping_lines(page, species):
    page.evaluate("sp => openFlow(sp, 'glc__D_e')", species)
    page.wait_for_timeout(1500)
    assert page.evaluate("EdgeLanes.overlaps(mapCy).map(o => o.r1 + '/' + o.r2)") == []


def test_no_overlap_after_adding_nutrients(page):
    page.evaluate("openFlow('NS21', 'glc__D_e')")
    page.wait_for_timeout(1200)
    page.evaluate("addMediumRoots(flowState, ['glc__D_e', 'glu__L_e'])")
    page.wait_for_timeout(1500)
    assert page.evaluate(EDGE_AUDIT) == []
    assert page.evaluate("EdgeLanes.overlaps(mapCy).map(o => o.r1 + '/' + o.r2)") == []
