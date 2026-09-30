// FluxOverview aggregation against the build-time pFBA snapshot.
const fs = require('fs'), path = require('path'), assert = require('assert');
const out = path.join(__dirname, '../../../../outputs/metabolic_map_20260922');
const D = JSON.parse(fs.readFileSync(path.join(out, 'model_data.json'), 'utf8'));
const snap = JSON.parse(fs.readFileSync(path.join(out, 'flux_snapshot.json'), 'utf8'));
const cy = fs.readFileSync(path.join(__dirname, '../web/js/cytoscape.js'), 'utf8');
const i = cy.indexOf('function atlasGroup'), j = cy.indexOf('\n', i);
global.atlasGroup = new Function('return ' + cy.slice(i, j).replace(/^function atlasGroup/, 'function'))();
const F = require('../web/js/flux_overview.js');
const C = require('../web/js/core.js');

assert.equal(snap.model_fingerprint, D.model_fingerprint, 'snapshot belongs to this model');
for (const s of D.species) {
  const e = snap.species[s.short];
  assert(e, s.short + ' missing from snapshot');
  if (!F.growth(e).grows) { assert.equal(s.short, 'Pf', 'only Pf is expected not to grow in the reference medium'); continue; }
  const fl = e.fluxes;
  // Exchange totals equal the exchange fluxes (with solvents included, mmol basis).
  const ex = F.exchanges(s, fl, {unit: 'mmol', solvents: true});
  const up = ex.uptake.reduce((a, r) => a + r.value, 0), sec = ex.secretion.reduce((a, r) => a + r.value, 0);
  let up2 = 0, sec2 = 0;
  for (const r of s.reactions) if (r.exchange) { const v = fl[r.id] || 0; if (Math.abs(v) > 1e-6) (v < 0 ? (up2 -= v) : (sec2 += v)); }
  assert(Math.abs(up - up2) < 1e-9 && Math.abs(sec - sec2) < 1e-9, s.short + ' exchange totals');
  assert(ex.uptake.every(r => r.value > 0) && ex.secretion.every(r => r.value > 0));
  // Solvents are hidden unless asked for.
  const hidden = F.exchanges(s, fl, {unit: 'mmol', solvents: false});
  assert(!hidden.uptake.some(r => /^h2o_/.test(r.pool)), 'water hidden by default');
  // Carbon-weighted view drops carbon-free pools (O2, ions).
  const cw = F.exchanges(s, fl, {unit: 'C', solvents: true});
  assert(!cw.uptake.some(r => r.pool === 'o2_e') && cw.uptake.length < ex.uptake.length);
  // Pathway totals: sum of |flux| over active internal reactions, each reaction counted once.
  const cats = F.pathways(s, fl, C.select(s));
  const total = cats.reduce((a, c) => a + c.total, 0);
  let expected = 0; for (const r of s.reactions) if (!r.exchange) { const v = Math.abs(fl[r.id] || 0); if (v > 1e-6) expected += v; }
  assert(Math.abs(total - expected) < 1e-9, s.short + ' pathway totals ' + total + ' vs ' + expected);
  assert(cats.every((c, k) => k === 0 || cats[k - 1].total >= c.total), 'sorted by activity');
  assert(cats.every(c => c.top.length === c.active && c.top.every((x, k) => k === 0 || Math.abs(c.top[k - 1].flux) >= Math.abs(x.flux))));
  console.log(s.short, 'growth', F.growth(e).value.toFixed(3), 'uptake pools', ex.uptake.length, 'categories', cats.length, 'top:', cats.slice(0, 3).map(c => c.category + ' ' + c.total.toFixed(1)).join(' | '));
}
assert.equal(F.carbons('C6H12O6'), 6); assert.equal(F.carbons('CO2'), 1); assert.equal(F.carbons('Cl'), 0); assert.equal(F.carbons('Ca'), 0); assert.equal(F.carbons(null), 0);
console.log('PASS flux overview aggregation');
