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
  const ex = F.exchanges(s, fl, {unit: 'mmol', solvents: true, net: false});
  const up = ex.uptake.reduce((a, r) => a + r.value, 0), sec = ex.secretion.reduce((a, r) => a + r.value, 0);
  let up2 = 0, sec2 = 0;
  for (const r of s.reactions) if (r.exchange) { const v = fl[r.id] || 0; if (Math.abs(v) > 1e-6) (v < 0 ? (up2 -= v) : (sec2 += v)); }
  assert(Math.abs(up - up2) < 1e-9 && Math.abs(sec - sec2) < 1e-9, s.short + ' exchange totals');
  assert(ex.uptake.every(r => r.value > 0) && ex.secretion.every(r => r.value > 0));
  // Redox pairs of the same element are netted; without netting they stay as two bands.
  const raw = F.exchanges(s, fl, {unit: 'mmol', solvents: false, net: false}), netted = F.exchanges(s, fl, {unit: 'mmol', solvents: false, net: true});
  const sum = a => a.reduce((x, r) => x + r.value, 0);
  assert(Math.abs((sum(raw.uptake) - sum(netted.uptake)) - (sum(raw.secretion) - sum(netted.secretion))) < 1e-9, 'netting removes equal amounts from both sides');
  for (const r of netted.redox) assert(r.value > 0);
  if (s.short === 'OR16') assert(netted.redox.some(r => /Fe/.test(r.from) && /Fe/.test(r.to)), 'OR16 Fe3+/Fe2+ pair is netted');
  // Limit flag: an uptake on its applied limit is marked.
  if (e.uptake_limits) { const lim = F.exchanges(s, fl, {unit: 'mmol', solvents: false, limits: e.uptake_limits, net: false}); assert(lim.uptake.some(r => r.atBound), s.short + ' has limit-bound uptakes'); }
  // Solvents are hidden unless asked for.
  const hidden = F.exchanges(s, fl, {unit: 'mmol', solvents: false, net: false});
  assert(!hidden.uptake.some(r => /^h2o_/.test(r.pool)), 'water hidden by default');
  // Carbon-weighted view drops carbon-free pools (O2, ions).
  const cw = F.exchanges(s, fl, {unit: 'C', solvents: true, net: false});
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
// Prepared media: Pf needs a carbon source it can use; the comparison scenarios are consistent.
assert(snap.scenarios.length >= 3 && snap.scenarios[0].id === 'reference');
const grow = (id, sp) => F.growth(snap.scenarios.find(x => x.id === id).species[sp]).grows;
assert(!grow('reference', 'Pf') && grow('lac', 'Pf') && grow('glc', 'Pf'), 'Pf grows on lactate/glucose, not in the reference medium');
assert(grow('reference', 'OR16') && grow('reference', 'NS21'));
for (const sc of snap.scenarios) for (const s of D.species) { const e = sc.species[s.short]; if (e.status === 'optimal') assert(e.mass_balance_residual < 1e-7, sc.id + ' ' + s.short + ' mass balance'); }
console.log('PASS flux overview aggregation');

// Carbon columns: merging small nodes must conserve totals in every column and on both link sets.
{
  const cf = {flows: [['a', 'X', 'biomass', 5], ['a', 'Y', 'secr:co2_e', 3], ['b', 'X', 'secr:co2_e', 2], ['c', 'Z', 'other', 1], ['d', 'W', 'biomass', 0.5], ['e', 'V', 'other', 0.25]]};
  const col = F.carbonColumns(cf, 2);
  const sum = a => a.reduce((x, r) => x + r.value, 0), total = 11.75;
  assert(Math.abs(col.total - total) < 1e-12);
  for (const part of [col.src, col.cat, col.fate, col.left, col.right]) assert(Math.abs(sum(part) - total) < 1e-12);
  assert(col.src.length === 3 && col.src.at(-1).key === '\u0000other', 'small sources merged into その他, kept last');
  assert(F.carbonColumns({}, 3).total === 0);
  console.log('PASS carbon column aggregation');
}
