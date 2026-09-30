// Hidden-reaction balance on the core map: displayed + hidden fluxes at a metabolite must cancel (steady state).
const fs = require('fs'), path = require('path'), assert = require('assert');
const out = path.join(__dirname, '../../../../outputs/metabolic_map_20260922');
const D = JSON.parse(fs.readFileSync(path.join(out, 'model_data.json'), 'utf8'));
const snap = JSON.parse(fs.readFileSync(path.join(out, 'flux_snapshot.json'), 'utf8'));
const C = require('../web/js/core.js'), B = require('../web/js/flux_balance.js');
let checked = 0;
for (const sc of snap.scenarios) for (const s of D.species) {
  const e = sc.species[s.short];
  if (!e.fluxes || !(e.objective_value > 1e-6)) continue;
  const spec = C.select(s), shown = new Set(spec.reactions.map(x => x.r.id));
  for (const mid of Object.values(spec.mids)) {
    if (!s.metabolites[mid]) continue;
    const hidden = B.imbalance(s, mid, shown, e.fluxes).net;
    let displayed = 0;
    for (const r of s.reactions) if (shown.has(r.id) && r.stoich[mid] !== undefined) displayed += (e.fluxes[r.id] || 0) * r.stoich[mid];
    assert(Math.abs(hidden + displayed) < 1e-5, `${sc.id} ${s.short} ${mid}: hidden ${hidden} + displayed ${displayed} != 0`);
    checked++;
  }
}
// The example that prompted this check: OR16 pyruvate is only fed by drawn reactions, but 18 mmol/gDW/h leave through undrawn ones.
const or16 = D.species.find(s => s.short === 'OR16'), ref = snap.scenarios[0].species.OR16;
const spec = C.select(or16), pyr = B.imbalance(or16, spec.mids.pyr, new Set(spec.reactions.map(x => x.r.id)), ref.fluxes);
assert(pyr.net < -10 && pyr.who[0][0] === 'ACLS', 'OR16 pyruvate leaves mainly through ACLS');
console.log('PASS flux balance: ' + checked + ' metabolite balances close; OR16 pyr hidden net ' + pyr.net.toFixed(2));
