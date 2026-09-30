// Writes outputs/metabolic_map_20260922/categories.json: {species: {reaction_id: category}}.
// Uses the very same classification the viewer uses (core skeleton first, then atlasGroup),
// so the Python side never re-implements the keyword rules.
const fs = require('fs'), path = require('path');
const out = path.join(__dirname, '../../../../outputs/metabolic_map_20260922');
const D = JSON.parse(fs.readFileSync(path.join(out, 'model_data.json'), 'utf8'));
const cy = fs.readFileSync(path.join(__dirname, '../web/js/cytoscape.js'), 'utf8');
const i = cy.indexOf('function atlasGroup'), j = cy.indexOf('\n', i);
const atlasGroup = new Function('return ' + cy.slice(i, j).replace(/^function atlasGroup/, 'function'))();
const Core = require('../web/js/core.js');
const LABEL = {EMP: '解糖系・糖新生', PPP: 'ペントースリン酸経路', ED: 'ED経路', TCA: 'TCA関連代謝'};
const result = {};
for (const s of D.species) {
  const core = new Map(Core.select(s).reactions.map(x => [x.r.id, LABEL[x.group]]));
  // A reaction that moves the same metabolite between compartments is a transport step, whatever its
  // name says (e.g. LACLt2, proton-coupled symporters): the carbon tracing needs these to be transparent.
  const base = id => id.replace(/_[cep]\d*$/, '');
  const compOf = id => (id.match(/_([cep])\d*$/) || [])[1];
  const moves = r => {
    const seen = new Map();
    for (const m of Object.keys(r.stoich)) { const b = base(m), c = compOf(m); if (!c) continue; if (!seen.has(b)) seen.set(b, new Set()); seen.get(b).add(c); }
    return [...seen.values()].some(set => set.size > 1);
  };
  result[s.short] = Object.fromEntries(s.reactions.map(r => [r.id, r.exchange ? '交換・境界' : moves(r) && !core.has(r.id) ? '輸送' : (core.get(r.id) || atlasGroup(r).replace(/^注釈: /, ''))]));
}
fs.writeFileSync(path.join(out, 'categories.json'), JSON.stringify(result));
console.log('categories.json', Object.fromEntries(Object.entries(result).map(([k, v]) => [k, new Set(Object.values(v)).size + ' categories'])));
