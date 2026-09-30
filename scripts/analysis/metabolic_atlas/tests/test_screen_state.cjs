// setView / setLayer / reportProblem / noteServerResult with a minimal DOM stub.
const fs = require('fs'), path = require('path'), assert = require('assert'), vm = require('vm');
const src = fs.readFileSync(path.join(__dirname, '../web/js/state.js'), 'utf8');
function el() {
  const e = {style: {}, dataset: {}, children: [], _cls: new Set(), firstChild: {},
    classList: {toggle: (c, on) => on ? e._cls.add(c) : e._cls.delete(c)},
    append(x) { e.children.push(x); x.parent = e }, before() {}, remove() { if (e.parent) e.parent.children.splice(e.parent.children.indexOf(e), 1) },
    querySelector: sel => sel === 'button' ? {} : sel === 'span' ? e.firstChild : null, setAttribute() {}, set innerHTML(v) {}};
  return e;
}
const ids = {};
for (const i of ['mappane', 'searchpane', 'provenance', 'export', 'mapTab', 'flowTab', 'searchTab', 'sourceTab']) ids[i] = el();
const banners = el(); banners.id = 'atlasBanners';
const doc = {body: el(), getElementById: i => i === 'atlasBanners' ? banners : ids[i] || null,
  querySelector: sel => sel === 'main' ? el() : (sel.includes('data-key') ? banners.children.find(c => sel.includes('"' + c.dataset.key + '"')) || null : null),
  createElement: () => el()};
const ctx = {document: doc, CSS: {escape: s => s}, $: i => ids[i], cyHost: el(), escherHost: el(), flowMode: false};
vm.createContext(ctx);
vm.runInContext(src + ';this.AtlasUI=AtlasUI;this.setView=setView;this.setLayer=setLayer;this.reportProblem=reportProblem;this.noteServerResult=noteServerResult;', ctx);

ctx.setView('search');
assert.equal(ids.mappane.style.display, 'none'); assert.equal(ids.searchpane.style.display, 'block');
assert.ok(ids.searchTab._cls.has('active')); assert.ok(!ids.mapTab._cls.has('active'));
ctx.setView('map');
assert.equal(ids.mappane.style.display, 'grid'); assert.ok(ids.mapTab._cls.has('active')); assert.ok(!ids.flowTab._cls.has('active'));
ctx.flowMode = true; ctx.setView('map');
assert.ok(ids.flowTab._cls.has('active')); assert.ok(!ids.mapTab._cls.has('active'));
assert.throws(() => ctx.setView('nope'));

ctx.setLayer('escher');
assert.equal(ctx.escherHost.style.display, 'block'); assert.equal(ctx.cyHost.style.display, 'none'); assert.equal(ctx.AtlasUI.layer, 'escher');
ctx.setLayer('cy');
assert.equal(ctx.cyHost.style.display, 'block'); assert.equal(ctx.escherHost.style.display, 'none');

// Server reachability: a single failure is tolerated, two report, success clears.
ctx.noteServerResult(false); assert.notEqual(doc.body.dataset.server, 'down');
ctx.noteServerResult(false); assert.equal(doc.body.dataset.server, 'down'); assert.equal(banners.children.length, 1);
ctx.noteServerResult(false); assert.equal(banners.children.length, 1, 'no duplicate banner');
ctx.noteServerResult(true); assert.equal(doc.body.dataset.server, 'up'); assert.equal(banners.children.length, 0);
console.log('screen state ok');
