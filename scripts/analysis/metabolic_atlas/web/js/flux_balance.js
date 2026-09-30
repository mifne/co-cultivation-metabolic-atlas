/* Flux balance notes on the core map.
 * The map shows only the curated skeleton and the chosen nutrient pipelines. Where the FBA solution
 * moves flux between a displayed metabolite and reactions that are NOT drawn (pyruvate -> valine
 * synthesis, r5p <- nucleotide salvage, ...), particles arrive or leave with no visible line.
 * Each such metabolite gets an orange note with the net flux through the hidden reactions.
 */
/* Pure part (unit-tested): net flux of a metabolite through reactions that are not drawn. */
const FluxBalance = (() => {
  function imbalance(species, mid, shown, fluxes) {
    let net = 0;
    const who = [];
    for (const r of species.reactions) {
      const c = r.stoich[mid];
      if (c === undefined || shown.has(r.id)) continue;
      const v = (fluxes[r.id] || 0) * c;
      if (Math.abs(v) < 1e-9) continue;
      net += v; who.push([r.id, v]);
    }
    who.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    return {net, who};
  }
  return {imbalance};
})();
if (typeof module !== 'undefined') module.exports = FluxBalance;

if (typeof applyFluxView !== 'undefined') (() => {
  const MIN = 0.1;                  // mmol/gDW/h; smaller imbalances are noise for this purpose
  const baseApply = applyFluxView;
  const fmt = v => Math.abs(v) >= 10 ? Math.abs(v).toFixed(1) : Math.abs(v).toFixed(2);

  // Notes live in an HTML overlay above the map (not as graph nodes), so they can never take part in
  // the map's layout, session saving or FBA bookkeeping, and stay a constant, readable size.
  let layer = null, items = [];
  function ensureLayer() {
    if (layer && layer.parentNode === cyHost) return layer;
    layer?.remove();
    layer = document.createElement('div');
    layer.id = 'fluxNotes';
    layer.setAttribute('aria-hidden', 'true');
    cyHost.style.position = 'relative';
    cyHost.append(layer);
    return layer;
  }

  function reposition() {
    if (!mapCy || mapCy.destroyed() || !layer) return;
    for (const it of items) {
      const n = mapCy.getElementById(it.host);
      const visible = n.length && n.visible() && !n.hasClass('coreCofactorHidden');
      it.el.style.display = visible ? '' : 'none';
      if (!visible) continue;
      const p = n.renderedPosition();
      it.el.style.transform = `translate(${p.x}px, ${p.y - 20}px) translate(-50%, -100%)`;
    }
  }

  function update() {
    if (window.__noFluxNotes) return;
    items = [];
    if (layer) layer.textContent = '';
    if (!flowMode || !mapCy || mapCy.destroyed() || !flowState?.coreMode) return;
    const st = flowState;
    if (!st.fluxEnabled || st.fluxResult?.status !== 'optimal') return;
    const fluxes = st.fluxResult.fluxes;
    const shown = new Set(mapCy.nodes().filter(n => n.data('reaction')).map(n => n.data('reaction')));
    const box = ensureLayer();
    mapCy.nodes('[coreKey]').forEach(n => {
      const mid = n.data('mid');
      if (!mid || !st.s.metabolites[mid]) return;
      const {net, who} = FluxBalance.imbalance(st.s, mid, shown, fluxes);
      if (Math.abs(net) < MIN) return;
      const top = who.slice(0, 2).map(x => x[0]).join('・') + (who.length > 2 ? ' ほか' : '');
      const el = document.createElement('div');
      el.className = 'fluxNote';
      el.title = `${mid}：表示していない反応との正味の出入り ${net < 0 ? '−' : '+'}${fmt(net)} mmol/gDW/h（${who.slice(0, 5).map(x => x[0] + ' ' + (x[1] > 0 ? '+' : '−') + fmt(x[1])).join('、')}）`;
      el.innerHTML = `<b>${net < 0 ? '→ 他の反応へ' : '← 他の反応から'} ${fmt(net)}</b><small>${top}</small>`;
      box.append(el);
      items.push({host: n.id(), el});
    });
    if (!mapCy._fluxNoteRender) { mapCy._fluxNoteRender = true; mapCy.on('render', reposition); }
    reposition();
  }

  applyFluxView = function (...args) {
    const out = baseApply.apply(this, args);
    try { update(); } catch (e) { console.warn('flux balance notes', e); }
    return out;
  };
})();
