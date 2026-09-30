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
  const sgn = v => (v > 0 ? '+' : '−') + fmt(v);

  // Compact chips over the map (HTML overlay, not graph nodes), one hover card, and a list in the side panel.
  let layer = null, card = null, items = [];
  function ensureLayer() {
    if (layer && layer.parentNode === cyHost) return layer;
    layer?.remove();
    layer = document.createElement('div');
    layer.id = 'fluxNotes';
    layer.setAttribute('aria-hidden', 'true');
    card = document.createElement('div');
    card.className = 'fluxCard';
    card.hidden = true;
    layer.append(card);
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
      it.el.style.transform = `translate(${p.x + 11}px, ${p.y - 26}px)`;
    }
    if (!card.hidden && card._item) placeCard(card._item);
  }

  function placeCard(it) {
    const p = mapCy.getElementById(it.host).renderedPosition(), w = layer.clientWidth, h = layer.clientHeight;
    const x = Math.min(Math.max(8, p.x + 16), Math.max(8, w - card.offsetWidth - 8));
    const y = Math.min(Math.max(8, p.y - card.offsetHeight - 10), Math.max(8, h - card.offsetHeight - 8));
    card.style.transform = `translate(${x}px, ${y}px)`;
  }

  function showCard(it) {
    const rs = new Map(flowState.s.reactions.map(r => [r.id, r]));
    const max = Math.max(...it.who.map(x => Math.abs(x[1])));
    card._item = it;
    card.innerHTML = `<div class="fc-head"><strong>${esc(it.name)}</strong><code>${esc(it.mid)}</code></div>` +
      `<p>${it.net < 0 ? '表示していない反応へ出ていく' : '表示していない反応から入る'}正味 <b>${fmt(it.net)}</b> mmol/gDW/h</p>` +
      '<ul>' + it.who.slice(0, 6).map(([id, v]) => `<li><span class="fc-id">${esc(id)}</span><span class="fc-name">${esc(rs.get(id)?.name || '')}</span>` +
        `<span class="fc-bar"><i style="width:${Math.max(4, Math.abs(v) / max * 100)}%;background:${v > 0 ? '#2f8f83' : '#c2571a'}"></i></span><span class="fc-val">${sgn(v)}</span></li>`).join('') + '</ul>' +
      `<div class="fc-foot">${it.who.length > 6 ? `ほか ${it.who.length - 6} 反応。` : ''}＋は生成、−は消費。これらの反応はマップに描いていません。</div>`;
    card.hidden = false;
    placeCard(it);
  }

  function sidePanel(list) {
    const anchor = document.getElementById('centralAdvanced') || document.getElementById('fluxControls');
    if (!anchor) return;
    let d = document.getElementById('fluxHidden');
    if (!list.length) { d?.remove(); return; }
    if (!d) {
      d = document.createElement('details');
      d.id = 'fluxHidden';
      d.className = 'fluxHidden';
      if (anchor.id === 'centralAdvanced') anchor.before(d); else anchor.append(d);
    }
    const keepOpen = d.open;
    d.innerHTML = `<summary>表示していない反応との出入り <span class="fh-count">${list.length}</span></summary>` +
      '<p class="fh-note">FBA解で、マップに描いていない反応と大きく出入りしている代謝物です。クリックで該当箇所へ移動します。</p>' +
      '<ul>' + list.map((it, i) => `<li><button type="button" data-fh="${i}"><span class="fc-arrow ${it.net < 0 ? '' : 'in'}">${it.net < 0 ? '↗' : '↙'}</span><span class="fh-name">${esc(it.name)}</span><span class="fh-val">${fmt(it.net)}</span></button></li>`).join('') + '</ul>';
    d.open = keepOpen;
    d.onclick = e => {
      const b = e.target.closest('[data-fh]');
      if (!b) return;
      const n = mapCy.getElementById(list[Number(b.dataset.fh)].host);
      if (n.length) mapCy.animate({center: {eles: n}, zoom: Math.max(mapCy.zoom(), 0.8)}, {duration: 250});
    };
  }

  function update() {
    if (window.__noFluxNotes) return;
    const st = flowState;
    const active = flowMode && mapCy && !mapCy.destroyed() && st?.coreMode && st.fluxEnabled && st.fluxResult?.status === 'optimal';
    const shown = active ? new Set(mapCy.nodes().filter(n => n.data('reaction')).map(n => n.data('reaction'))) : new Set();
    // applyFluxView runs very often (zoom, animation). Rebuild only when the solution or the drawn reactions change,
    // so a hover card stays open and nothing flickers.
    const key = active ? [st.fluxResult.fluxes ? Object.keys(st.fluxResult.fluxes).length : 0, st.s.short, [...shown].sort().join(',')].join('|') : '';
    if (active && key === update.key && items.length && layer?.parentNode === cyHost) { reposition(); return; }
    update.key = key;
    items = [];
    if (layer) { layer.textContent = ''; layer.append(card); card.hidden = true; }
    document.getElementById('fluxHidden')?.remove();
    if (!active) return;
    const fluxes = st.fluxResult.fluxes;
    const box = ensureLayer();
    mapCy.nodes('[coreKey]').forEach(n => {
      const mid = n.data('mid');
      if (!mid || !st.s.metabolites[mid]) return;
      const {net, who} = FluxBalance.imbalance(st.s, mid, shown, fluxes);
      if (Math.abs(net) < MIN) return;
      const it = {host: n.id(), mid, name: st.s.metabolites[mid].name || mid, net, who};
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'fluxChip' + (net < 0 ? '' : ' in');
      el.innerHTML = `<span class="fc-arrow">${net < 0 ? '↗' : '↙'}</span>${fmt(net)}`;
      el.setAttribute('aria-label', `${it.name}：表示していない反応との正味 ${net < 0 ? '流出' : '流入'} ${fmt(net)}`);
      el.addEventListener('mouseenter', () => showCard(it));
      el.addEventListener('focus', () => showCard(it));
      el.addEventListener('mouseleave', () => { card.hidden = true; });
      el.addEventListener('blur', () => { card.hidden = true; });
      box.append(el);
      it.el = el;
      items.push(it);
    });
    items.sort((a, b) => Math.abs(b.net) - Math.abs(a.net));
    sidePanel(items);
    if (!mapCy._fluxNoteRender) { mapCy._fluxNoteRender = true; mapCy.on('render', reposition); }
    reposition();
  }

  applyFluxView = function (...args) {
    const out = baseApply.apply(this, args);
    try { update(); } catch (e) { console.warn('flux balance notes', e); }
    return out;
  };
})();
