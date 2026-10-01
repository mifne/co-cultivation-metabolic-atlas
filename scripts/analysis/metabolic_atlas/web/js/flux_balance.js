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
  /* Everything the FBA solution does with one metabolite: producing and consuming reactions, largest first. */
  function turnover(species, mid, fluxes, shown) {
    const producers = [], consumers = [];
    for (const r of species.reactions) {
      const c = r.stoich[mid];
      if (c === undefined) continue;
      const v = (fluxes[r.id] || 0) * c;
      if (Math.abs(v) < 1e-9) continue;
      (v > 0 ? producers : consumers).push({id: r.id, name: r.name, exchange: !!r.exchange, drawn: shown ? shown.has(r.id) : null, value: Math.abs(v)});
    }
    producers.sort((a, b) => b.value - a.value); consumers.sort((a, b) => b.value - a.value);
    const sum = l => l.reduce((x, r) => x + r.value, 0);
    return {producers, consumers, produced: sum(producers), consumed: sum(consumers)};
  }
  return {imbalance, turnover};
})();
if (typeof module !== 'undefined') module.exports = FluxBalance;

if (typeof showCompoundStructure !== 'undefined') (() => {
  /* Hover on a metabolite: show what the FBA solution does with it (default) or its structure (toggle).
   * The preview is a fixed, click-through box, so the switch lives in the side panel. */
  const KEY = 'metabolic-atlas-hover-structure';
  const fmt = v => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
  let structureMode = false;
  try { structureMode = localStorage.getItem(KEY) === '1'; } catch { /* storage unavailable */ }

  function fluxReady() {
    const st = flowState;
    return !!(st && st.fluxEnabled && st.fluxResult?.status === 'optimal');
  }

  function row(it, color, max) {
    const tag = it.exchange ? '<em>交換</em>' : it.drawn === false ? '<em class="hid">非表示</em>' : '';
    return `<li><span class="hv-id">${esc(it.id)}</span><span class="hv-name">${esc(it.name || '')}</span>${tag}` +
      `<span class="hv-bar"><i style="width:${Math.max(3, it.value / max * 100)}%;background:${color}"></i></span><span class="hv-val">${fmt(it.value)}</span></li>`;
  }

  function fbaPanel(n) {
    const st = flowState, mid = n.data('mid'), m = st.s.metabolites[mid];
    const shown = new Set(mapCy.nodes().filter(x => x.data('reaction')).map(x => x.data('reaction')));
    const t = FluxBalance.turnover(st.s, mid, st.fluxResult.fluxes, shown);
    const max = Math.max(t.produced, t.consumed, 1e-9);
    const list = (title, arr, color) => `<h4>${title}<span>${fmt(arr.reduce((a, r) => a + r.value, 0))}</span></h4>` +
      (arr.length ? `<ul>${arr.slice(0, 5).map(r => row(r, color, arr[0].value)).join('')}</ul>${arr.length > 5 ? `<p class="hv-more">ほか ${arr.length - 5} 反応</p>` : ''}` : '<p class="hv-more">なし</p>');
    structurePreview.innerHTML = `<div class="hv-head"><strong>${esc(m?.name || mid)}</strong><code>${esc(mid)}</code></div>` +
      (t.produced + t.consumed < 1e-9 ? '<p class="hv-none">この解では流量がありません（ゼロ）。</p>' :
        `<div class="hv-turn">回転量 <b>${fmt(Math.max(t.produced, t.consumed))}</b> <small>mmol/gDW/h</small></div>` +
        list('生成', t.producers, '#2f8f83') + list('消費', t.consumers, '#c2571a')) +
      '<div class="hv-foot">共通pFBAの解 · 「非表示」はマップに描いていない反応 · 構造式はサイドパネルのスイッチで切替</div>';
    structurePreview.classList.add('hv-fba');
    structurePreview.hidden = false;
  }

  // The preview stays open while the pointer is on it (it used to be click-through and vanished
  // as soon as the cursor left the node), so its contents can be read and scrolled.
  structurePreview.addEventListener('mouseenter', () => { clearTimeout(toolsTimer); toolsTimer = null; clearTimeout(structureTimer); });
  structurePreview.addEventListener('mouseleave', () => { clearTimeout(toolsTimer); toolsTimer = setTimeout(hideCompoundTools, 500); });

  // Wheel over the panel scrolls the panel only; it must never reach the page or the map (which would close it).
  structurePreview.addEventListener('wheel', e => {
    e.stopPropagation();
    const el = structurePreview;
    const canScroll = el.scrollHeight > el.clientHeight + 1;
    if (!canScroll || (e.deltaY < 0 && el.scrollTop <= 0) || (e.deltaY > 0 && el.scrollTop + el.clientHeight >= el.scrollHeight - 1)) e.preventDefault();
  }, {passive: false});

  // Placement: our panel grows to its content (scrolls when taller than the window) and sits close to the node,
  // so the pointer can travel from the node onto it. The wavy omission marks are shown only when zoomed in
  // and scale with the zoom.
  const MIN_ZOOM_MARKS = 0.8;
  const placeBase = positionCompoundTools;
  positionCompoundTools = function (...args) {
    placeBase.apply(this, args);
    if (!hoveredCompound || !mapCy || mapCy.destroyed()) return;
    const z = mapCy.zoom(), p = hoveredCompound.renderedPosition(), r = cyHost.getBoundingClientRect();
    const x = r.left + p.x, y = r.top + p.y;
    const k = Math.min(1.5, Math.max(0.8, z));
    compoundTools.style.visibility = z >= MIN_ZOOM_MARKS ? '' : 'hidden';
    compoundTools.style.transform = `translate(-50%, -50%) scale(${k.toFixed(2)})`;
    if (!structurePreview.classList.contains('hv-fba') || structurePreview.hidden) return;
    const gap = 14 + 14 * k, w = Math.min(330, innerWidth - 24);
    structurePreview.style.width = w + 'px';
    structurePreview.style.height = 'auto';
    structurePreview.style.maxHeight = (innerHeight - 24) + 'px';
    const h = Math.min(structurePreview.scrollHeight + 2, innerHeight - 24);
    const above = y - gap - 12, below = innerHeight - y - gap - 12;
    const top = above >= h || above >= below ? Math.max(12, y - gap - h) : Math.min(innerHeight - h - 12, y + gap);
    structurePreview.style.height = h + 'px';
    structurePreview.style.top = top + 'px';
    structurePreview.style.bottom = 'auto';
    structurePreview.style.left = Math.max(12, Math.min(innerWidth - w - 12, x - w / 2)) + 'px';
  };

  const base = showCompoundStructure;
  showCompoundStructure = async function (n, token) {
    if (!structureMode && fluxReady() && token === structureToken) {
      try { fbaPanel(n); if (typeof positionCompoundTools === 'function') positionCompoundTools(); return; } catch (e) { console.warn('hover FBA panel', e); }
    }
    structurePreview.classList.remove('hv-fba');
    return base(n, token);
  };

  function installSwitch() {
    if (document.getElementById('hoverStructureSwitch')) return;
    const anchor = document.querySelector('.coreToggle');
    if (!anchor) return;
    const lab = document.createElement('label');
    lab.className = 'coreToggle switch';
    lab.id = 'hoverStructureSwitch';
    lab.innerHTML = '<input type="checkbox" role="switch"><span class="sw-track"><i></i></span><span class="sw-text">ノードのホバーで構造式を表示<small>オフ：その代謝物のFBAデータを表示</small></span>';
    const box = lab.querySelector('input');
    box.checked = structureMode;
    box.onchange = () => { structureMode = box.checked; try { localStorage.setItem(KEY, structureMode ? '1' : '0'); } catch { /* ignore */ } structurePreview.hidden = true; };
    anchor.after(lab);
  }
  // The central controls are rebuilt on species change and session restore: (re)attach the switch whenever they appear.
  new MutationObserver(installSwitch).observe($('context'), {childList: true, subtree: true});
  installSwitch();
})();
