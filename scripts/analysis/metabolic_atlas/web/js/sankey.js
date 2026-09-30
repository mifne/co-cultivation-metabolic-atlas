/* Lightweight overview: a Sankey diagram of species -> reaction categories.
 * Pure inline SVG built from the exported model (no Cytoscape graph, no per-reaction nodes), so the
 * "全体マップ" tab opens instantly. Band width = number of reactions registered in the model; it is
 * an inventory, not a flux. Detailed networks are opened on demand from it.
 */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1000, PAD = 18, LEFT = 210, RIGHT = 700, BAR = 16, TOP = 24;
  const host = document.createElement('div');
  host.id = 'sankeyHost';
  host.hidden = true;
  cyHost.parentNode.append(host);
  AtlasUI.hosts = AtlasUI.hosts || {};
  AtlasUI.hosts.sankey = host;

  const OTHER = 'その他・未分類';
  function counts() {
    const bySpecies = D.species.map(s => {
      const m = new Map();
      for (const r of s.reactions) {
        const g = atlasGroup(r);
        m.set(g, (m.get(g) || 0) + 1);
      }
      return {short: s.short, total: s.reactions.length, groups: m};
    });
    const totals = new Map();
    for (const s of bySpecies) for (const [g, n] of s.groups) totals.set(g, (totals.get(g) || 0) + n);
    // Largest first, the catch-all last so the meaningful categories read from the top.
    const order = [...totals.keys()].sort((a, b) => a === OTHER ? 1 : b === OTHER ? -1 : totals.get(b) - totals.get(a));
    return {bySpecies, totals, order};
  }

  function layout(model, height) {
    const {bySpecies, totals, order} = model;
    const grand = bySpecies.reduce((a, s) => a + s.total, 0);
    const k = (height - PAD * (order.length - 1)) / grand;
    const species = [], groups = [], links = [];
    let y = TOP;
    // Species column is centred against the taller category column.
    const speciesHeight = grand * k + PAD * (bySpecies.length - 1);
    let sy = TOP + ((height + PAD * 0) - speciesHeight) / 2;
    for (const s of bySpecies) { species.push({...s, y: sy, h: s.total * k, used: 0}); sy += s.total * k + PAD; }
    for (const g of order) { groups.push({name: g, total: totals.get(g), y, h: totals.get(g) * k, used: 0}); y += totals.get(g) * k + PAD; }
    for (const g of groups) for (const s of species) {
      const n = s.groups.get(g.name); if (!n) continue;
      const h = n * k;
      links.push({s, g, n, h, sy: s.y + s.used, gy: g.y + g.used}); s.used += h; g.used += h;
    }
    return {species, groups, links, height: y - PAD + TOP};
  }

  function draw() {
    const lay = layout(counts(), 520);
    const el = (name, attrs, text) => {
      const e = document.createElementNS(NS, name);
      for (const [a, v] of Object.entries(attrs)) e.setAttribute(a, v);
      if (text !== undefined) e.textContent = text;
      return e;
    };
    const svg = el('svg', {viewBox: `0 0 ${W} ${lay.height + 20}`, role: 'img', id: 'sankeySvg',
      'aria-label': '菌種ごとの反応分類の内訳（反応数）'});
    const linkLayer = el('g', {}), nodeLayer = el('g', {}), labelLayer = el('g', {});
    svg.append(linkLayer, nodeLayer, labelLayer);
    const x0 = LEFT + BAR, x1 = RIGHT, mid = (x0 + x1) / 2;
    for (const l of lay.links) {
      const y0 = l.sy + l.h / 2, y1 = l.gy + l.h / 2;
      const p = el('path', {d: `M${x0},${y0} C${mid},${y0} ${mid},${y1} ${x1},${y1}`, fill: 'none',
        stroke: colors[l.s.short], 'stroke-width': Math.max(1, l.h), 'stroke-opacity': .34, class: 'sk-link',
        'data-species': l.s.short, 'data-group': l.g.name, tabindex: 0, role: 'button',
        'aria-label': `${l.s.short} ${l.g.name} ${l.n}反応を開く`});
      p.append(el('title', {}, `${names[l.s.short]} → ${l.g.name}：${l.n.toLocaleString()} 反応（クリックで反応一覧を開く）`));
      linkLayer.append(p);
    }
    for (const s of lay.species) {
      nodeLayer.append(el('rect', {x: LEFT, y: s.y, width: BAR, height: s.h, rx: 3, fill: colors[s.short], class: 'sk-node',
        'data-species-node': s.short, tabindex: 0, role: 'button', 'aria-label': `${names[s.short]} の詳細`}));
      const t = el('text', {x: LEFT - 12, y: s.y + s.h / 2 - 2, 'text-anchor': 'end', class: 'sk-label', 'font-weight': 700}, s.short);
      const t2 = el('text', {x: LEFT - 12, y: s.y + s.h / 2 + 15, 'text-anchor': 'end', class: 'sk-sub'}, `${s.total.toLocaleString()} 反応`);
      labelLayer.append(t, t2);
    }
    // Labels keep a minimum pitch; thin bands get a short leader line to the nudged label.
    let lastLabelY = -Infinity;
    for (const g of lay.groups) {
      const barH = Math.max(g.h, 2), cy = g.y + barH / 2;
      nodeLayer.append(el('rect', {x: RIGHT, y: g.y, width: BAR, height: barH, rx: 3, fill: '#506b78', class: 'sk-node',
        'data-group-node': g.name, tabindex: 0, role: 'button', 'aria-label': `${g.name} の内訳`}));
      const ly = Math.max(cy, lastLabelY + 32);
      lastLabelY = ly;
      if (Math.abs(ly - cy) > 3)
        labelLayer.append(el('polyline', {points: `${RIGHT + BAR},${cy} ${RIGHT + BAR + 12},${ly} ${RIGHT + BAR + 20},${ly}`, fill: 'none', stroke: '#9fb3bb', 'stroke-width': 1}));
      const x = RIGHT + BAR + (Math.abs(ly - cy) > 3 ? 26 : 10);
      labelLayer.append(el('text', {x, y: ly - 1, class: 'sk-label'}, g.name.replace(/^注釈: /, '')),
        el('text', {x, y: ly + 14, class: 'sk-sub'}, g.total.toLocaleString() + ' 反応'));
    }
    svg.setAttribute('viewBox', `0 0 ${W} ${Math.max(lay.height, lastLabelY + 24) + 10}`);
    return svg;
  }

  const pools = [['lac__L_e', 'L-乳酸'], ['ppa_e', 'プロピオン酸'], ['ac_e', '酢酸'], ['o2_e', '酸素'], ['nh4_e', 'アンモニウム'],
    ['C30_oligo_e', 'C30オリゴマー'], ['odtd_e', 'ODTD'], ['b12_e', 'B12'], ['glc__D_e', 'グルコース']];

  function render() {
    host.innerHTML = '';
    const intro = document.createElement('div');
    intro.className = 'sk-intro';
    intro.innerHTML = '<p><strong>モデルの全体像（反応数の内訳）</strong>　帯の太さはモデルに登録された<em>反応の数</em>です。物質やフラックスの流れではありません。' +
      '「その他・未分類」は名称・注釈から分類できなかった反応で、生化学的な意味を持つ分類ではありません。</p>' +
      '<div class="sk-actions"><span>詳しく見る：</span>' +
      D.species.map(s => `<button type="button" data-core="${s.short}" style="border-color:${colors[s.short]}">${s.short} の中心代謝マップ</button>`).join('') +
      '</div>';
    host.append(intro, draw());
    const poolBox = document.createElement('div');
    poolBox.className = 'sk-pools';
    poolBox.innerHTML = '<span>共有培地の成分：</span>' + pools.map(([id, label]) => `<button type="button" data-pool="${esc(id)}">${esc(label)}</button>`).join('');
    const detail = document.createElement('div');
    detail.id = 'sankeyDetail';
    detail.setAttribute('role', 'status');
    detail.textContent = '帯（菌種→分類）をクリックすると、その分類の反応一覧を「反応を調べる」で開きます。左の菌種、右の分類をクリックすると内訳を表示します。';
    host.append(poolBox, detail);
  }

  function leaveToCy() {
    flowCanvasSuspended = false;
    setLayer('cy');
  }

  function showSpecies(short) {
    const s = D.species.find(x => x.short === short), m = atlasGroups.get(short);
    $('sankeyDetail').innerHTML = `<strong>${esc(names[short])}</strong>　${s.reactions.length.toLocaleString()} 反応 / ${Object.keys(s.metabolites).length.toLocaleString()} 代謝物<br>` +
      [...m].sort((a, b) => b[1] - a[1]).map(([g, n]) => `<button type="button" data-open="${esc(short)}|${esc(g)}">${esc(g.replace(/^注釈: /, ''))} ${n}</button>`).join(' ');
  }
  function showGroup(g) {
    $('sankeyDetail').innerHTML = `<strong>${esc(g.replace(/^注釈: /, ''))}</strong>　` +
      D.species.map(s => { const n = atlasGroups.get(s.short).get(g)?.length || 0; return n ? `<button type="button" data-open="${esc(s.short)}|${esc(g)}" style="border-color:${colors[s.short]}">${s.short} ${n.toLocaleString()} 反応を開く</button>` : ''; }).join(' ');
  }
  // Category drill-down: the reaction list of that species and category (light), not the whole-cell graph.
  function openGroup(short, g) {
    $('species').value = short;
    $('query').value = '';
    $('kind').value = 'all';
    setSearchGroup(g);
    tab('search');
    renderResults();
  }

  host.addEventListener('click', e => {
    const t = e.target;
    const open = t.closest('[data-open]');
    if (open) { const [short, g] = open.dataset.open.split('|'); openGroup(short, g); return; }
    const core = t.closest('[data-core]');
    if (core) { leaveToCy(); openFlow(core.dataset.core, 'glc__D_e'); return; }
    const pool = t.closest('[data-pool]');
    if (pool) { showPoolInSankey(pool.dataset.pool); return; }
    const link = t.closest('.sk-link');
    if (link) { openGroup(link.dataset.species, link.dataset.group); return; }
    const sn = t.closest('[data-species-node]');
    if (sn) { showSpecies(sn.dataset.speciesNode); return; }
    const gn = t.closest('[data-group-node]');
    if (gn) showGroup(gn.dataset.groupNode);
  });
  host.addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.closest?.('[tabindex]')) { e.preventDefault(); e.target.dispatchEvent(new MouseEvent('click', {bubbles: true})); }
  });

  function showPoolInSankey(id) {
    const label = pools.find(p => p[0] === id)?.[1] || id;
    let html = `<strong>${esc(label)}</strong> <code>${esc(id)}</code>　モデルの交換上下限（培養中の実際の条件ではありません）<br>`;
    for (const sp of D.species) {
      const rs = sp.reactions.filter(r => r.pool === id);
      html += `<span style="color:${colors[sp.short]};font-weight:700">${sp.short}</span> ` +
        (rs.length ? rs.map(r => `${esc(r.id)} [${r.bounds[0]}, ${r.bounds[1]}]`).join(', ') : '交換反応なし') + '　';
    }
    $('sankeyDetail').innerHTML = html;
  }

  function enter() {
    flowCanvasSuspended = true;
    flowMode = false;          // the flow (core) map is not the visible view any more
    if (!host.firstChild) render();
    setLayer('sankey');
    setView('map');            // refresh tab highlighting now that flowMode is off
  }
  window.showSankey = () => { tab('map'); enter(); };
  // Any other map view resets the header text it had set itself.
  const baseSetLayer = setLayer;
  setLayer = function (layer) {
    baseSetLayer(layer);
    const wide = ['sankey', 'fluxov', 'home'].includes(layer);
    host.hidden = layer !== 'sankey';
    controls.style.display = wide ? 'none' : '';
    $('context').style.display = wide ? 'none' : '';
  };

  $('mapTab').onclick = () => { tab('map'); enter(); };
  const baseExport = $('export').onclick;
  $('export').onclick = () => {
    if (AtlasUI.layer !== 'sankey') return baseExport();
    const svg = $('sankeySvg').cloneNode(true);
    svg.setAttribute('xmlns', NS);
    svg.insertAdjacentHTML('afterbegin', '<style>text{font-family:sans-serif;fill:#243f49}.sk-sub{font-size:12px;fill:#657980}.sk-label{font-size:14px}</style><rect width="100%" height="100%" fill="white"/>');
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type: 'image/svg+xml;charset=utf-8'}));
    const a = document.createElement('a'); a.href = url; a.download = 'model_overview_sankey.svg'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
})();
