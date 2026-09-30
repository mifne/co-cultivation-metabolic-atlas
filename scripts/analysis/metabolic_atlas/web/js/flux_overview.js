/* "FBAの流れ": which medium components each species takes up, what it secretes, and how much
 * reaction flux each metabolic category carries — from the common pFBA solution.
 *
 * Every species is solved on its own with the same medium (the reference medium, or the medium
 * edited on the core map). Nothing here models cross-feeding between species.
 * Numbers are pFBA fluxes in mmol/gDW/h; "C-mmol" multiplies by the carbon count of the metabolite.
 */
const FluxOverview = (() => {
  const EPS = 1e-6;
  const GROUP_LABEL = {EMP: '解糖系・糖新生', PPP: 'ペントースリン酸経路', ED: 'ED経路', TCA: 'TCA関連代謝'};
  const SOLVENT = /^(h2o|h|co2|hco3)_[ecp]\d*$/;   // shown only on request: they dwarf everything else

  function carbons(formula) {
    const m = String(formula || '').match(/C(?![a-z])(\d*)/);
    return m ? Number(m[1] || 1) : 0;
  }

  function displayName(s, mid) {
    const m = s.metabolites[mid] || {};
    let name = m.name || mid;
    if (m.formula) name = name.replace(new RegExp('\\s+' + m.formula.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$'), '');
    return name.replace(/\s+mitochondria$/i, '').replace(/^(.{22}).+$/, '$1…');
  }

  /* Exchange fluxes: negative = uptake, positive = secretion (the model's sign convention). */
  function exchanges(s, fluxes, {unit = 'mmol', solvents = false, minFlux = 0, limits = null, net = true} = {}) {
    const uptake = new Map(), secretion = new Map();
    for (const r of s.reactions) {
      if (!r.exchange) continue;
      const v = fluxes[r.id] || 0;
      if (Math.abs(v) <= EPS) continue;
      const mid = Object.keys(r.stoich)[0];
      if (!solvents && SOLVENT.test(mid)) continue;
      const w = unit === 'C' ? carbons(s.metabolites[mid]?.formula) : 1;
      const amount = Math.abs(v) * w;
      if (amount <= EPS || amount < minFlux) continue;
      const pool = r.pool || mid;
      const target = v < 0 ? uptake : secretion;
      const row = target.get(pool) || {pool, mid, name: displayName(s, mid), value: 0, reactions: [], atBound: false};
      row.value += amount; row.reactions.push(r.id);
      // Uptake sitting exactly on the model's own lower bound is a limit, not a measured demand.
      const cap = limits?.[r.id] ?? (Number.isFinite(r.bounds[0]) && Math.abs(r.bounds[0]) < 1000 ? Math.abs(r.bounds[0]) : null);
      if (v < 0 && cap !== null && Math.abs(v) >= 0.999 * cap) { row.atBound = true; row.cap = cap; }
      target.set(pool, row);
    }
    // Inorganic ions taken up and released in another redox state (Fe3+ in, Fe2+ out) mostly cancel:
    // report the net electron-acceptor step instead of two large bands.
    const redox = [];
    if (net) for (const u of [...uptake.values()]) {
      const fu = s.metabolites[u.mid]?.formula;
      if (!fu || /C/.test(fu.replace(/Cl|Ca|Co|Cu|Cr|Cs|Cd/g, ''))) continue;
      for (const o of [...secretion.values()]) {
        if (o.pool === u.pool || s.metabolites[o.mid]?.formula !== fu) continue;
        const c = Math.min(u.value, o.value);
        if (c <= EPS) continue;
        redox.push({from: u.name, to: o.name, value: c});
        u.value -= c; o.value -= c;
        if (u.value <= EPS) uptake.delete(u.pool);
        if (o.value <= EPS) secretion.delete(o.pool);
      }
    }
    const sort = m => [...m.values()].sort((a, b) => b.value - a.value);
    return {uptake: sort(uptake), secretion: sort(secretion), redox};
  }

  /* Category of an internal reaction: the curated central-carbon skeleton first, then the
   * same inventory groups the overview Sankey uses. */
  function categoryOf(r, coreIds) {
    if (r.exchange) return null;
    if (coreIds.has(r.id)) return GROUP_LABEL[coreIds.get(r.id)];
    return atlasGroup(r).replace(/^注釈: /, '');
  }

  /* Sum of |flux| per category over internal reactions. This is a measure of how much reaction
   * activity a category carries, not a flow that is conserved from one category to the next. */
  function pathways(s, fluxes, coreSpec) {
    const coreIds = new Map((coreSpec?.reactions || []).map(x => [x.r.id, x.group]));
    const cats = new Map();
    for (const r of s.reactions) {
      const v = fluxes[r.id] || 0;
      if (r.exchange || Math.abs(v) <= EPS) continue;
      const c = categoryOf(r, coreIds);
      const row = cats.get(c) || {category: c, total: 0, active: 0, top: []};
      row.total += Math.abs(v); row.active++; row.top.push({id: r.id, name: r.name, flux: v, equation: r.equation});
      cats.set(c, row);
    }
    for (const row of cats.values()) row.top.sort((a, b) => Math.abs(b.flux) - Math.abs(a.flux));
    return [...cats.values()].sort((a, b) => b.total - a.total);
  }

  function growth(entry) {
    const g = entry?.objective_value;
    return {value: g, grows: entry?.status === 'optimal' && Number.isFinite(g) && g > EPS};
  }

  /* Carbon tracing result -> three conserved columns (source, entry category, fate).
   * Smaller nodes of a column are merged into 「その他」 so the diagram stays readable. */
  function carbonColumns(cf, topN = 8) {
    const rows = (cf?.flows || []).filter(f => f[3] > EPS);
    const total = rows.reduce((a, f) => a + f[3], 0);
    const keep = idx => {
      const t = new Map();
      for (const f of rows) t.set(f[idx], (t.get(f[idx]) || 0) + f[3]);
      const top = new Set([...t.entries()].sort((a, b) => b[1] - a[1]).slice(0, topN).map(x => x[0]));
      return {top, name: k => top.has(k) ? k : '\u0000other'};
    };
    const S = keep(0), C = keep(1), F = keep(2);
    const left = new Map(), right = new Map(), col = [new Map(), new Map(), new Map()];
    const add = (m, a, b, v) => { const k = a + '\u0001' + b; m.set(k, (m.get(k) || 0) + v); };
    for (const [s, c, f, v] of rows) {
      const sk = S.name(s), ck = C.name(c), fk = F.name(f);
      add(left, sk, ck, v); add(right, ck, fk, v);
      col[0].set(sk, (col[0].get(sk) || 0) + v); col[1].set(ck, (col[1].get(ck) || 0) + v); col[2].set(fk, (col[2].get(fk) || 0) + v);
    }
    const nodes = m => [...m.entries()].map(([key, value]) => ({key, value})).sort((a, b) => (a.key === '\u0000other') - (b.key === '\u0000other') || b.value - a.value);
    const links = m => [...m.entries()].map(([k, value]) => { const [a, b] = k.split('\u0001'); return {a, b, value}; });
    return {total, src: nodes(col[0]), cat: nodes(col[1]), fate: nodes(col[2]), left: links(left), right: links(right)};
  }

  return {carbons, exchanges, pathways, growth, carbonColumns, EPS};
})();
if (typeof module !== 'undefined') module.exports = FluxOverview;

/* ---------- screen (browser only) ---------- */
if (typeof document !== 'undefined' && typeof cyHost !== 'undefined') (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const host = document.createElement('div');
  host.id = 'fluxOvHost';
  host.hidden = true;
  cyHost.parentNode.append(host);
  AtlasUI.hosts = AtlasUI.hosts || {};
  AtlasUI.hosts.fluxov = host;

  const snapshot = JSON.parse(document.getElementById('fluxSnapshot')?.textContent || 'null');
  const opts = {unit: 'mmol', solvents: false, minFlux: 0.05, net: true};
  let results = null;            // {short: {status, fluxes, objective_value, uptake_limits, source}}
  let scenarioId = 'reference';  // a snapshot scenario id, or 'custom' (medium edited on the core map)
  let openCategory = null;

  function mediumInUse() { return flowState?.mediumOverride || null; }
  function scenarios() {
    const list = snapshot?.scenarios || [];
    return mediumInUse() ? [...list, {id: 'custom', label: '編集した培地（中心代謝マップ）'}] : list;
  }
  function mediumOf(id) {
    if (id === 'custom') return mediumInUse();
    const sc = snapshot?.scenarios.find(x => x.id === id);
    return sc && Object.keys(sc.added).length ? {...D.medium, ...sc.added} : null;   // null = reference medium
  }

  async function solveOnServer(id) {
    const medium = mediumOf(id);
    const out = {};
    setNote('サーバーでpFBAを計算中…（菌種ごと、数秒かかります）');
    await Promise.all(D.species.map(async s => {
      try {
        const r = await atlasCalculate('ranking', {species: s.short, medium});
        out[s.short] = {...r, source: 'サーバーで計算'};
      } catch (e) { out[s.short] = {status: 'unknown', message: String(e.message || e)}; }
    }));
    return out;
  }

  async function load(force = false) {
    if (scenarioId === 'custom' && !mediumInUse()) scenarioId = 'reference';
    const sc = snapshot?.scenarios.find(x => x.id === scenarioId);
    if (sc && !force) {
      results = {};
      for (const [short, e] of Object.entries(sc.species)) results[short] = {...e, source: 'ビルド時に計算済み（' + sc.label + '）'};
      return;
    }
    results = await solveOnServer(scenarioId);
  }

  function setNote(text) { const n = document.getElementById('fluxOvNote'); if (n) n.textContent = text; }

  const el = (name, attrs = {}, text) => {
    const e = document.createElementNS(NS, name);
    for (const [a, v] of Object.entries(attrs)) e.setAttribute(a, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const fmt = v => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
  const unitLabel = () => opts.unit === 'C' ? 'C-mmol/gDW/h' : 'mmol/gDW/h';

  let lastExchange = [];

  /* Three columns: taken up -> species -> secreted. Band width = |exchange flux|. */
  function drawExchangeSankey() {
    const W = 1000, PAD = 10, X_L = 250, X_M = 470, X_R = 740, BAR = 16, TOP = 62;
    const data = D.species.map(s => {
      const e = results?.[s.short];
      const ok = e?.status === 'optimal' && e.fluxes;
      return {s, e, ex: ok ? FluxOverview.exchanges(s, e.fluxes, {...opts, limits: e.uptake_limits}) : null};
    });
    const pools = dir => {
      const m = new Map();
      for (const d of data) for (const r of d.ex?.[dir] || []) {
        const row = m.get(r.pool) || {pool: r.pool, name: r.name, value: 0, atBound: false, cap: null};
        row.value += r.value; row.atBound = row.atBound || r.atBound; row.cap = row.cap ?? r.cap; m.set(r.pool, row);
      }
      return [...m.values()].sort((a, b) => b.value - a.value);
    };
    lastExchange = data;
    const inPools = pools('uptake'), outPools = pools('secretion');
    const spTotals = data.map(d => Math.max(d.ex ? d.ex.uptake.reduce((a, r) => a + r.value, 0) : 0,
                                              d.ex ? d.ex.secretion.reduce((a, r) => a + r.value, 0) : 0));
    const colSum = arr => arr.reduce((a, r) => a + r.value, 0);
    const H = 560;
    const rows = Math.max(inPools.length, outPools.length, D.species.length, 1);
    const k = Math.max(1e-9, (H - PAD * (Math.max(inPools.length, outPools.length) - 1)) / Math.max(colSum(inPools), colSum(outPools), 1e-9));
    const svg = el('svg', {viewBox: `0 0 ${W} ${H + TOP + 30}`, id: 'fluxOvSvg', role: 'img', 'aria-label': '培地成分の取込と分泌（FBA）'});
    const links = el('g'), nodes = el('g'), labels = el('g');
    svg.append(links, nodes, labels);
    svg.append(el('text', {x: X_L - 12, y: 16, 'text-anchor': 'end', class: 'fo-colhead'}, '取り込む成分'),
               el('text', {x: X_M + BAR / 2, y: 14, 'text-anchor': 'middle', class: 'fo-colhead'}, '菌種'),
               el('text', {x: X_R + BAR + 12, y: 16, class: 'fo-colhead'}, '分泌する成分'));

    const place = (list, x) => { let y = TOP; const pos = new Map(); for (const r of list) { const h = Math.max(r.value * k, 2); pos.set(r.pool, {y, h, used: 0}); y += h + PAD; } return pos; };
    const inPos = place(inPools, X_L), outPos = place(outPools, X_R);
    // species column: stacked, height = own throughput
    let sy = TOP; const spPos = new Map();
    data.forEach((d, i) => { const h = Math.max(spTotals[i] * k, 8); spPos.set(d.s.short, {y: sy, h, inUsed: 0, outUsed: 0}); sy += h + 70; });

    const band = (x0, y0, x1, y1, w, color, title) => {
      const mid = (x0 + x1) / 2;
      const p = el('path', {d: `M${x0},${y0} C${mid},${y0} ${mid},${y1} ${x1},${y1}`, fill: 'none', stroke: color, 'stroke-width': Math.max(1, w), 'stroke-opacity': .38, class: 'fo-link'});
      p.append(el('title', {}, title));
      return p;
    };
    for (const d of data) {
      const sp = spPos.get(d.s.short), color = colors[d.s.short];
      if (d.ex) {
        for (const r of d.ex.uptake) {
          const ip = inPos.get(r.pool); if (!ip) continue;
          const w = r.value * k;
          links.append(band(X_L, ip.y + ip.used + w / 2, X_M, sp.y + sp.inUsed + w / 2, w, color, `${r.name} → ${d.s.short}：取込 ${fmt(r.value)} ${unitLabel()}`));
          ip.used += w; sp.inUsed += w;
        }
        for (const r of d.ex.secretion) {
          const op = outPos.get(r.pool); if (!op) continue;
          const w = r.value * k;
          links.append(band(X_M + BAR, sp.y + sp.outUsed + w / 2, X_R, op.y + op.used + w / 2, w, color, `${d.s.short} → ${r.name}：分泌 ${fmt(r.value)} ${unitLabel()}`));
          op.used += w; sp.outUsed += w;
        }
      }
      nodes.append(el('rect', {x: X_M, y: sp.y, width: BAR, height: sp.h, rx: 3, fill: color, class: 'fo-click', 'data-pipe-species': d.s.short, tabindex: 0, role: 'button', 'aria-label': `${d.s.short} の中心代謝マップを開く`}));
      const g = FluxOverview.growth(d.e);
      const cy = sp.y + sp.h / 2;
      labels.append(el('text', {x: X_M + BAR / 2, y: sp.y - 22, 'text-anchor': 'middle', class: 'fo-sp', 'font-weight': 700}, d.s.short));
      labels.append(el('text', {x: X_M + BAR / 2, y: sp.y - 6, 'text-anchor': 'middle', class: 'fo-sub'},
        !d.e ? '未計算' : d.e.status !== 'optimal' ? '計算不可' : g.grows ? `成長 ${g.value.toFixed(3)} /h` : '成長なし（目的関数=0）'));
    }
    for (const r of inPools) { const p = inPos.get(r.pool); nodes.append(el('rect', {x: X_L, y: p.y, width: BAR, height: p.h, rx: 2, fill: r.atBound ? '#b4531f' : '#506b78', class: 'fo-click', 'data-pipe-pool': r.pool, tabindex: 0, role: 'button', 'aria-label': `${r.name} の代謝パイプラインを開く`}));
      const t = el('text', {x: X_L - 8, y: p.y + p.h / 2 + 4, 'text-anchor': 'end', class: 'fo-label fo-click', 'data-pipe-pool': r.pool}, `${r.name} ${fmt(r.value)}${r.atBound ? ' ▲' : ''}`);
      if (r.atBound) t.append(el('title', {}, `取込上限に達しています（上限 ${fmt(r.cap)} mmol/gDW/h）。上限は培地濃度からMonod式・在庫量で決まる値で、菌の需要ではありません。`));
      labels.append(t); }
    for (const r of outPools) { const p = outPos.get(r.pool); nodes.append(el('rect', {x: X_R, y: p.y, width: BAR, height: p.h, rx: 2, fill: '#506b78'}));
      labels.append(el('text', {x: X_R + BAR + 8, y: p.y + p.h / 2 + 4, class: 'fo-label'}, `${r.name} ${fmt(r.value)}`)); }
    const bottom = Math.max(sy, ...[...inPos.values(), ...outPos.values()].map(p => p.y + p.h)) + 30;
    svg.setAttribute('viewBox', `0 0 ${W} ${bottom}`);
    const redox = data.flatMap(d => (d.ex?.redox || []).map(r => ({...r, sp: d.s.short})));
    return {svg, redox, empty: !inPools.length && !outPools.length};
  }

  /* Carbon-tracing Sankey: taken-up carbon -> first (non-transport) reaction category -> final fate.
   * A proportional-allocation estimate on the flux network, not atom mapping. */
  let carbonSpecies = null;
  function drawCarbonSankey(short) {
    const e = results?.[short], cf = e?.carbon_flows;
    const wrap = document.createElement('div');
    wrap.className = 'fo-sankey';
    if (!cf || cf.error || !cf.flows?.length) {
      wrap.innerHTML = `<p class="fo-empty">${!e || e.status !== 'optimal' ? '流量が計算されていないため表示できません。' : cf?.error ? '炭素追跡でエラー：' + esc(cf.error) : 'この計算結果には炭素追跡が含まれていません（再ビルド、またはサーバーで再計算してください）。'}</p>`;
      return wrap;
    }
    const col = FluxOverview.carbonColumns(cf, 8);
    const W = 1000, PAD = 10, X = [250, 500, 770], BAR = 16, TOP = 34, H = 460;
    const n = Math.max(col.src.length, col.cat.length, col.fate.length);
    const k = (H - PAD * (n - 1)) / col.total;
    const label = (dict, key, fallback) => key === '\u0000other' ? 'その他' : (dict[key]?.label || fallback || key);
    const srcName = key => label(cf.sources || {}, key);
    const fateName = key => label(cf.fates || {}, key);
    const fateColor = key => key === '\u0000other' ? '#9aa8ad' : key === 'biomass' ? '#2e9e6b' : /co2/i.test(key) ? '#8a9aa2' : cf.fates?.[key]?.kind === 'secretion' ? '#c9803a' : '#9aa8ad';
    const place = list => { let y = TOP; const m = new Map(); for (const nd of list) { const h = Math.max(nd.value * k, 2); m.set(nd.key, {y, h, used: 0, used_in: 0, used_out: 0, value: nd.value}); y += h + PAD; } return m; };
    const P = [place(col.src), place(col.cat), place(col.fate)];
    const svg = el('svg', {viewBox: `0 0 ${W} ${TOP + H + 60}`, id: 'fluxOvCarbonSvg', role: 'img', 'aria-label': '取り込んだ炭素の入口と行き先（推定）'});
    const links = el('g'), nodes = el('g'), labels = el('g');
    svg.append(links, nodes, labels);
    for (const [x, t, anchor] of [[X[0] - 12, '取り込む成分', 'end'], [X[1], '最初に入る代謝カテゴリ', 'start'], [X[2], '行き先', 'start']])
      svg.append(el('text', {x, y: 16, 'text-anchor': anchor, class: 'fo-colhead'}, t));
    const pct = v => (v / col.total * 100).toFixed(v / col.total < .1 ? 1 : 0) + '%';
    const band = (x0, y0, x1, y1, w, color, title) => {
      const mid = (x0 + x1) / 2;
      const p = el('path', {d: `M${x0},${y0} C${mid},${y0} ${mid},${y1} ${x1},${y1}`, fill: 'none', stroke: color, 'stroke-width': Math.max(1, w), 'stroke-opacity': .4, class: 'fo-link'});
      p.append(el('title', {}, title)); return p;
    };
    const order = (list, dict) => [...list].sort((a, b) => (dict.get(a.a)?.y ?? 0) - (dict.get(b.a)?.y ?? 0) || (P[1].get(a.b)?.y ?? P[2].get(a.b)?.y ?? 0) - (P[1].get(b.b)?.y ?? P[2].get(b.b)?.y ?? 0));
    for (const l of order(col.left, P[0])) {
      const a = P[0].get(l.a), b = P[1].get(l.b), w = l.value * k;
      links.append(band(X[0] + BAR, a.y + a.used + w / 2, X[1], b.y + b.used_in + w / 2, w, colors[short], `${srcName(l.a)} → ${l.b === '\u0000other' ? 'その他' : l.b}：${fmt(l.value)} C-mmol/gDW/h（${pct(l.value)}）`));
      a.used += w; b.used_in += w;
    }
    for (const l of order(col.right, P[1])) {
      const a = P[1].get(l.a), b = P[2].get(l.b), w = l.value * k;
      links.append(band(X[1] + BAR, a.y + a.used_out + w / 2, X[2], b.y + b.used + w / 2, w, fateColor(l.b), `${l.a === '\u0000other' ? 'その他' : l.a} → ${fateName(l.b)}：${fmt(l.value)} C-mmol/gDW/h（${pct(l.value)}）`));
      a.used_out += w; b.used += w;
    }
    // Two-line labels (name, then amount and share). The middle column sits on top of links, so it gets a white halo.
    const tidy = t => String(t).replace(/^(.+?)\s+\1$/, '$1').replace(/^(.+?)\s+(?:[A-Z][a-z]?\d*)+$/, '$1');
    // Labels keep a 32-unit pitch per column; a displaced label gets a short leader line.
    const drawNodes = (i, list, textFn, anchor, dx, halo) => {
      let last = -Infinity;
      for (const nd of list) {
        const p = P[i].get(nd.key);
        const clickable = i === 0 && nd.key !== '\u0000other';
        nodes.append(el('rect', {x: X[i], y: p.y, width: BAR, height: p.h, rx: 2, fill: i === 2 ? fateColor(nd.key) : '#506b78', ...(clickable ? {class: 'fo-click', 'data-pipe-pool': nd.key, 'data-pipe-species': short, tabindex: 0, role: 'button'} : {})}));
        const cy = p.y + p.h / 2, ly = Math.max(cy, last + 32);
        last = ly;
        if (Math.abs(ly - cy) > 3) {
          const x0 = anchor === 'end' ? X[i] : X[i] + BAR, x1 = anchor === 'end' ? X[i] - 6 : X[i] + BAR + 6;
          labels.append(el('polyline', {points: `${x0},${cy} ${x1},${ly}`, fill: 'none', stroke: '#9fb3bb', 'stroke-width': 1}));
        }
        const cls = 'fo-label' + (halo ? ' fo-halo' : '');
        labels.append(el('text', {x: X[i] + dx, y: ly - 1, 'text-anchor': anchor, class: cls + (clickable ? ' fo-click' : ''), 'font-weight': 600, ...(clickable ? {'data-pipe-pool': nd.key, 'data-pipe-species': short} : {})}, tidy(textFn(nd.key))),
                      el('text', {x: X[i] + dx, y: ly + 13, 'text-anchor': anchor, class: 'fo-sub' + (halo ? ' fo-halo' : '')}, `${fmt(nd.value)} C-mmol/gDW/h（${pct(nd.value)}）`));
      }
    };
    drawNodes(0, col.src, srcName, 'end', -8, false);
    drawNodes(1, col.cat, key => key === '\u0000other' ? 'その他' : key, 'start', BAR + 8, true);
    drawNodes(2, col.fate, fateName, 'start', BAR + 8, false);
    const bottom = Math.max(...P.map(m => Math.max(...[...m.values()].map(p => p.y + p.h))), ...[...labels.querySelectorAll('text')].map(t => Number(t.getAttribute('y')) + 20));
    svg.setAttribute('viewBox', `0 0 ${W} ${bottom + 12}`);
    wrap.append(svg);
    const note = document.createElement('p');
    note.className = 'fo-redox';
    note.textContent = `合計 ${fmt(col.total)} C-mmol/gDW/h。元素（炭素数）収支に基づく比例配分による推定で、原子追跡ではありません。補酵素類の炭素は追跡から除いています。「最初に入る代謝カテゴリ」は輸送反応を除いた最初の反応の分類です。`;
    wrap.append(note);
    return wrap;
  }

  function carbonSection() {
    const box = document.createElement('section');
    const grows = D.species.filter(s => FluxOverview.growth(results?.[s.short]).grows);
    if (!grows.length) return box;
    if (!grows.some(s => s.short === carbonSpecies)) carbonSpecies = grows[0].short;
    const h = document.createElement('h3');
    h.className = 'fo-h3';
    h.innerHTML = '取り込んだ炭素の行き先 <small>菌種ごと。培地の炭素が、どの代謝カテゴリに入り、バイオマス・CO₂・分泌物のどれになるか（推定）</small>';
    const tabs = document.createElement('div');
    tabs.className = 'fo-tabs';
    tabs.innerHTML = grows.map(s => `<button type="button" data-carbon="${esc(s.short)}" class="${s.short === carbonSpecies ? 'on' : ''}" style="border-color:${colors[s.short]}">${esc(s.short)}</button>`).join('');
    box.append(h, tabs, drawCarbonSankey(carbonSpecies));
    return box;
  }

  /* From the FBA overview straight into the core map: connect the clicked medium component to the
   * central metabolism (its transporter/enzyme chain) and show this solution's fluxes on it, animated.
   * The solution shown on the map is the one displayed here, so it also works without the server. */
  function openPipeline(short, pool) {
    const e = results?.[short];
    const sameMap = flowState?.coreMode && flowState.s.short === short && mapCy && !mapCy.destroyed();
    if (sameMap) flowTab.onclick(); else { tab('map'); openFlow(short, 'glc__D_e'); }
    const st = flowState;
    st.mediumOverride = mediumOf(scenarioId);
    let added = 0;
    if (pool && !st.mediumRoots.has(pool)) added = addMediumRoots(st, [pool]);
    if (e?.status === 'optimal' && e.fluxes) {
      st.fluxResult = {...e, conditions: {source: e.source, scenario: scenarioId}};
      st.commonRankings = new Map([[short + ':' + JSON.stringify(st.mediumOverride || null), Promise.resolve(st.fluxResult)]]);
      st.fluxEnabled = true;
      const box = $('fluxEnabled'); if (box) box.checked = true;
      const status = $('fluxStatus'); if (status) status.textContent = 'FBAの流れタブの解を表示中：矢印＝正味方向、粒子の速さ＝流量の大小。灰色＝ゼロ流量。';
      fluxDirty = true;
      applyFluxView();
      refreshMediumSummary();
    }
    const entry = st.coreConnections.find(x => x.pool === pool);
    const ids = new Set(entry?.chain || []);
    if (entry?.target) ids.add('m_' + entry.target);
    const focus = mapCy.nodes().filter(n => ids.has(n.id()));
    if (focus.length) {
      mapCy.fit(focus.union(focus.neighborhood('node')), 90);
      if (mapCy.zoom() > 1.1) { const bb = focus.boundingBox(); mapCy.zoom({level: 1.1, position: {x: (bb.x1 + bb.x2) / 2, y: (bb.y1 + bb.y2) / 2}}); mapCy.center(focus); }
    }
    reportProblem(pool
      ? (entry?.status === 'connected' || entry?.target
          ? `${short}：${pool} の代謝パイプライン（${(entry.reactions || []).length}反応）を中心代謝${entry.target ? ' の ' + entry.target : ''} へ接続して表示中。${entry.reason || ''}`
          : `${short}：${pool} — ${entry?.reason || '中心代謝への接続は見つかりませんでした。'}`)
      : `${short} の中心代謝マップ（FBAの流れタブの解を重ねて表示）`, {kind: 'info', key: 'pipeline'});
  }

  function pipelineChooser(pool) {
    const bar = document.getElementById('foPipeBar');
    if (!bar) return;
    const who = lastExchange.filter(d => d.ex?.uptake.some(r => r.pool === pool));
    if (who.length === 1) { openPipeline(who[0].s.short, pool); return; }
    const label = who[0]?.ex.uptake.find(r => r.pool === pool)?.name || pool;
    bar.innerHTML = `<strong>${esc(label)}</strong> を取り込む菌種：` + who.map(d => `<button type="button" data-pipe-species="${esc(d.s.short)}" data-pipe-pool="${esc(pool)}" style="border-color:${colors[d.s.short]}">${esc(d.s.short)} の中心代謝マップで開く</button>`).join(' ');
  }

  function pathwayPanel() {
    const wrap = document.createElement('div');
    wrap.className = 'fo-paths';
    for (const s of D.species) {
      const e = results?.[s.short], g = FluxOverview.growth(e);
      const box = document.createElement('section');
      box.style.borderTopColor = colors[s.short];
      box.innerHTML = `<h3 style="color:${colors[s.short]}">${esc(s.short)}</h3>`;
      if (!e) { box.insertAdjacentHTML('beforeend', '<p class="fo-empty">未計算</p>'); wrap.append(box); continue; }
      if (e.status !== 'optimal') { box.insertAdjacentHTML('beforeend', `<p class="fo-empty">${esc(e.message || '計算できませんでした')}</p>`); wrap.append(box); continue; }
      if (!g.grows) { box.insertAdjacentHTML('beforeend', '<p class="fo-empty">この培地では成長できず（目的関数=0）、流量はすべて0です。' + (scenarioId !== 'reference' ? '' : '参照培地にはPfが使える炭素源（グルコース・乳酸・プロピオン酸）がなく、Pfの酸素交換も閉じています。') + '中心代謝マップで炭素源などを加えて「サーバーで再計算」してください（例：乳酸10 mMで成長）。</p>'); wrap.append(box); continue; }
      const cats = FluxOverview.pathways(s, e.fluxes, CoreMetabolism.select(s));
      const max = cats[0]?.total || 1;
      box.insertAdjacentHTML('beforeend', `<p class="fo-sub2">成長 ${g.value.toFixed(3)} /h · 活性のある内部反応 ${cats.reduce((a, c) => a + c.active, 0)}</p>`);
      const list = document.createElement('div');
      for (const c of cats) {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'fo-row';
        row.dataset.sp = s.short; row.dataset.cat = c.category;
        row.innerHTML = `<span class="fo-name">${esc(c.category)}</span><span class="fo-bar"><i style="width:${Math.max(2, c.total / max * 100)}%;background:${colors[s.short]}"></i></span><span class="fo-val">${fmt(c.total)}<small> · ${c.active}反応</small></span>`;
        list.append(row);
        if (openCategory && openCategory.sp === s.short && openCategory.cat === c.category) {
          const t = document.createElement('table');
          t.className = 'fo-top';
          t.innerHTML = '<thead><tr><th>反応</th><th>名称</th><th>流量</th></tr></thead><tbody>' +
            c.top.slice(0, 12).map(x => `<tr data-sp="${esc(s.short)}" data-id="${esc(x.id)}"><td>${esc(x.id)}</td><td>${esc(x.name || x.equation)}</td><td>${x.flux > 0 ? '' : '−'}${fmt(Math.abs(x.flux))}</td></tr>`).join('') + '</tbody>';
          list.append(t);
        }
      }
      box.append(list);
      wrap.append(box);
    }
    return wrap;
  }

  /* Growth of every species under every prepared medium: the quickest way to see who can use what. */
  function compareTable() {
    const list = scenarios();
    if (list.length < 2) return '';
    const cell = (sc, short) => {
      const e = sc.species?.[short];
      if (!e) return '<td class="fo-na">—</td>';
      const g = FluxOverview.growth(e);
      return g.grows ? `<td class="fo-g">${g.value.toFixed(3)}</td>` : '<td class="fo-g0">成長なし</td>';
    };
    return '<table class="fo-compare"><caption>培地ごとの成長速度（/h）　行をクリックで表示を切り替え</caption><thead><tr><th>培地</th>' +
      D.species.map(s => `<th style="color:${colors[s.short]}">${esc(s.short)}</th>`).join('') + '</tr></thead><tbody>' +
      list.map(sc => `<tr data-scenario="${esc(sc.id)}"${sc.id === scenarioId ? ' class="on"' : ''}><td>${esc(sc.label)}</td>${D.species.map(s => cell(sc, s.short)).join('')}</tr>`).join('') + '</tbody></table>';
  }

  function render() {
    host.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'fo-head';
    const cur = scenarios().find(x => x.id === scenarioId);
    head.innerHTML = '<p><strong>培地成分 → 各菌種 → 代謝の流れ（FBA）</strong>　共通のpFBA解（GEMの目的関数を最大化し、総絶対流量を最小化）。' +
      '<em>各菌種を単独で、同じ培地条件で解いた結果</em>で、菌種間の分泌物の授受は含みません。流量ゼロは他の最適解でもゼロとは限りません。</p>' +
      compareTable() +
      `<p class="fo-cond">表示中の培地条件：<strong>${esc(cur?.label || '')}</strong>　${esc(Object.values(results || {}).find(v => v?.source)?.source || '')}</p>` +
      '<div class="fo-controls">' +
      `<label>単位 <select id="foUnit"><option value="mmol"${opts.unit === 'mmol' ? ' selected' : ''}>mmol/gDW/h</option><option value="C"${opts.unit === 'C' ? ' selected' : ''}>C-mmol/gDW/h（炭素換算）</option></select></label>` +
      `<label>最小流量 <select id="foMin">${[0.01, 0.05, 0.1, 0.5, 1].map(v => `<option value="${v}"${opts.minFlux === v ? ' selected' : ''}>${v}</option>`).join('')}</select></label>` +
      `<label><input type="checkbox" id="foSolv"${opts.solvents ? ' checked' : ''}> 水・H⁺・CO₂も表示</label>` +
      `<label><input type="checkbox" id="foNet"${opts.net ? ' checked' : ''}> 同じ元素の取込と分泌（Fe³⁺→Fe²⁺）は相殺</label>` +
      '<button type="button" id="foRecalc">この培地をサーバーで再計算</button><span id="fluxOvNote" role="status"></span></div>' +
      '<p class="fo-legend"><span class="fo-dot" style="background:#506b78"></span>成分　<span class="fo-dot" style="background:#b4531f"></span>▲ 取込上限に達している成分（需要ではなく上限で決まった値）</p>';
    host.append(head);
    const sk = drawExchangeSankey();
    const skWrap = document.createElement('div');
    skWrap.className = 'fo-sankey';
    if (sk.empty) skWrap.innerHTML = '<p class="fo-empty">表示できる交換流量がありません。閾値を下げるか、培地条件を確認してください。</p>';
    skWrap.append(sk.svg);
    host.append(skWrap);
    const pb = document.createElement('div');
    pb.id = 'foPipeBar';
    pb.className = 'fo-pipe';
    pb.innerHTML = '取り込む成分（左の帯）や菌種をクリックすると、その成分の<strong>代謝パイプライン</strong>（輸送・酵素反応）を中心代謝マップで開き、このFBA解の流量をアニメーション表示します。';
    host.append(pb);
    if (sk.redox.length) {
      const rn = document.createElement('p');
      rn.className = 'fo-redox';
      rn.innerHTML = '相殺した往復：' + sk.redox.map(r => `${esc(r.sp)}：${esc(r.from)}の取込と${esc(r.to)}の分泌 ${fmt(r.value)}（酸化還元による電子の受け渡しで、正味の物質収支はほぼ0）`).join('；');
      host.append(rn);
    }
    host.append(carbonSection());
    const h2 = document.createElement('h3');
    h2.className = 'fo-h3';
    h2.innerHTML = '代謝カテゴリごとの反応流量 <small>Σ|v|（mmol/gDW/h）。カテゴリ間で保存される「流れ」ではなく、そこを通る反応活性の目安です。行をクリックで上位反応。</small>';
    host.append(h2, pathwayPanel());
    const note = document.createElement('p');
    note.className = 'fo-foot';
    note.innerHTML = '<strong>読み方の注意：</strong>参照培地の炭素源は少量のアミノ酸・ヌクレオシドだけで、酸素の取込が上限に張り付いています。そのため、余った還元力を捨てる経路（Fe³⁺→Fe²⁺の還元、CO・バリンの分泌、窒素の放出など）が最適解に入り、大きな交換流量として現れます。' +
      'OR16での感度計算では、Fe²⁺・CO・バリンの分泌を禁じると成長が4〜33%下がりました。数値は元素収支が閉じた最適解ですが、生理的な予測ではなく、このモデルと取込上限ルール（Monod式×在庫量）の下での一つの解です。▲は上限で決まった取込です。';
    host.append(note);
  }

  host.addEventListener('change', e => {
    if (e.target.id === 'foUnit') opts.unit = e.target.value;
    else if (e.target.id === 'foMin') opts.minFlux = Number(e.target.value);
    else if (e.target.id === 'foSolv') opts.solvents = e.target.checked;
    else if (e.target.id === 'foNet') opts.net = e.target.checked;
    else return;
    render();
  });
  host.addEventListener('click', async e => {
    if (e.target.id === 'foRecalc') { await load(true); render(); return; }
    const pipe = e.target.closest('[data-pipe-pool],[data-pipe-species]');
    if (pipe) {
      const sp = pipe.dataset.pipeSpecies, pool = pipe.dataset.pipePool;
      if (sp && pool) openPipeline(sp, pool);
      else if (pool) pipelineChooser(pool);
      else openPipeline(sp, null);
      return;
    }
    const cs = e.target.closest('[data-carbon]');
    if (cs) { carbonSpecies = cs.dataset.carbon; render(); return; }
    const sc = e.target.closest('tr[data-scenario]');
    if (sc) { scenarioId = sc.dataset.scenario; openCategory = null; await load(); render(); return; }
    const row = e.target.closest('.fo-row');
    if (row) {
      const same = openCategory && openCategory.sp === row.dataset.sp && openCategory.cat === row.dataset.cat;
      openCategory = same ? null : {sp: row.dataset.sp, cat: row.dataset.cat};
      render(); return;
    }
    const tr = e.target.closest('tr[data-id]');
    if (tr) { tab('search'); $('species').value = tr.dataset.sp; setSearchGroup(null); $('query').value = tr.dataset.id; renderResults(); showReaction(tr.dataset.sp, tr.dataset.id); }
  });

  async function enter() {
    if (mediumInUse() && scenarioId === 'reference' && !window.__foSeen) { scenarioId = 'custom'; }
    window.__foSeen = true;
    flowCanvasSuspended = true;
    flowMode = false;
    setLayer('fluxov');
    setView('map');
    host.innerHTML = '<p class="fo-empty" style="padding:24px">読み込み中…</p>';
    await load();
    render();
  }

  window.openFluxScenario = id => { scenarioId = id; tab('map'); enter(); };

  const tabBtn = document.createElement('button');
  tabBtn.id = 'fluxTab';
  tabBtn.textContent = 'FBAの流れ';
  $('mapTab').after(tabBtn);
  tabBtn.onclick = () => { tab('map'); enter(); };

  const baseSetLayer = setLayer;
  setLayer = function (layer) {
    baseSetLayer(layer);
  };
  const baseExport = $('export').onclick;
  $('export').onclick = () => {
    if (AtlasUI.layer !== 'fluxov') return baseExport();
    const svg = $('fluxOvSvg')?.cloneNode(true);
    if (!svg) return;
    svg.setAttribute('xmlns', NS);
    svg.insertAdjacentHTML('afterbegin', '<style>text{font-family:sans-serif;fill:#243f49}.fo-sub{font-size:12px;fill:#657980}.fo-label{font-size:12px}.fo-colhead{font-size:12px;fill:#657980}.fo-sp{font-size:15px}</style><rect width="100%" height="100%" fill="white"/>');
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type: 'image/svg+xml;charset=utf-8'}));
    const a = document.createElement('a'); a.href = url; a.download = 'fba_exchange_sankey.svg'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
})();
