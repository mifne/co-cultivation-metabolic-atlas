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

  return {carbons, exchanges, pathways, growth, EPS};
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
  let results = null;            // {short: {status, fluxes, objective_value, source}}
  let openCategory = null;

  function mediumInUse() { return flowState?.mediumOverride || null; }

  async function load(force = false) {
    const custom = mediumInUse();
    const stale = !results || results.__custom !== !!custom;
    if (!force && !stale) return;
    if (!custom && snapshot && !force) {
      results = {__custom: false};
      for (const [short, e] of Object.entries(snapshot.species)) results[short] = {...e, source: 'ビルド時のpFBA（参照培地）'};
      return;
    }
    results = {__custom: !!custom};
    setNote('サーバーでpFBAを計算中…（菌種ごと、数秒かかります）');
    await Promise.all(D.species.map(async s => {
      try {
        const r = await loadCommonRanking(s.short);
        results[s.short] = {...r, source: custom ? 'サーバーで計算（編集した培地）' : 'サーバーで再計算（参照培地）'};
      } catch (e) { results[s.short] = {status: 'unknown', message: String(e.message || e)}; }
    }));
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
      nodes.append(el('rect', {x: X_M, y: sp.y, width: BAR, height: sp.h, rx: 3, fill: color}));
      const g = FluxOverview.growth(d.e);
      const cy = sp.y + sp.h / 2;
      labels.append(el('text', {x: X_M + BAR / 2, y: sp.y - 22, 'text-anchor': 'middle', class: 'fo-sp', 'font-weight': 700}, d.s.short));
      labels.append(el('text', {x: X_M + BAR / 2, y: sp.y - 6, 'text-anchor': 'middle', class: 'fo-sub'},
        !d.e ? '未計算' : d.e.status !== 'optimal' ? '計算不可' : g.grows ? `成長 ${g.value.toFixed(3)} /h` : '成長なし（目的関数=0）'));
    }
    for (const r of inPools) { const p = inPos.get(r.pool); nodes.append(el('rect', {x: X_L, y: p.y, width: BAR, height: p.h, rx: 2, fill: r.atBound ? '#b4531f' : '#506b78'}));
      const t = el('text', {x: X_L - 8, y: p.y + p.h / 2 + 4, 'text-anchor': 'end', class: 'fo-label'}, `${r.name} ${fmt(r.value)}${r.atBound ? ' ▲' : ''}`);
      if (r.atBound) t.append(el('title', {}, `取込上限に達しています（上限 ${fmt(r.cap)} mmol/gDW/h）。上限は培地濃度からMonod式・在庫量で決まる値で、菌の需要ではありません。`));
      labels.append(t); }
    for (const r of outPools) { const p = outPos.get(r.pool); nodes.append(el('rect', {x: X_R, y: p.y, width: BAR, height: p.h, rx: 2, fill: '#506b78'}));
      labels.append(el('text', {x: X_R + BAR + 8, y: p.y + p.h / 2 + 4, class: 'fo-label'}, `${r.name} ${fmt(r.value)}`)); }
    const bottom = Math.max(sy, ...[...inPos.values(), ...outPos.values()].map(p => p.y + p.h)) + 30;
    svg.setAttribute('viewBox', `0 0 ${W} ${bottom}`);
    const redox = data.flatMap(d => (d.ex?.redox || []).map(r => ({...r, sp: d.s.short})));
    return {svg, redox, empty: !inPools.length && !outPools.length};
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
      if (!g.grows) { box.insertAdjacentHTML('beforeend', '<p class="fo-empty">この培地では成長できず（目的関数=0）、流量はすべて0です。' + (mediumInUse() ? '' : '参照培地にはPfが使える炭素源（グルコース・乳酸・プロピオン酸）がなく、Pfの酸素交換も閉じています。') + '中心代謝マップで炭素源などを加えて「サーバーで再計算」してください（例：乳酸10 mMで成長）。</p>'); wrap.append(box); continue; }
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

  function render() {
    host.innerHTML = '';
    const custom = mediumInUse();
    const head = document.createElement('div');
    head.className = 'fo-head';
    head.innerHTML = '<p><strong>培地成分 → 各菌種 → 代謝の流れ（FBA）</strong>　共通のpFBA解（GEMの目的関数を最大化し、総絶対流量を最小化）。' +
      '<em>各菌種を単独で、同じ培地条件で解いた結果</em>で、菌種間の分泌物の授受は含みません。流量ゼロは他の最適解でもゼロとは限りません。</p>' +
      `<p class="fo-cond">培地条件：<strong>${custom ? '編集した培地' : '参照培地'}</strong>　${esc(Object.values(results || {}).find(v => v?.source)?.source || '')}</p>` +
      '<div class="fo-controls">' +
      `<label>単位 <select id="foUnit"><option value="mmol"${opts.unit === 'mmol' ? ' selected' : ''}>mmol/gDW/h</option><option value="C"${opts.unit === 'C' ? ' selected' : ''}>C-mmol/gDW/h（炭素換算）</option></select></label>` +
      `<label>最小流量 <select id="foMin">${[0.01, 0.05, 0.1, 0.5, 1].map(v => `<option value="${v}"${opts.minFlux === v ? ' selected' : ''}>${v}</option>`).join('')}</select></label>` +
      `<label><input type="checkbox" id="foSolv"${opts.solvents ? ' checked' : ''}> 水・H⁺・CO₂も表示</label>` +
      `<label><input type="checkbox" id="foNet"${opts.net ? ' checked' : ''}> 同じ元素の取込と分泌（Fe³⁺→Fe²⁺）は相殺</label>` +
      '<button type="button" id="foRecalc">サーバーで再計算</button><span id="fluxOvNote" role="status"></span></div>' +
      '<p class="fo-legend"><span class="fo-dot" style="background:#506b78"></span>成分　<span class="fo-dot" style="background:#b4531f"></span>▲ モデル既定の取込上限に達している成分（需要ではなく上限で決まった値）</p>';
    host.append(head);
    const sk = drawExchangeSankey();
    const skWrap = document.createElement('div');
    skWrap.className = 'fo-sankey';
    if (sk.empty) skWrap.innerHTML = '<p class="fo-empty">表示できる交換流量がありません。閾値を下げるか、培地条件を確認してください。</p>';
    skWrap.append(sk.svg);
    host.append(skWrap);
    if (sk.redox.length) {
      const rn = document.createElement('p');
      rn.className = 'fo-redox';
      rn.innerHTML = '相殺した往復：' + sk.redox.map(r => `${esc(r.sp)}：${esc(r.from)}の取込と${esc(r.to)}の分泌 ${fmt(r.value)}（酸化還元による電子の受け渡しで、正味の物質収支はほぼ0）`).join('；');
      host.append(rn);
    }
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
    const row = e.target.closest('.fo-row');
    if (row) {
      const same = openCategory && openCategory.sp === row.dataset.sp && openCategory.cat === row.dataset.cat;
      openCategory = same ? null : {sp: row.dataset.sp, cat: row.dataset.cat};
      render(); return;
    }
    const tr = e.target.closest('tr[data-id]');
    if (tr) { tab('search'); $('species').value = tr.dataset.sp; setSearchGroup(null); $('query').value = tr.dataset.id; renderResults(); showReaction(tr.dataset.sp, tr.dataset.id); }
  });

  let savedTitle = null;
  async function enter() {
    if (!savedTitle) savedTitle = document.querySelector('.panelhead h2').textContent;
    document.querySelector('.panelhead h2').textContent = 'FBAの流れ · 培地成分から代謝へ';
    flowCanvasSuspended = true;
    flowMode = false;
    setLayer('fluxov');
    setView('map');
    host.innerHTML = '<p class="fo-empty" style="padding:24px">読み込み中…</p>';
    await load();
    render();
  }

  const tabBtn = document.createElement('button');
  tabBtn.id = 'fluxTab';
  tabBtn.textContent = 'FBAの流れ';
  $('mapTab').after(tabBtn);
  tabBtn.onclick = () => { tab('map'); enter(); };

  const baseSetLayer = setLayer;
  setLayer = function (layer) {
    baseSetLayer(layer);
    if (layer !== 'fluxov' && savedTitle) { document.querySelector('.panelhead h2').textContent = savedTitle; savedTitle = null; }
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
