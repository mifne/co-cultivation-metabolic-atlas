/* Cytoscape-side theme (node/edge appearance, legend, type scale). Page CSS is in css/90-theme.css.
 * Presentation only: no data, layout coordinates or FBA logic is touched. */
(() => {
  const PATHWAY = {
    emp:{color:'#2f6fc4',soft:'#e6effb',keys:['g6p','f6p','fdp','dhap','g3p','dpg','pg3','pg2','pep','pyr']},
    ppp:{color:'#7a4fb5',soft:'#f0eaf8',keys:['pgl','pgc','ru5p','r5p','xu5p','s7p','e4p']},
    ed:{color:'#c07a12',soft:'#fbf0dc',keys:['kdp']},
    tca:{color:'#12876f',soft:'#e0f3ee',keys:['accoa','cit','icit','oxs','akg','succoa','succ','fum','mal','oaa']}
  };

  const FONT = "'Inter','Noto Sans JP','Hiragino Sans','Yu Gothic UI',sans-serif";
  const rules = [
    {selector:'node',style:{'font-family':FONT}},
    {selector:'node[kind="fm"]',style:{'text-outline-color':'#fbfcfc','text-outline-width':4,'text-outline-opacity':1,'text-background-opacity':0,color:'#2b4653','font-weight':600,'border-width':3,'border-color':'#8aa1ab','background-color':'#fff'}},
    {selector:'node[kind="fr"]',style:{color:'#38566a','font-weight':700,'text-background-color':'#ffffff','text-background-opacity':.96,'text-background-shape':'roundrectangle','text-background-padding':4,'text-border-width':1,'text-border-color':'#cbd9de','text-border-opacity':1}},
    {selector:'node[kind="fr"]',style:{shape:'round-rectangle',width:16,height:16,'background-color':'#476677','border-width':0,'text-margin-y':-8,'z-index':20}},
    {selector:'node[kind="feed"]',style:{'font-family':FONT,'font-weight':700,color:'#0f5f57','text-outline-color':'#fbfcfc','text-outline-width':4}},
    {selector:'node[kind="fold"]',style:{'background-color':'#e7eef0',color:'#4a6673','font-weight':600}},
    {selector:'node[coreKey]',style:{width:32,height:32,'text-margin-y':14}},
    {selector:'edge',style:{'line-cap':'round','width':2.4,'line-color':'#8da5af','target-arrow-color':'#8da5af','arrow-scale':1.15}},
    {selector:'edge.mainPath',style:{'line-color':'#476677','target-arrow-color':'#476677',width:3}},
    {selector:'edge.reactionSide',style:{'line-color':'#a7b9c0','target-arrow-color':'#a7b9c0',width:1.8}},
    {selector:'node:selected, node.selectedFlow',style:{'overlay-color':'#0f766e','overlay-opacity':.14,'overlay-padding':9}},
    {selector:'node[kind="pathway"], node[kind="pool"]',style:{'background-color':'#fff','border-color':'#2a8c82','border-width':3,'color':'#2b4653','font-weight':600,'text-outline-color':'#fbfcfc','text-outline-width':4}},
    {selector:'node[kind="cm"]',style:{'background-color':'#9bb3bc'}},
    {selector:'node[kind="coreLabel"]',style:{'font-family':FONT,'font-weight':800,'text-background-opacity':1,'text-background-shape':'roundrectangle','text-background-padding':12,'text-border-opacity':0,'text-outline-width':0}}
  ];
  for (const [id, p] of Object.entries(PATHWAY)) {
    for (const k of p.keys)
      rules.push({selector:'node[coreKey="'+k+'"]',style:{'border-color':p.color,'border-width':4,'background-color':'#fff'}});
    rules.push({selector:'#core_label_'+id,style:{color:p.color,'text-background-color':p.soft}});
  }

  const legendItems = [['解糖系・糖新生',PATHWAY.emp.color],['ペントースリン酸経路',PATHWAY.ppp.color],['ED 経路',PATHWAY.ed.color],['TCA 関連代謝',PATHWAY.tca.color]];
  function ensureLegend() {
    if (document.getElementById('atlasLegend') || typeof cyHost === 'undefined' || !cyHost.parentNode) return;
    const box = document.createElement('div');
    box.id = 'atlasLegend';
    box.innerHTML = '<b>PATHWAY</b>' + legendItems.map(([t, c]) => '<span><i style="border-color:' + c + '"></i>' + t + '</span>').join('') +
      '<b style="margin-top:4px">LINES</b><span><u style="background:#476677"></u>反応の主な入力→出力</span><span><u style="background:#a7b9c0;height:2px"></u>同じ反応の他の基質・生成物</span>';
    cyHost.parentNode.style.position = 'relative';
    cyHost.parentNode.append(box);
  }

  // The pathway legend only means something on the central-carbon map.
  function syncLegend() {
    const box = document.getElementById('atlasLegend');
    if (box) box.style.display = (typeof flowMode !== 'undefined' && flowMode && flowState?.coreMode) ? '' : 'none';
  }

  function themeCy() {
    syncLegend();
    if (typeof mapCy === 'undefined' || !mapCy) return;
    const st = mapCy.style();
    const seen = st._atlasRuleKeys ? st._atlasRuleKeys.size : 0;
    if (st._themeSeen === seen) return;
    st.append(rules).update();
    if (!mapCy._themeZoom) { mapCy._themeZoom = true; mapCy.on('zoom', bumpTypography); }
    st._themeSeen = st._atlasRuleKeys ? st._atlasRuleKeys.size : 0;
  }

  // Larger on-map type: the earlier zoom-compensated sizes were too small at fit-to-screen zoom.
  // Where each metabolite's own label sits, so it never lands on a reaction label.
  // TCA ring: outward from the ring centre. Vertical chains: to the right of the node.
  const VERTICAL = new Set(['dhap', 'g6p', 'pgl', 'pgc', 'pyr', 'accoa']);
  const RING = new Set(PATHWAY.tca.keys.filter(k => k !== 'accoa'));
  // Reaction names on the TCA circle sit outside it, so they never lie on the ring's own lines.
  function placeRingReactionLabels(fontSize) {
    const ring = CoreMetabolism.tcaRing, [cx, cy0] = ring.center;
    mapCy.nodes('[kind="fr"]').forEach(n => {
      const p = n.position(), dx = p.x - cx, dy = p.y - cy0, d = Math.hypot(dx, dy);
      if (Math.abs(d - ring.radius) > 1e-6) return;
      const ux = dx / d, uy = dy / d, w = String(n.data('label') || n.data('reaction')).length * fontSize * 0.3;
      n.style({'text-valign': 'center', 'text-halign': 'center',
               'text-margin-x': ux * (w + 22), 'text-margin-y': uy * (fontSize * 0.5 + 20)});
    });
  }

  function placeCoreLabels(fontSize) {
    const centre = mapCy.getElementById('core_label_tca');
    const gap = 26;
    // Side labels stay centred and are shifted by half their width: Cytoscape clips
    // text-halign left/right labels when the font size is changed after layout.
    const half = n => (String(n.data('mid') || n.data('coreKey')).length * fontSize * 0.3) + gap;
    mapCy.nodes('[coreKey]').forEach(n => {
      const k = n.data('coreKey');
      let pos = null;
      if (RING.has(k) && centre.length) {
        const dx = n.position('x') - centre.position('x'), dy = n.position('y') - centre.position('y');
        pos = Math.abs(dx) > Math.abs(dy) * 0.55
          ? {'text-margin-x': dx > 0 ? half(n) : -half(n), 'text-margin-y': 0}
          : {'text-margin-x': 0, 'text-margin-y': dy > 0 ? gap : -gap};
        if (pos['text-margin-y'] === 0) pos['text-valign'] = 'center';
        else pos['text-valign'] = dy > 0 ? 'bottom' : 'top';
      } else if (k === 'g6p') {
        // PGI leaves g6p to the right, G6PDH2r below and the nutrient chain to the left.
        pos = {'text-margin-x': 0, 'text-margin-y': -gap, 'text-valign': 'top'};
      } else if (VERTICAL.has(k)) {
        pos = {'text-margin-x': half(n), 'text-margin-y': 0, 'text-valign': 'center'};
      }
      if (pos) n.style(pos);
    });
  }

  // Straight edges must not run through another metabolite's name. Detour those (only) with a
  // two-bend route that leaves horizontally and enters the target at a steep angle.
  function labelRect(n, fs) {
    const p = n.position(), w = String(n.data('mid') || n.data('coreKey')).length * fs * 0.3;
    const k = n.data('coreKey'), side = VERTICAL.has(k) || RING.has(k);
    return side ? {x1: p.x - 16, x2: p.x + 2 * (w + 26), y1: p.y - fs, y2: p.y + fs}
                : {x1: p.x - w - 6, x2: p.x + w + 6, y1: p.y + 8, y2: p.y + fs + 34};
  }
  function segHitsRect(a, b, r) {
    let t0 = 0, t1 = 1;
    const dx = b.x - a.x, dy = b.y - a.y;
    for (const [p, q] of [[-dx, a.x - r.x1], [dx, r.x2 - a.x], [-dy, a.y - r.y1], [dy, r.y2 - a.y]]) {
      if (p === 0) { if (q < 0) return false; continue; }
      const t = q / p;
      if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; }
    }
    return true;
  }
  // Routing is decided at a fixed reference font size so it does not change while zooming.
  const ROUTE_FS = 24;
  function routeAroundLabels() {
    if (!flowState?.coreMode) return;
    const metabolites = mapCy.nodes('[coreKey]').filter(n => n.visible());
    mapCy.edges().forEach(e => {
      if (e.hasClass('coreCofactorHidden')) return;
      const a = e.source(), b = e.target(), rn = a.data('reaction') ? a : b.data('reaction') ? b : null;
      const mn = rn === a ? b : a;
      if (!rn || !mn.data('coreKey')) return;
      const routed = e.data('themeRouted');
      if (!routed && e.style('curve-style') !== 'straight') return;
      const pa = a.position(), pb = b.position();
      const hit = metabolites.some(m => m.id() !== mn.id() && segHitsRect(pa, pb, labelRect(m, ROUTE_FS)));
      let route = null;
      if (hit) {
        // Bend at the reaction's own height, ~100 units before the metabolite, then go straight in.
        const from = rn.position(), to = mn.position(), sign = to.x >= from.x ? 1 : -1;
        const bend = {x: to.x - sign * 100, y: from.y};
        const S = rn === a ? pa : pb, T = rn === a ? pb : pa;
        const dx = T.x - S.x, dy = T.y - S.y, L2 = dx * dx + dy * dy, L = Math.sqrt(L2);
        const w = ((bend.x - S.x) * dx + (bend.y - S.y) * dy) / L2;
        const d = ((bend.x - S.x) * (-dy) + (bend.y - S.y) * dx) / L;
        // Only accept a detour whose bend lies between the endpoints; otherwise keep it straight.
        if (w > 0.1 && w < 0.9) route = {w, d};
      }
      if (route) {
        e.data('themeRouted', 1);
        e.style({'curve-style': 'segments', 'segment-weights': [route.w], 'segment-distances': [route.d]});
      } else if (routed) {
        e.data('themeRouted', 0);
        e.style({'curve-style': 'straight'});
      }
    });
  }

  function bumpTypography() {
    if (typeof mapCy === 'undefined' || !mapCy || !flowState?.coreMode) return;
    const z = mapCy.zoom();
    mapCy.batch(() => {
      const fs = Math.min(46, Math.max(19, 10.5 / z));
      mapCy.nodes('[coreKey]').style({'font-size': fs});
      placeCoreLabels(fs);
      placeRingReactionLabels(mapCy.nodes('[kind="fr"]').length ? Math.min(28, Math.max(16, 9 / z)) : 16);
      routeAroundLabels();
      mapCy.nodes('[kind="fr"]').style({'font-size': Math.min(28, Math.max(16, 9 / z))});
    });
  }

  const baseRender = flowRender;
  flowRender = function (...args) {
    const out = baseRender.apply(this, args);
    try { themeCy(); ensureLegend(); syncLegend(); bumpTypography(); } catch (e) { console.warn('atlas theme', e); }
    return out;
  };
})();
