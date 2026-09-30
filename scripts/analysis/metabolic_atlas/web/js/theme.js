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
    box.innerHTML = '<b>PATHWAY</b>' + legendItems.map(([t, c]) => '<span><i style="border-color:' + c + '"></i>' + t + '</span>').join('');
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
  function bumpTypography() {
    if (typeof mapCy === 'undefined' || !mapCy || !flowState?.coreMode) return;
    const z = mapCy.zoom();
    mapCy.batch(() => {
      mapCy.nodes('[coreKey]').style({'font-size': Math.min(46, Math.max(19, 10.5 / z))});
      mapCy.nodes('[kind="fr"]').style({'font-size': Math.min(28, Math.max(16, 9 / z))});
    });
  }

  // The overview levels are drawn by atlasDraw rather than flowRender; theme them too.
  const baseAtlasDraw = atlasDraw;
  atlasDraw = function (...args) {
    const out = baseAtlasDraw.apply(this, args);
    try { themeCy(); } catch (e) { console.warn('atlas theme', e); }
    return out;
  };

  const baseRender = flowRender;
  flowRender = function (...args) {
    const out = baseRender.apply(this, args);
    try { themeCy(); ensureLegend(); syncLegend(); bumpTypography(); } catch (e) { console.warn('atlas theme', e); }
    return out;
  };
})();
