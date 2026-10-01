/* Edge audit for the core map: several reactions can draw a side line out of the same metabolite along the
 * same trunk, so a particle on one reaction's line looks like it belongs to another (e.g. g3p -> TKT1 vs
 * TALA). This module (1) detects collinear overlaps between lines of different reactions and (2) fans the
 * overlapping side lines out into parallel lanes by shifting their attachment point on the node.
 * Display only: no coordinates of nodes, reactions or stoichiometry are touched.
 */
const EdgeLanes = (() => {
  const LANE = 9;           // model px between neighbouring lanes (nodes are ~26 px wide)
  const EPS_PAR = 0.02, EPS_DIST = 4, MIN_OVERLAP = 25;

  function polyline(e) {
    const pts = [e.sourceEndpoint()];
    try { const sp = e.segmentPoints(); if (sp && sp.length) pts.push(...sp); } catch { /* straight edge */ }
    pts.push(e.targetEndpoint());
    return pts.map(p => [p.x, p.y]);
  }

  function activeEdges(cy) {
    return cy.edges().filter(e => e.visible() && !e.hasClass('coreCofactorHidden') && e.data('kind') !== 'identityLink');
  }

  function reactionOf(e) { const s = e.source(); return (s.data('reaction') ? s : e.target()).data('reaction') || ''; }

  /* Pairs of different reactions whose lines run on top of each other for more than MIN_OVERLAP px. */
  function overlaps(cy) {
    const segs = [];
    activeEdges(cy).forEach(e => {
      const pts = polyline(e), rid = reactionOf(e);
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1], len = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (len > 1) segs.push({id: e.id(), rid, a, b, len});
      }
    });
    const out = [];
    for (let i = 0; i < segs.length; i++) for (let j = i + 1; j < segs.length; j++) {
      const s1 = segs[i], s2 = segs[j];
      if (s1.id === s2.id || s1.rid === s2.rid) continue;
      const d1 = [s1.b[0] - s1.a[0], s1.b[1] - s1.a[1]], d2 = [s2.b[0] - s2.a[0], s2.b[1] - s2.a[1]];
      if (Math.abs((d1[0] * d2[1] - d1[1] * d2[0]) / (s1.len * s2.len)) > EPS_PAR) continue;
      if (Math.abs(((s2.a[0] - s1.a[0]) * d1[1] - (s2.a[1] - s1.a[1]) * d1[0]) / s1.len) > EPS_DIST) continue;
      const u = [d1[0] / s1.len, d1[1] / s1.len], t = q => (q[0] - s1.a[0]) * u[0] + (q[1] - s1.a[1]) * u[1];
      const ov = Math.min(s1.len, Math.max(t(s2.a), t(s2.b))) - Math.max(0, Math.min(t(s2.a), t(s2.b)));
      if (ov > MIN_OVERLAP) out.push({e1: s1.id, e2: s2.id, r1: s1.rid, r2: s2.rid, length: ov});
    }
    return out;
  }

  /* Direction in which an edge leaves a node: 'h+', 'h-', 'v+', 'v-' (from the first segment at that end). */
  function leaving(e, atSource) {
    const pts = polyline(e), a = atSource ? pts[0] : pts[pts.length - 1], b = atSource ? pts[1] : pts[pts.length - 2];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    return Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? 'h+' : 'h-') : (dy >= 0 ? 'v+' : 'v-');
  }

  /* Give side lines that leave a node in the same direction their own parallel lanes. */
  function separate(cy) {
    const groups = new Map();
    activeEdges(cy).forEach(e => {
      for (const atSource of [true, false]) {
        const node = atSource ? e.source() : e.target();
        if (!node.data('mid')) continue;               // only metabolite ends
        const key = node.id() + '|' + leaving(e, atSource);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push({e, atSource, node});
      }
    });
    let moved = 0;
    for (const [key, items] of groups) {
      if (items.length < 2) continue;
      const reactions = new Set(items.map(x => reactionOf(x.e)));
      if (reactions.size < 2) continue;
      const vertical = key.endsWith('v+') || key.endsWith('v-');
      // the backbone line stays on the node's axis; side lines fan out on both sides of it
      const backbone = items.filter(x => x.e.hasClass('reactionBackbone'));
      const sides = items.filter(x => !x.e.hasClass('reactionBackbone')).sort((p, q) => p.e.id().localeCompare(q.e.id()));
      const lanes = backbone.length ? sides.map((_, i) => (i % 2 ? -1 : 1) * (Math.floor(i / 2) + 1) * LANE)
                                     : sides.map((_, i) => (i - (sides.length - 1) / 2) * LANE);
      sides.forEach((x, i) => {
        const off = lanes[i], v = vertical ? `${off}px 0px` : `0px ${off}px`;
        x.e.style(x.atSource ? 'source-endpoint' : 'target-endpoint', v);
        moved++;
      });
    }
    return moved;
  }

  return {overlaps, separate, leaving, LANE};
})();
if (typeof module !== 'undefined') module.exports = EdgeLanes;

if (typeof flowRender !== 'undefined') (() => {
  // The core geometry (applyCentralGeometry) runs right after flowRender and resets the attachment points,
  // so lanes are applied one tick later, once per distinct geometry.
  let timer = 0, lastSig = '';
  function run() {
    timer = 0;
    try {
      if (!flowMode || !flowState?.coreMode || !mapCy || mapCy.destroyed()) return;
      const sig = activeSig();
      if (sig === lastSig) return;
      EdgeLanes.separate(mapCy);
      lastSig = activeSigAfter = activeSig();
    } catch (e) { console.warn('edge lanes', e); }
  }
  let activeSigAfter = '';
  // signature of the lines as the core code drew them (before our offsets are applied)
  function activeSig() {
    return mapCy.edges().filter(e => e.visible()).map(e => e.id() + ':' + e.style('source-endpoint') + ':' + e.style('target-endpoint')).join('|') +
      '#' + mapCy.nodes('[kind="fm"]').map(n => Math.round(n.position('x')) + ',' + Math.round(n.position('y'))).join(';');
  }
  function schedule() { if (!timer) timer = setTimeout(run, 0); }
  const baseRender = flowRender;
  flowRender = function (...args) { const out = baseRender.apply(this, args); schedule(); return out; };
  const baseApply = applyFluxView;
  applyFluxView = function (...args) { const out = baseApply.apply(this, args); schedule(); return out; };
})();
