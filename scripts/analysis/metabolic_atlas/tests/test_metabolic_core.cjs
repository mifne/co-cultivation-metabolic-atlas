const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const C=require('../web/js/core.js');
const D=JSON.parse(fs.readFileSync(path.join(__dirname,'../../../../outputs/metabolic_map_20260922/model_data.json'),'utf8'));

function verifyRoute(s,route,spec){
  if(!route.ex)return;
  assert(route.ex.exchange&&route.ex.uptake);
  let mid=Object.keys(route.ex.stoich)[0];
  const used=new Set();
  for(const step of route.steps){
    assert.equal(step.from,mid);assert(step.r.stoich[mid]*step.sign<0);
    assert(step.r.stoich[step.to]*step.sign>0);
    assert(step.sign===1?step.r.bounds[1]>0:step.r.bounds[0]<0);
    assert(!used.has(step.r.id));used.add(step.r.id);mid=step.to;
  }
  assert.equal(route.mid,mid);
  if(route.status==='connected')assert(spec.targets.has(mid));
}
for(const s of D.species){
  const spec=C.select(s),index=C.makeIndex(s);
  assert(spec.reactions.length>=25);
  for(const step of spec.reactions){
    assert(step.r.stoich[step.from]*step.sign<0);assert(step.r.stoich[step.to]*step.sign>0);
    assert(step.sign===1?step.r.bounds[1]>0:step.r.bounds[0]<0);
  }
  const glucose=C.connect(s,'glc__D_e',spec,index);verifyRoute(s,glucose,spec);
  assert.equal(glucose.status,'connected');assert.equal(glucose.mid,spec.mids.g6p);
  assert(!glucose.steps.some(x=>C.currency.test(x.to)));
  const lac=C.connect(s,'lac__L_e',spec,index);verifyRoute(s,lac,spec);assert.equal(lac.mid,spec.mids.pyr);
  const nh4=C.connect(s,'nh4_e',spec,index);verifyRoute(s,nh4,spec);assert.equal(nh4.status,'imported');
  const oxygen=C.connect(s,'o2_e',spec,index);verifyRoute(s,oxygen,spec);
  if(s.short!=='Pf'){assert.equal(oxygen.mid,'o2_c');assert(oxygen.steps.every(x=>s.metabolites[x.to].formula==='O2'))}
  assert.equal(C.connect(s,'not_a_pool',spec,index).status,'unavailable');
  // All advertised uptake roots terminate without invented steps or bound violations.
  for(const pool of new Set(s.reactions.filter(r=>r.exchange&&r.uptake&&r.pool).map(r=>r.pool)))verifyRoute(s,C.connect(s,pool,spec,index),spec);
  const ring=['cit','icit','akg','succoa','succ','fum','mal','oaa'];
  for(const key of ring){const p=spec.positions.get(spec.mids[key]);assert(Math.abs(Math.hypot(p.x-1520,p.y-610)-250)<1e-8)}
  console.log(s.short,spec.reactions.length,'core reactions; glucose, lactate, ammonium, oxygen validated');
}
const ns=D.species.find(s=>s.short==='NS21'),nsSpec=C.select(ns);
assert(!nsSpec.reactions.some(x=>x.r.id==='PFK'));
assert(nsSpec.reactions.some(x=>x.r.id==='EDD'));
assert(nsSpec.missing.some(x=>x.label==='PFK'));
const or=C.select(D.species[0]);assert(or.reactions.some(x=>x.r.id==='FRD2rpp'&&x.sign===1));
const pf=C.select(D.species[2]);assert(pf.missing.some(x=>x.label==='SUCOAS'));
assert(pf.reactions.some(x=>x.r.id==='rxn01100_c0'&&x.sign===-1));
assert(C.currency.test('S_cpd00002_c0'));assert(C.currency.test('atp_c'));

// Exercise real state insertion code with only drawing/UI operations stubbed.
const flow=fs.readFileSync(path.join(__dirname,'../web/js/flow.js'),'utf8');
const core=fs.readFileSync(path.join(__dirname,'../web/js/core.js'),'utf8');
const ctx={CoreMetabolism:C,D,pools:[],flowCurrency:C.currency,flowRender(){},separateSideCompounds(){},applyCentralGeometry(){},refreshCentralSummary(){},centralFit(){},mapCy:{elements:()=>({remove(){}})}};
vm.createContext(ctx);
vm.runInContext(flow.slice(flow.indexOf('function flowIndex('),flow.indexOf('function flowChoose(')),ctx);
vm.runInContext(flow.slice(flow.indexOf('function branchEnsure('),flow.indexOf('flowChoose=function(id)')),ctx);
ctx.ensureRequiredSubstrateRecords=()=>{};
vm.runInContext(core.slice(core.indexOf('  function isCentralNode'),core.indexOf('  function applyCentralGeometry')),ctx);
ctx.centralFit=()=>{};
ctx.st=ctx.flowIndex('NS21','glc__D_e');ctx.seedCentralMetabolism(ctx.st);
const before=new Map([...ctx.st.keep].filter(id=>ctx.st.nodes.get(id).data.coreKey||ctx.st.nodes.get(id).data.coreReaction).map(id=>[id,{...ctx.st.nodes.get(id).position}]));
assert.equal(ctx.st.mediumRoots.size,0);
ctx.centralAddNutrients(ctx.st,['glc__D_e','glu__L_e','nh4_e']);
assert.equal(ctx.st.mediumRoots.size,3);
for(const [id,p] of before)assert.deepEqual({...ctx.st.nodes.get(id).position},p,'Core moved: '+id);
assert(ctx.st.keep.has('feed_glc__D_e'));assert(ctx.st.keep.has('r_GLCpts_1'));
const retained=ctx.st.keep.size;
ctx.centralAddNutrients(ctx.st,['glc__D_e']);assert.equal(ctx.st.keep.size,retained);
assert.equal(ctx.st.coreConnections.length,3);
console.log('PASS: central skeleton stays fixed during multiple nutrient additions; duplicate addition is idempotent');
