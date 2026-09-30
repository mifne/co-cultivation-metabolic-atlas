/* Curated central-carbon skeleton and bounded, model-directed nutrient connectors.
 * This module never changes GEM bounds or the medium used by FBA.
 */
const CoreMetabolism = (() => {
  const aliases = {
    g6p:['g6p_c','00079'], f6p:['f6p_c','00072'], fdp:['fdp_c','00290'],
    dhap:['dhap_c','00095'], g3p:['g3p_c','00102'], dpg:['13dpg_c','00203'],
    pg3:['3pg_c','00169'], pg2:['2pg_c','00482'], pep:['pep_c','00061'],
    pyr:['pyr_c','00020'], accoa:['accoa_c','00022'], pgl:['6pgl_c','00911'],
    pgc:['6pgc_c','00284'], ru5p:['ru5p__D_c','00171'], r5p:['r5p_c','00101'],
    xu5p:['xu5p__D_c','00198'], s7p:['s7p_c','00238'], e4p:['e4p_c','00236'],
    kdp:['2ddg6p_c','00805'], cit:['cit_c','00137'], icit:['icit_c','00260'],
    oxs:['oxalsucc_c','03187'], akg:['akg_c','00024'], succoa:['succoa_c','00078'],
    succ:['succ_c','00036'], fum:['fum_c','00106'], mal:['mal__L_c','00130'], oaa:['oaa_c','00032']
  };
  const steps = [
    ['EMP','PGI','g6p','f6p','PGI','rxn00558_c0'],
    ['EMP','PFK','f6p','fdp','PFK','rxn00545_c0'],
    ['EMP','F1,6BP → F6P','fdp','f6p','FBP','rxn00551_c0'],
    ['EMP','FBA','fdp','g3p','FBA','rxn00786_c0'],
    ['EMP','TPI','dhap','g3p','TPI','rxn00747_c0'],
    ['EMP','GAPD','g3p','dpg','GAPD','rxn00781_c0'],
    ['EMP','PGK','dpg','pg3','PGK','rxn01100_c0'],
    ['EMP','PGM','pg3','pg2','PGM','rxn01106_c0'],
    ['EMP','ENO','pg2','pep','ENO','rxn00459_c0'],
    ['EMP','PYK','pep','pyr','PYK','rxn00148_c0'],
    ['PPP','G6PDH','g6p','pgl','G6PDH2r','rxn00604_c0'],
    ['PPP','PGL','pgl','pgc','PGL','rxn01476_c0'],
    ['PPP','GND','pgc','ru5p','GND','rxn01115_c0'],
    ['PPP','RPI','ru5p','r5p','RPI','rxn00777_c0'],
    ['PPP','RPE','ru5p','xu5p','RPE','rxn01116_c0'],
    ['PPP','TKT1','r5p','s7p','TKT1','rxn01200_c0'],
    ['PPP','TALA','s7p','e4p','TALA','rxn01333_c0'],
    ['PPP','TKT2','xu5p','f6p','TKT2','rxn00785_c0'],
    ['ED','EDD','pgc','kdp','EDD',null],
    ['ED','EDA','kdp','pyr','EDA',null],
    ['Entry','Pyruvate → Acetyl-CoA','pyr','accoa','PDH','rxn05938_c0'],
    ['TCA','CS','oaa','cit','CS','rxn00256_c0'],
    ['TCA','ACONT','cit','icit','ACONT','rxn00973_c0'],
    ['TCA','ICDH','icit','akg','ICDHyr',null],
    ['TCA','ICDH (1)','icit','oxs',null,'rxn01387_c0'],
    ['TCA','ICDH (2)','oxs','akg',null,'rxn00199_c0'],
    ['TCA','2-oxoglutarate → Succinyl-CoA','akg','succoa','AKGDH','rxn05939_c0'],
    ['TCA','SUCOAS','succoa','succ','SUCOAS',null],
    ['TCA','Succinate → Fumarate','succ','fum','SUCDi|FRD2rpp','rxn09272_c0'],
    ['TCA','FUM','fum','mal','FUM','rxn00799_c0'],
    ['TCA','MDH','mal','oaa','MDH','rxn00248_c0']
  ];
  const points = {g6p:[0,0],f6p:[190,0],fdp:[380,0],g3p:[570,0],dhap:[570,-150],
    dpg:[760,0],pg3:[950,0],pg2:[1140,0],pep:[1330,0],pyr:[1520,0],accoa:[1520,220],
    pgl:[0,220],pgc:[0,440],ru5p:[0,660],r5p:[220,660],xu5p:[220,440],s7p:[440,660],e4p:[440,440],kdp:[220,220]};
  const ring=['cit','icit','akg','succoa','succ','fum','mal','oaa'];
  ring.forEach((k,i)=>{const a=(-90+i*45)*Math.PI/180;points[k]=[1520+250*Math.cos(a),610+250*Math.sin(a)]});
  points.oxs=[1750,514];
  const currency = /^(?:(?:h|h2o|atp|adp|amp|gtp|gdp|pi|ppi|nad|nadh|nadp|nadph|coa|co2|hco3|o2|fad|fadh2|q8|q8h2|mqn8|mql8|2dmmq8|2dmmql8|fdxox|fdxrd)_[cep]\d*|(?:S_)?cpd(?:00001|00002|00003|00004|00005|00006|00007|00008|00009|00010|00011|00012|00015|00018|00031|00038|00067|00982|11620|11621|15499|15500|15560|15561)_\w+)$/;
  function select(s){
    const mids=Object.fromEntries(Object.entries(aliases).map(([k,[bigg,seed]])=>[k,s.short==='Pf'?'S_cpd'+seed+'_c0':bigg]));
    const rs=new Map(s.reactions.map(r=>[r.id,r])),reactions=[],missing=[];
    for(const [group,label,a,b,bigg,pf] of steps){
      const ids=(s.short==='Pf'?pf:bigg)?.split('|')||[];
      // The optional alternative ICDH definition and ED are species-specific.
      if(!ids.length){if(label==='SUCOAS')missing.push({group,label,reason:'登録なし'});continue}
      const from=mids[a],to=mids[b];let chosen;
      for(const id of ids){const r=rs.get(id);if(!r||!r.stoich[from]||!r.stoich[to]||r.stoich[from]*r.stoich[to]>=0)continue;
        const sign=r.stoich[from]<0?1:-1;if(sign===1?r.bounds[1]<=0:r.bounds[0]>=0)continue;
        chosen={group,label,r,sign,from,to,a,b};break;
      }
      if(chosen)reactions.push(chosen);else missing.push({group,label,reason:'登録なし、または指定方向を許容しない'});
    }
    const positions=new Map(),targets=new Set();
    for(const x of reactions)for(const key of [x.a,x.b]){const mid=mids[key],p=points[key];positions.set(mid,{x:p[0],y:p[1]});targets.add(mid)}
    return {reactions,missing,positions,targets,mids};
  }
  function carbons(s,mid){const f=s.metabolites[mid]?.formula||'',m=f.match(/C(\d*)(?![a-z])/);return m?Number(m[1]||1):0}
  function intracellular(s,mid){const c=s.metabolites[mid]?.compartment;return c==='c'||c==='c0'||/_c\d*$/.test(mid)}
  function makeIndex(s){
    const consuming=new Map();
    for(const r of s.reactions){
      if(r.exchange||/^(?:DM_|SK_|sink|biomass|BIOMASS)/i.test(r.id)||Object.keys(r.stoich).length<2||r.id in (s.objective||{}))continue;
      for(const sign of [1,-1]){if(sign===1?r.bounds[1]<=0:r.bounds[0]>=0)continue;
        const inputs=Object.keys(r.stoich).filter(m=>r.stoich[m]*sign<0),outputs=Object.keys(r.stoich).filter(m=>r.stoich[m]*sign>0);
        for(const from of inputs){if(!consuming.has(from))consuming.set(from,[]);consuming.get(from).push({r,sign,outputs})}
      }
    }return consuming;
  }
  function connect(s,pool,spec,index=makeIndex(s)){
    const exchanges=s.reactions.filter(r=>r.exchange&&r.uptake&&r.pool===pool);
    if(!exchanges.length)return {pool,status:'unavailable',reason:'このモデルに取込可能な交換反応がありません。',steps:[]};
    const frontier=[],best=new Map(),scheduled=new Map();
    const compare=(a,b)=>a.cost-b.cost||a.mid.localeCompare(b.mid);
    function enqueue(item){
      if(scheduled.has(item.mid)&&scheduled.get(item.mid)<=item.cost)return;
      scheduled.set(item.mid,item.cost);frontier.push(item);let i=frontier.length-1;
      while(i){const parent=(i-1)>>1;if(compare(frontier[parent],item)<=0)break;frontier[i]=frontier[parent];i=parent}frontier[i]=item;
    }
    function dequeue(){
      const first=frontier[0],last=frontier.pop();if(!frontier.length)return first;let i=0;
      while(i*2+1<frontier.length){let child=i*2+1;if(child+1<frontier.length&&compare(frontier[child+1],frontier[child])<0)child++;
        if(compare(last,frontier[child])<=0)break;frontier[i]=frontier[child];i=child}
      frontier[i]=last;return first;
    }
    for(const ex of exchanges){const mid=Object.keys(ex.stoich)[0];enqueue({mid,ex,steps:[],cost:0,carbon:carbons(s,mid)>0})}
    let fallback=null,visited=0;
    while(frontier.length&&visited++<6000){
      const path=dequeue();
      if(best.has(path.mid)&&best.get(path.mid)<=path.cost)continue;best.set(path.mid,path.cost);
      if(spec.targets.has(path.mid)&&(!path.carbon||!currency.test(path.mid)))return {...path,pool,status:'connected',reason:'構造上の接続候補（流量・原子追跡の証明ではありません）'};
      if(intracellular(s,path.mid)&&(!fallback||path.cost<fallback.cost))fallback=path;
      // Inorganic inputs may join a core reaction as the actual co-substrate.
      if(!path.carbon&&intracellular(s,path.mid)&&spec.reactions.some(x=>x.r.stoich[path.mid]*x.sign<0))return {...path,pool,status:'cosubstrate',reason:'中心代謝の補助基質として接続。炭素骨格への変換ではありません。'};
      if(path.steps.length>=10||(!path.carbon&&intracellular(s,path.mid)))continue;
      for(const item of index.get(path.mid)||[]){
        if(path.steps.some(x=>x.r.id===item.r.id))continue;
        for(const to of item.outputs){
          if(to===path.mid||best.has(to))continue;
          if(intracellular(s,path.mid)&&!intracellular(s,to))continue;
          if(path.carbon&&(currency.test(to)||carbons(s,to)<2||currency.test(path.mid)))continue;
          // Inorganic roots may be transported, but must not become H+ or water
          // simply by following a shared reaction. This is not atom mapping.
          if(!path.carbon){const initial=Object.keys(path.ex.stoich)[0],formula=s.metabolites[initial]?.formula;if(!formula||s.metabolites[to]?.formula!==formula)continue}
          const otherSubstrates=Object.entries(item.r.stoich).filter(([m,c])=>m!==path.mid&&c*item.sign<0&&!currency.test(m));
          const cost=path.cost+1+Math.abs(carbons(s,path.mid)-carbons(s,to))*.15+otherSubstrates.length*.35;
          enqueue({...path,mid:to,cost,steps:[...path.steps,{r:item.r,sign:item.sign,from:path.mid,to}]});
        }
      }
    }
    const path=fallback||{ex:exchanges[0],mid:Object.keys(exchanges[0].stoich)[0],steps:[]};
    return {...path,pool,status:fallback?'imported':'unconnected',reason:fallback?'細胞内への取込まで表示。選定した中心代謝への接続は探索範囲内で見つかりません。':'交換反応まで表示。中心代謝への接続は探索範囲内で見つかりません。'};
  }
  const shortLabels=Object.fromEntries(Object.entries(aliases).map(([key,[mid]])=>[key,mid.replace(/_c$/,'')]));
  return {select,connect,makeIndex,currency,points,shortLabels};
})();
if(typeof module!=='undefined')module.exports=CoreMetabolism;

if(typeof document!=='undefined'){
  function isCentralNode(data,st=flowState){return !!st?.coreMode&&!!(data?.coreKey||data?.coreReaction)}
  function centralPut(st,id,position){const n=st.nodes.get(id);if(n){n.position={...position};st.keep.add(id);st.layoutDone.add(id)}}
  function centralReaction(st,entry){
    const id=branchEnsure(st,entry.r,entry.sign),n=st.nodes.get(id);
    n.data.coreReaction=true;n.data.coreFrom=entry.from;n.data.coreTo=entry.to;
    n.data.requirementOrigin=entry.from;
    const a=st.coreSpec.positions.get(entry.from),b=st.coreSpec.positions.get(entry.to);
    centralPut(st,id,{x:(a.x+b.x)/2,y:(a.y+b.y)/2});
    // Parallel PFK / FBPase use separate, short lanes.
    if(entry.label==='F1,6BP → F6P'&&st.coreSpec.reactions.some(x=>x.label==='PFK'))n.position.y+=80;
    for(const [mid,c] of Object.entries(entry.r.stoich)){
      const nid='m_'+mid;st.keep.add(nid);st.keepEdges.add(branchConnect(st,c*entry.sign<0?nid:id,c*entry.sign<0?id:nid));
    }
  }
  function seedCentralMetabolism(st){
    st.coreMode=true;st.coreSpec=CoreMetabolism.select(st.s);st.coreIndex=CoreMetabolism.makeIndex(st.s);
    st.coreConnections=[];st.coreAuxVisible=false;st.mediumRoots=new Set();
    for(const key of ['keep','keepEdges','expanded','visibleProducts','renderNodes','layoutDone'])st[key]=new Set();
    st.majorShown=new Map();st.majorPending=new Map();st.hiddenReactions=new Set();st.hiddenNodes=new Set();
    for(const [mid,p] of st.coreSpec.positions){const n=st.nodes.get('m_'+mid);n.data.coreKey=Object.keys(st.coreSpec.mids).find(k=>st.coreSpec.mids[k]===mid);centralPut(st,'m_'+mid,p)}
    for(const entry of st.coreSpec.reactions)centralReaction(st,entry);
    // Position non-backbone products locally; do not inherit the old BFS layout.
    for(const entry of st.coreSpec.reactions){const id=branchEnsure(st,entry.r,entry.sign),p=st.nodes.get(id).position;let i=0;
      for(const mid of Object.keys(entry.r.stoich)){const n=st.nodes.get('m_'+mid);if(!n.data.coreKey&&!st.layoutDone.has(n.data.id))centralPut(st,n.data.id,{x:p.x+(++i%2?70:-70),y:p.y+85})}
    }
    st.selected='m_'+st.coreSpec.mids.g6p;st.path=new Set(st.keep);st.pathEdges=new Set(st.keepEdges);
    mapCy.elements().remove();flowRender();separateSideCompounds();applyCentralGeometry();centralFit();
  }
  function centralFit(){if(!flowState?.coreMode)return;mapCy.resize();mapCy.fit(mapCy.elements().filter(e=>e.visible()),48);scheduleSessionSave()}
  function centralAddNutrients(st,poolsToAdd){
    const added=[];
    for(const pool of poolsToAdd){
      if(st.mediumRoots.has(pool))continue;
      const result=CoreMetabolism.connect(st.s,pool,st.coreSpec,st.coreIndex);
      if(!result.ex){st.coreConnections.push({pool,status:result.status,reason:result.reason});continue}
      const terminal='m_'+result.mid,anchor=st.nodes.get(terminal)?.data.coreKey?st.nodes.get(terminal).position:{x:-330,y:st.coreConnections.length*280};
      const inward=anchor.y<=100&&anchor.x>150?{x:0,y:-1}:anchor.x<700?{x:-1,y:0}:{x:1,y:0};
      const chain=['feed_'+pool,branchEnsure(st,result.ex,1),'m_'+Object.keys(result.ex.stoich)[0]];
      const feed=chain[0];st.nodes.set(feed,{data:{id:feed,kind:'feed',pool,label:pool},position:{x:0,y:0}});
      st.nodes.get(chain[1]).data.direction='取込';
      for(const x of result.steps){const rid=branchEnsure(st,x.r,x.sign);st.nodes.get(rid).data.requirementOrigin=x.from;st.nodes.get(rid).data.connectorFrom=x.from;st.nodes.get(rid).data.connectorTo=x.to;chain.push(rid,'m_'+x.to)}
      const occupied=[...st.keep].map(id=>st.nodes.get(id)).filter(n=>n?.position&&!CoreMetabolism.currency.test(n.data.mid||''));
      const place=(i,lane)=>({x:anchor.x+inward.x*(chain.length-1-i)*155-inward.y*lane,y:anchor.y+inward.y*(chain.length-1-i)*155+inward.x*lane});
      const laneScore=lane=>chain.reduce((score,id,i)=>{if(st.keep.has(id))return score;const p=place(i,lane);return score+occupied.filter(n=>Math.abs(n.position.x-p.x)<95&&Math.abs(n.position.y-p.y)<65).length*1000},Math.abs(lane)*.02);
      const lane=[0,220,-220,440,-440].sort((a,b)=>laneScore(a)-laneScore(b))[0];
      for(let i=0;i<chain.length;i++){
        const id=chain[i],n=st.nodes.get(id);
        if(!st.keep.has(id))centralPut(st,id,place(i,lane));
        st.keep.add(id);if(i)st.keepEdges.add(branchConnect(st,chain[i-1],id));
        if(n.data.reaction&&!n.data.coreReaction){n.data.nutrientConnector=true;ensureRequiredSubstrateRecords(id);branchProducts(st,id)}
      }
      st.mediumRoots.add(pool);st.coreConnections.push({pool,status:result.status,reason:result.reason,target:result.mid,reactions:result.steps.map(x=>x.r.id)});added.push(pool);
    }
    // Root paths remain available to legacy search/branch selection, without reflowing the core.
    const roots=[...st.mediumRoots].map(pool=>'feed_'+pool);st.dist=new Map(roots.map(id=>[id,0]));st.parent=new Map();st.queue=[...roots];
    for(let i=0;i<st.queue.length;i++)for(const e of st.out.get(st.queue[i])||[]){const to=e.data.target;if(!st.dist.has(to)){st.dist.set(to,st.dist.get(st.queue[i])+1);st.parent.set(to,{id:st.queue[i],edge:e.data.id});st.queue.push(to)}}
    flowRender();separateSideCompounds();applyCentralGeometry();refreshCentralSummary();return added.length;
  }
  function applyCentralGeometry(){
    const st=flowState,cy=mapCy;if(!st?.coreMode||!cy)return;
    const core=cy.nodes().filter(n=>isCentralNode(n.data(),st));core.removeClass('escherAux').addClass('centralSkeleton');
    cy.nodes('[coreKey]').addClass('escherPrimary');
    cy.nodes('[kind="fm"]').forEach(n=>{
      const hidden=!st.coreAuxVisible&&CoreMetabolism.currency.test(n.data('mid')||'')&&!isCentralNode(n.data(),st)&&!st.coreConnections.some(x=>x.target===n.data('mid'));
      n.toggleClass('coreCofactorHidden',hidden);
    });
    cy.edges().forEach(e=>e.toggleClass('coreCofactorHidden',e.source().hasClass('coreCofactorHidden')||e.target().hasClass('coreCofactorHidden')));
    cy.edges('.reactionSide').filter(e=>e.source().data('coreKey')&&e.target().data('reaction')||e.target().data('coreKey')&&e.source().data('reaction')).forEach(e=>{
      // Shared central intermediates use rectilinear joins, never auxiliary-placement curves.
      const a=e.source().position(),b=e.target().position();
      e.style({'source-endpoint':'outside-to-node','target-endpoint':'outside-to-node'});
      if(Math.abs(a.y-b.y)<30&&Math.abs(a.x-b.x)>250){e.style({'curve-style':'segments','segment-weights':[.08,.92],'segment-distances':[a.x<b.x?-105:105,a.x<b.x?-105:105]})}
      else e.style({'curve-style':'taxi','taxi-direction':Math.abs(a.x-b.x)>Math.abs(a.y-b.y)?'horizontal':'vertical','taxi-turn':35,'taxi-turn-min-distance':18});
    });
    cy.nodes('[kind="coreLabel"]').remove();
    const labels=[['emp','解糖系・糖新生',830,-170],['ppp','ペントースリン酸経路',220,810],['tca','TCA 関連代謝',1520,610]];
    if(st.coreSpec.reactions.some(x=>x.group==='ED'))labels.push(['ed','ED 経路',710,240]);
    cy.add(labels.map(([id,label,x,y])=>({data:{id:'core_label_'+id,kind:'coreLabel',label},position:{x,y},grabbable:false,selectable:false})));
    appendFlowStyle(cy,[
      {selector:'.coreCofactorHidden',style:{display:'none'}},
      {selector:'node[coreKey]',style:{width:26,height:26,'font-size':20,'text-margin-y':10}},
      {selector:'node[coreReaction]',style:{'font-size':17}},
      {selector:'node[kind="coreLabel"]',style:{shape:'rectangle',width:1,height:1,'background-opacity':0,'border-width':0,'font-size':30,'font-weight':'bold','text-valign':'center','text-margin-y':0,'text-background-opacity':0,color:'#456775',events:'no'}}
    ]).update();
    applyCentralTypography();fluxDirty=true;
  }
  function applyCentralTypography(){
    if(!flowState?.coreMode||!mapCy)return;const z=mapCy.zoom();
    mapCy.batch(()=>{
      mapCy.nodes('[coreKey]').forEach(n=>n.style({label:z<.5?CoreMetabolism.shortLabels[n.data('coreKey')]:n.data('mid'),'font-size':Math.min(44,Math.max(18,10/z))}));
      mapCy.nodes('[kind="fr"]').style({'text-opacity':z<.32?0:1,'font-size':Math.min(24,Math.max(14,8/z))});
      mapCy.nodes('[kind="feed"]').style({'font-size':Math.min(40,Math.max(18,10/z))});
      mapCy.nodes('[kind="coreLabel"]').style({'font-size':Math.min(54,Math.max(28,12/z))});
    });
  }
  function refreshCentralSummary(){
    const st=flowState,box=$('centralSummary');if(!st?.coreMode||!box)return;
    const counts={};for(const x of st.coreSpec.reactions)counts[x.group]=(counts[x.group]||0)+1;
    box.innerHTML='<p>'+esc(names[st.s.short])+' · 中心代謝 '+st.coreSpec.reactions.length+'反応</p><p>栄養を選ぶと、交換・輸送・変換反応を中心代謝へ接続します。</p><p class="coreNote">対応なし：'+st.coreSpec.missing.map(x=>esc(x.label)).join('、')+'。未登録の区間は補っていません。</p>'+
      '<details><summary>採用経路と未登録の区間</summary><p>解糖系・糖新生 '+(counts.EMP||0)+' / PPP '+(counts.PPP||0)+' / ED '+(counts.ED||0)+' / TCA関連 '+(counts.TCA||0)+'</p><p>'+st.coreSpec.missing.map(x=>esc(x.label)+'：'+esc(x.reason)).join('<br>')+'</p><p>モデルの登録状況であり、実細胞の欠損を示すものではありません。</p></details>'+
      '<div id="centralConnections">'+st.coreConnections.map(x=>'<p><strong>'+esc(x.pool)+'</strong> → '+esc(x.target||'接続なし')+'<br><small>'+esc(x.reason)+'</small></p>').join('')+'</div>';
    if($('mediumRootsSummary'))$('mediumRootsSummary').textContent=st.mediumRoots.size?'栄養の入口：'+[...st.mediumRoots].join('、'):'栄養の入口は未選択';
  }
  function installCentralControls(){
    const st=flowState;if(!st?.coreMode)return;
    document.querySelector('.panelhead h2').textContent='中心代謝と栄養の接続 · '+names[st.s.short];flowTab.textContent='中心代謝マップ';
    let panel=$('centralControls');if(!panel){panel=document.createElement('section');panel.id='centralControls';panel.innerHTML='<h3>中心代謝から探索</h3><div id="centralSummary"></div><button id="centralAdd">＋ 栄養を選択して接続</button> <button id="centralFit">全体を表示</button><div class="coreFocus"><button data-core-focus="EMP">解糖系</button><button data-core-focus="PPP">PPP</button><button data-core-focus="TCA">TCA</button></div><label class="coreToggle"><input id="centralAux" type="checkbox">水・ATPなどの補助基質も表示</label><p class="coreNote">初期表示は中心炭素代謝の骨格です。補助基質は上の切替、全量論は反応IDのクリックで確認できます。接続線は構造上の候補で、実際の流れはFBA表示で確認してください。</p>';
      $('context').prepend(panel);
    }
    const speciesLabel=$('flowSpecies').parentElement;panel.prepend(speciesLabel);
    $('centralAdd').onclick=()=>$('addMediumComponents').click();$('centralFit').onclick=centralFit;
    panel.querySelectorAll('[data-core-focus]').forEach(b=>b.onclick=()=>{const entries=st.coreSpec.reactions.filter(x=>x.group===b.dataset.coreFocus),ids=new Set(entries.flatMap(x=>['m_'+x.from,'m_'+x.to,branchEnsure(st,x.r,x.sign)]));mapCy.fit(mapCy.nodes().filter(n=>ids.has(n.id())&&n.visible()),60);scheduleSessionSave()});
    $('centralAux').checked=!!st.coreAuxVisible;$('centralAux').onchange=()=>{st.coreAuxVisible=$('centralAux').checked;applyCentralGeometry();if(st.coreAuxVisible){separateSideCompounds();applyCentralGeometry()}applyFluxView();scheduleSessionSave()};
    $('flowPool').parentElement.hidden=true;$('addMediumComponents').hidden=true;
    $('flowRoot').textContent='栄養の入口へ';$('flowRoot').onclick=()=>{const ids=[...st.mediumRoots].map(x=>'feed_'+x),nodes=mapCy.nodes().filter(n=>ids.includes(n.id()));if(nodes.length){mapCy.fit(nodes,100);if(mapCy.zoom()>1)mapCy.zoom(1);mapCy.center(nodes)}else $('centralAdd').click()};
    $('flowFind').textContent='化合物・反応を探す';
    if(!$('resetCentralView')){const reset=document.createElement('button');reset.id='resetCentralView';reset.textContent='中心代謝の初期表示に戻す';reset.onclick=()=>{seedCentralMetabolism(st);installCentralControls();refreshMediumSummary();scheduleSessionSave()};panel.append(reset)}
    $('flowSpecies').onchange=()=>openFlow($('flowSpecies').value,'glc__D_e');
    if(!mapCy._coreTypography){mapCy._coreTypography=true;mapCy.on('zoom',applyCentralTypography)}
    // Move advanced controls below the primary workflow rather than hiding capabilities.
    if(!$('centralAdvanced')){const details=document.createElement('details');details.id='centralAdvanced';const summary=document.createElement('summary');summary.textContent='計算・保存・詳細設定';details.append(summary);for(const child of [...$('context').children])if(child!==panel)details.append(child);$('context').append(details)}
    refreshCentralSummary();
  }
  const originalCoreNodeCheck=isTcaLayoutNode;
  isTcaLayoutNode=function(data,st=flowState){return isCentralNode(data,st)||originalCoreNodeCheck(data,st)};
  const originalCorePair=isTcaBackbonePair;
  isTcaBackbonePair=function(reaction,mid,st=flowState){if(st?.coreMode){const entry=st.coreSpec?.reactions.find(x=>x.r.id===reaction);if(entry)return mid===entry.from||mid===entry.to;const n=[...st.keep].map(id=>st.nodes.get(id)?.data).find(d=>d?.reaction===reaction&&d.nutrientConnector);if(n)return mid===n.connectorFrom||mid===n.connectorTo}return originalCorePair(reaction,mid,st)};
  const originalCoreCenter=flowCenterRoute;
  flowCenterRoute=function(exchangeOnly=false){if(!flowState?.coreMode)return originalCoreCenter(exchangeOnly);if(exchangeOnly){$('flowRoot')?.click();return}centralFit()};
  const originalCoreReflow=flowReflow;
  flowReflow=function(){if(!flowState?.coreMode)return originalCoreReflow();};
  const originalMajorRecord=recordMajorEntrances;
  recordMajorEntrances=function(st,id){originalMajorRecord(st,id);if(st.coreMode){st.majorPending?.delete('tca');st.majorPending?.delete('glycolysis')}};
  const originalMediumRoots=addMediumRoots;
  addMediumRoots=function(st,selected){return st.coreMode?centralAddNutrients(st,selected):originalMediumRoots(st,selected)};
  const originalCoreRender=flowRender;
  flowRender=function(){originalCoreRender();if(flowState?.coreMode)applyCentralGeometry()};
  const originalCoreOpen=openFlow;
  openFlow=function(short='NS21',pool='glc__D_e'){
    originalCoreOpen(short,pool);seedCentralMetabolism(flowState);installCentralControls();
  };
  const originalCorePack=packAtlasSession;
  packAtlasSession=function(){const p=originalCorePack();if(flowState.coreMode){p.view.coreMode=1;p.view.coreAuxVisible=!!flowState.coreAuxVisible;p.view.coreConnections=flowState.coreConnections}return p};
  const originalCoreRestore=restoreAtlasSession;
  restoreAtlasSession=async function(payload){
    await originalCoreRestore(payload);const st=flowState;
    if(payload.view.coreMode===1){st.coreMode=true;st.coreSpec=CoreMetabolism.select(st.s);st.coreIndex=CoreMetabolism.makeIndex(st.s);st.coreAuxVisible=!!payload.view.coreAuxVisible;st.coreConnections=Array.isArray(payload.view.coreConnections)?payload.view.coreConnections:[];st.mediumRoots=new Set(payload.view.mediumRoots);installCentralControls();applyCentralGeometry();refreshCentralSummary();scheduleSessionSave()}
    else{st.coreMode=false;$('context').prepend($('flowSpecies').parentElement);$('centralControls')?.remove();$('flowPool').parentElement.hidden=false;$('addMediumComponents').hidden=false;mapCy.nodes('[kind="coreLabel"]').remove();mapCy.elements().removeClass('coreCofactorHidden');document.querySelector('.panelhead h2').textContent='保存した経路 · '+names[st.s.short];const b=document.createElement('button');b.textContent='中心代謝から新しく探索';b.onclick=()=>openFlow(st.s.short,'glc__D_e');$('context').prepend(b);}
  };
  const originalCoreStart=startAtlas;
  startAtlas=async function(){
    let legacy=null;try{const raw=localStorage.getItem('metabolic-atlas-session');if(raw&&JSON.parse(raw).view?.coreMode===1)localStorage.setItem('metabolic-atlas-core-migrated','1');if(raw&&JSON.parse(raw).view?.coreMode!==1&&!localStorage.getItem('metabolic-atlas-core-migrated')&&!new URLSearchParams(location.search).has('session')){legacy=raw;localStorage.setItem('metabolic-atlas-before-core',raw);localStorage.setItem('metabolic-atlas-core-migrated','1');localStorage.removeItem('metabolic-atlas-session')}}catch{}
    const migrating=!!legacy;
    await originalCoreStart();
    if(migrating){try{const payload=JSON.parse(legacy),v=validateAtlasSession(payload);if(v.species!==flowState.s.short)openFlow(v.species,'glc__D_e');flowState.mediumOverride=v.mediumOverride;flowState.mediumDraft=v.mediumDraft;refreshMediumSummary();scheduleSessionSave()}catch{/* The old session remains available for explicit recovery. */}}
    try{legacy=legacy||localStorage.getItem('metabolic-atlas-before-core')}catch{}
    if(legacy){const b=document.createElement('button');b.id='restorePreCore';b.textContent='更新前の表示を復元';b.onclick=async()=>{try{await restoreAtlasSession(JSON.parse(legacy))}catch(e){sessionError(e)}};$('atlasSessionControls').append(b)}
  };
  
}
