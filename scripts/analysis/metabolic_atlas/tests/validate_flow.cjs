const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'../../../..');
const src=fs.readFileSync(path.join(__dirname,'../web/js/flow.js'),'utf8');
const D=JSON.parse(fs.readFileSync(path.join(root,'outputs/metabolic_map_20260922/model_data.json'),'utf8'));
const context=vm.createContext({D,pools:[]});
vm.runInContext(src.slice(src.indexOf('const flowCurrency='),src.indexOf('function flowChoose')),context);
const results=[];
for(const s of D.species)for(const pool of ['glu__L_e','nh4_e','pi_e','glc__D_e']){
 const st=vm.runInContext(`flowIndex(${JSON.stringify(s.short)},${JSON.stringify(pool)})`,context);
 for(const e of st.edges)if(!st.nodes.has(e.data.source)||!st.nodes.has(e.data.target))throw Error('missing endpoint');
 for(const [id,p] of st.parent){const e=st.edges.find(e=>e.data.id===p.edge);if(e.data.source!==p.id||e.data.target!==id)throw Error('invalid path');}
 for(const n of st.nodes.values())if(n.position&&(!Number.isFinite(n.position.x)||!Number.isFinite(n.position.y)))throw Error('invalid coordinate');
 results.push({species:s.short,pool,reachable:st.dist.size,rootUptakeEdges:(st.out.get('feed')||[]).length});
}
fs.writeFileSync(path.join(root,'outputs/metabolic_map_20260922/flow_validation.json'),JSON.stringify({checks:'edge endpoints, directed predecessor paths, finite coordinates',results},null,2));console.log(JSON.stringify(results));
