"""Export real curated GEM records and a self-contained local SVG viewer."""
from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from main import load_requested_models
from src.b12_evidence import inspect_and_curate
from src.metabolite_ids import canonical_metabolite_id
from cobra.util.solver import linear_reaction_coefficients

OUT=ROOT/'outputs/metabolic_map_20260922'
OUT.mkdir(parents=True,exist_ok=True)
models,evidence=inspect_and_curate(load_requested_models(None,'pf-helper3'))
# The atlas shows Pf with its real exchanges connected to the shared pools (opt-in curation; the dFBA defaults are unchanged).
from src.pf_curation import curate_pf_exchanges
pf_curation_report={}
for _name in list(models):
    if 'freudenreichii' in _name.lower() or _name.lower().startswith('pf'):
        models[_name],pf_curation_report=curate_pf_exchanges(models[_name])
data={'generated':datetime.now(timezone.utc).isoformat(),'species':[], 'b12_audit':evidence,
      'mode':'static curated GEM; no solved fluxes', 'sources':{}}
WEB=ROOT/'scripts/analysis/metabolic_atlas/web'
CSS_FILES=['00-base','10-cytoscape','20-escher','30-flow','40-core','50-runtime','60-sankey','62-home','65-flux_overview','90-theme']
# Script load order matters: later modules wrap functions defined by earlier ones.
JS_MODULES=['00-overview','state','cytoscape','escher','flow','flux','runtime','core','flux_balance','sankey','flux_overview','home','theme']
VENDOR=['cytoscape.min.js','escher.min.js','CYTOSCAPE_LICENSE','ESCHER_LICENSE']
paths=['main.py','src/utils.py','src/metabolite_ids.py','src/b12_evidence.py',
       'src/dfba_simulator.py','src/audited_dfba.py','src/physiology_dfba.py',
       'models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml',
       'models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml',
       'models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml',
       'models/genome/AP019371.1.faa','models/genome/NS21.faa','scripts/analysis/metabolic_atlas/build.py']
paths+=['scripts/analysis/metabolic_atlas/web/index.template.html']
paths+=['scripts/analysis/metabolic_atlas/web/css/%s.css'%n for n in CSS_FILES]
paths+=['scripts/analysis/metabolic_atlas/web/js/%s.js'%n for n in JS_MODULES]
paths+=['scripts/analysis/metabolic_atlas/web/vendor/'+n for n in VENDOR[:2]]
for p in paths:data['sources'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
for name,model in models.items():
    short='OR16' if 'OR16' in name else 'NS21' if 'NS21' in name else 'Pf'
    exchange_ids={r.id for r in model.exchanges}
    species={'objective':{r.id:float(c) for r,c in linear_reaction_coefficients(model).items()},'objective_direction':model.objective.direction,'name':name,'short':short,'model_id':model.id,'reactions':[],
             'metabolites':{m.id:dict(id=m.id,name=m.name,formula=m.formula,compartment=m.compartment,annotation=m.annotation) for m in model.metabolites}}
    for r in model.reactions:
        stoich={m.id:float(c) for m,c in r.metabolites.items()}
        record=dict(id=r.id,name=r.name,equation=r.reaction,bounds=list(r.bounds),gpr=r.gene_reaction_rule,
                    subsystem=r.subsystem,annotation=r.annotation,stoich=stoich,exchange=r.id in exchange_ids)
        if record['exchange'] and len(stoich)==1:
            mid,c=next(iter(stoich.items()))
            record.update(pool=canonical_metabolite_id(mid),uptake=any(v*c>0 for v in r.bounds),secretion=any(v*c<0 for v in r.bounds))
        species['reactions'].append(record)
    assert len(species['reactions'])==len(model.reactions)
    assert len(species['metabolites'])==len(model.metabolites)
    data['species'].append(species)
reference=ROOT/'results/equal_budget_comparison_20260908/design.json'
if reference.exists():
    data['medium_reference']=str(reference.relative_to(ROOT))
    data['medium']=json.loads(reference.read_text())['common_initial_medium']
    data['sources'][str(reference.relative_to(ROOT))]=hashlib.sha256(reference.read_bytes()).hexdigest()
data["model_fingerprint"]=hashlib.sha256(json.dumps({"species":data["species"],"medium":data["medium"]},sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
serialized=json.dumps(data,ensure_ascii=False,allow_nan=False)
(OUT/'model_data.json').write_text(serialized,encoding='utf-8')
# Reaction categories (same rules as the viewer) for the server-side carbon tracing.
import subprocess
subprocess.run(['node',str(ROOT/'scripts/analysis/metabolic_atlas/tools/categories.cjs')],check=True)
# pFBA snapshots so the flux overview opens instantly and works without the server:
# the reference medium plus a few single-carbon-source additions for comparing the three species.
sys.path.insert(0,str(ROOT/'scripts/analysis/metabolic_atlas/server'))
import fba_service
SCENARIOS=[('reference','参照培地',{}),
           ('fed','乳酸流加相当（乳酸6 mM＋NH₄ 3 mM）',{'lac__L_e':6.0,'nh4_e':3.0}),
           ('lac','＋L-乳酸 10 mM',{'lac__L_e':10.0}),
           ('glc','＋グルコース 10 mM',{'glc__D_e':10.0}),
           ('ppa','＋プロピオン酸 10 mM',{'ppa_e':10.0}),
           ('ac','＋酢酸 10 mM',{'ac_e':10.0})]
def solve(sp,medium):
    res=fba_service.check(sp['short'],'',1,ranking=True,medium_override=medium)
    entry={'status':res.get('status'),'message':res.get('message'),'objective':res.get('objective'),'objective_value':res.get('objective_value')}
    if res.get('status')=='optimal':
        entry['fluxes']={k:v for k,v in res['fluxes'].items() if abs(v)>1e-9}
        entry['mass_balance_residual']=res.get('mass_balance_residual')
        entry['uptake_limits']=res.get('uptake_limits')
        if res.get('carbon_flows'):entry['carbon_flows']=res['carbon_flows']
    return entry
snapshot={'method':'pFBA (fba_service.check ranking)','model_fingerprint':data['model_fingerprint'],'scenarios':[]}
for sid,label,add in SCENARIOS:
    medium={**data['medium'],**add}
    snapshot['scenarios'].append({'id':sid,'label':label,'added':add,'species':{sp['short']:solve(sp,medium if add else None) for sp in data['species']}})
snapshot['species']=snapshot['scenarios'][0]['species']   # reference medium, kept for older readers
(OUT/'flux_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def read(path):
    return path.read_text(encoding='utf-8')
template=read(WEB/'index.template.html')
for marker in ('__MODEL_DATA__','__FLUX_SNAPSHOT__','/*__CSS__*/','<!--__SCRIPTS__-->'):
    assert template.count(marker)==1,marker
css='\n'.join(read(WEB/'css'/(n+'.css')) for n in CSS_FILES)
def script(name):
    return '<script>'+read(WEB/'js'/(name+'.js'))+'</script>'
scripts=[]
for name in JS_MODULES:
    if name=='cytoscape': scripts.append('<script>'+read(WEB/'vendor/cytoscape.min.js')+'</script>')
    if name=='escher': scripts.append('<script src="vendor/escher.min.js"></script>')
    scripts.append(script(name))
scripts.append('<script>startAtlas();</script></html>')
html=(template.replace('__MODEL_DATA__',serialized.replace('<','\\u003c'))
      .replace('__FLUX_SNAPSHOT__',json.dumps(snapshot,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')).replace('/*__CSS__*/',css).replace('<!--__SCRIPTS__-->',''.join(scripts)))
(OUT/'vendor').mkdir(exist_ok=True)
for name in VENDOR:
    (OUT/'vendor'/name).write_bytes((WEB/'vendor'/name).read_bytes())
(OUT/'index.html').write_text(html,encoding='utf-8')
nodes=[];edges=[]
for s in data['species']:
    prefix=s['short']+'::'
    for mid,m in s['metabolites'].items():
        nodes.append({'data':{'id':prefix+'m::'+mid,'name':m['name'],'model_id':mid,'species':s['short'],'kind':'metabolite','formula':m['formula'] or '', 'compartment':m['compartment']}})
    for r in s['reactions']:
        rid=prefix+'r::'+r['id']
        nodes.append({'data':{'id':rid,'name':r['name'],'model_id':r['id'],'species':s['short'],'kind':'reaction','gpr':r['gpr'],'lower':r['bounds'][0],'upper':r['bounds'][1],'equation':r['equation']}})
        for mid,co in r['stoich'].items():
            met=prefix+'m::'+mid
            edges.append({'data':{'id':str(len(edges)),'source':met if co<0 else rid,'target':rid if co<0 else met,'coefficient':co,'semantics':'stoichiometric positive direction; see reaction bounds'}})
ids={n['data']['id'] for n in nodes}
assert len(ids)==len(nodes) and all(e['data']['source'] in ids and e['data']['target'] in ids for e in edges)
(OUT/'full_network.cyjs').write_text(json.dumps({'elements':{'nodes':nodes,'edges':edges}},ensure_ascii=False),encoding='utf-8')
checks={'species':[{ 'name':s['short'],'reactions':len(s['reactions']),'metabolites':len(s['metabolites']),
                    'exchanges':sum(r['exchange'] for r in s['reactions'])} for s in data['species']],
        'all_stoichiometry_ids_present':all(all(m in s['metabolites'] for m in r['stoich']) for s in data['species'] for r in s['reactions']),
        'finite_json':True,'no_flux_values':True,'source_sha256':data['sources']}
assert checks['all_stoichiometry_ids_present']
(OUT/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks['species'],ensure_ascii=False))
