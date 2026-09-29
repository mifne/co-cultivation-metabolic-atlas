"""Export real curated GEM records and a self-contained local SVG viewer."""
from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from main import load_requested_models
from src.b12_evidence import inspect_and_curate
from src.metabolite_ids import canonical_metabolite_id
from cobra.util.solver import linear_reaction_coefficients

OUT=ROOT/'outputs/metabolic_map_20260922'
OUT.mkdir(parents=True,exist_ok=True)
models,evidence=inspect_and_curate(load_requested_models(None,'pf-helper3'))
data={'generated':datetime.now(timezone.utc).isoformat(),'species':[], 'b12_audit':evidence,
      'mode':'static curated GEM; no solved fluxes', 'sources':{}}
paths=['main.py','src/utils.py','src/metabolite_ids.py','src/b12_evidence.py',
       'src/dfba_simulator.py','src/audited_dfba.py','src/physiology_dfba.py',
       'models/sbml/final_consortium/Actinoplanes_sp_OR16_lcp.xml',
       'models/sbml/final_consortium/Rhizobacter_gummiphilus_NS21.xml',
       'models/sbml/helper_candidates/Propionibacterium_freudenreichii_shermanii_curated.xml',
       'models/genome/AP019371.1.faa','models/genome/NS21.faa',
       'scripts/analysis/build_metabolic_map_20260922.py','scripts/analysis/metabolic_map_template.html',
       'scripts/analysis/metabolic_map_flux.js','scripts/analysis/metabolic_map_flow.js','scripts/analysis/metabolic_map_complete.js','scripts/analysis/metabolic_map_escher.js','outputs/metabolic_map_20260922/vendor/escher.min.js','scripts/analysis/metabolic_map_cytoscape.js','outputs/metabolic_map_20260922/vendor/cytoscape.min.js']
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
serialized=json.dumps(data,ensure_ascii=False,allow_nan=False)
(OUT/'model_data.json').write_text(serialized,encoding='utf-8')
template=(ROOT/'scripts/analysis/metabolic_map_template.html').read_text(encoding='utf-8')
assert template.count('__MODEL_DATA__')==1
html=template.replace('__MODEL_DATA__',serialized.replace('<','\\u003c'))
extension=(ROOT/'scripts/analysis/metabolic_map_cytoscape.js').read_text(encoding='utf-8')
library=(OUT/'vendor/cytoscape.min.js').read_text(encoding='utf-8')
html=html.replace('</body>','').replace('</html>','')+'<script>'+library+'</script><script>'+extension+'</script></html>'
escher_library=(OUT/'vendor/escher.min.js').read_text(encoding='utf-8')
escher_extension=(ROOT/'scripts/analysis/metabolic_map_escher.js').read_text(encoding='utf-8')
html=html.replace('</html>','')+'<script src='+chr(34)+'vendor/escher.min.js'+chr(34)+'></script><script>'+escher_extension+'</script></html>'
html=html.replace('</html>','')+'<script>'+(ROOT/'scripts/analysis/metabolic_map_complete.js').read_text(encoding='utf-8')+'</script></html>'
html=html.replace('</html>','')+'<script>'+(ROOT/'scripts/analysis/metabolic_map_flow.js').read_text(encoding='utf-8')+'</script></html>'
html=html.replace('</html>','')+'<script>'+(ROOT/'scripts/analysis/metabolic_map_flux.js').read_text(encoding='utf-8')+'</script></html>'
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
