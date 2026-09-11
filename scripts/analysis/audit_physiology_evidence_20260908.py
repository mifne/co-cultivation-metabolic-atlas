"""Preserve real-GEM physiology evidence, source identity and example settings."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from main import load_requested_models
from src.b12_evidence import inspect_and_curate
from src.physiology_dfba import is_atp_hydrolysis
from cobra.io import write_sbml_model

OUT=ROOT/'results/physiology_implementation_20260908';OUT.mkdir(exist_ok=True)
models=load_requested_models(None,'pf-helper3');curated,b12=inspect_and_curate(models)
evidence={}
for name,model in curated.items():
    candidates=[]
    for r in model.reactions:
        consumed=[m for m,c in r.metabolites.items() if c<0 and ('atp'==m.id.split('_')[0].lower() or 'cpd00002' in m.id)]
        produced=[m for m,c in r.metabolites.items() if c>0 and ('adp'==m.id.split('_')[0].lower() or 'cpd00008' in m.id)]
        if is_atp_hydrolysis(r):
            candidates.append(dict(id=r.id,name=r.name,equation=r.reaction,bounds=r.bounds,
                balance=r.check_mass_balance(),gpr=r.gene_reaction_rule,
                metabolites={m.id:dict(name=m.name,coefficient=c,formula=m.formula) for m,c in r.metabolites.items()}))
    evidence[name]=candidates
    if b12.get(name,{}).get('changes'):
        dest=OUT/(name+'_annotation_curated.xml');write_sbml_model(model,str(dest))
(OUT/'b12_evidence.json').write_text(json.dumps(b12,indent=2))
(OUT/'maintenance_candidates.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps({n:[dict(id=r['id'],equation=r['equation'],balance=r['balance']) for r in rs] for n,rs in evidence.items()},indent=2))
print('B12 GPR changes:',json.dumps({n:r['changes'] for n,r in b12.items()}))
