"""Read-only inventory of vitamin interfaces relevant to proposed cross-feeding."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis.screen_propionibacterium_helper import _models
result={}
for name, model in _models(True).items():
    vitamin=[]
    for met in model.metabolites:
        label=(met.id+' '+str(met.name)).lower()
        if any(t in label for t in ['cobalamin','cobamide','cobamamide','cobinamide','corrin','b12','cbl']):
            vitamin.append(dict(id=met.id,name=met.name,compartment=met.compartment,
                reactions=[dict(id=r.id,equation=r.reaction,bounds=r.bounds,genes=r.gene_reaction_rule) for r in sorted(met.reactions,key=lambda r:r.id)]))
    result[name]=dict(vitamin_metabolites=vitamin,exchanges=[dict(id=r.id,equation=r.reaction,bounds=r.bounds) for r in model.exchanges if any(t in (r.id+' '+r.reaction).lower() for t in ['b12','cbl','lac__l','ppa_e','o2_'])])
path=ROOT/'results/symbiosis_reassessment_20260908/structure_audit.json'
path.write_text(json.dumps(result,indent=2))
for name,data in result.items():
    print(name, 'vitamin metabolites:', [(m['id'],m['name'],len(m['reactions'])) for m in data['vitamin_metabolites']])
    print('interfaces:',data['exchanges'])
