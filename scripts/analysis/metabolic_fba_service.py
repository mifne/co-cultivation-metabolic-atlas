"""Read-only reaction feasibility for the atlas reference-medium snapshot."""
import json, hashlib, threading
from pathlib import Path
import cobra
from cobra.flux_analysis.loopless import loopless_solution
from cobra.util.array import create_stoichiometric_matrix
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cultivation_numerics import compute_uptake_limits
from src.metabolite_ids import canonical_metabolite_id
DATA=ROOT/'outputs/metabolic_map_20260922/model_data.json'
lock=threading.Lock()
cache={}
THRESHOLD=1e-6

def check(short,rid,sign,ranking=False,medium_override=None):
 if sign not in (-1,1):return {'status':'unknown','message':'方向が不正です'}
 raw=DATA.read_bytes(); fingerprint=hashlib.sha256(raw).hexdigest()
 d=json.loads(raw)
 if medium_override is not None:
  allowed=set(d['medium'])|{r.get('pool') for sp in d['species'] for r in sp['reactions'] if r.get('pool')}
  if not isinstance(medium_override,dict) or any(k not in allowed or isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for k,v in medium_override.items()):return {'status':'unknown','message':'培地濃度は登録成分の有限な非負数で指定してください'}
 medium=dict(d['medium'] if medium_override is None else medium_override)
 key=(fingerprint,short,rid,sign,ranking,json.dumps(medium,sort_keys=True))
 with lock:
  if key in cache:return cache[key]
  d=json.loads(raw); s=next((s for s in d['species'] if s['short']==short),None)
  if not s or (not ranking and not any(r['id']==rid for r in s['reactions'])):return {'status':'unknown','message':'反応がありません'}
  conditions={'medium':'参照培地の初期濃度（全成分）' if medium_override is None else 'ユーザー指定培地濃度','medium_mmol_l':medium,'uptake_rule':'既存 compute_uptake_limits：Monod + inventory cap','max_uptake_mmol_g_h':20,'biomass_g_l':0.21,'inventory_window_h':0.025,'oxygen':'初期DOの在庫・Monod制限、再曝気なし','water_proton':'既存規則：水1000、H+ 0.1以下','growth':'増殖下限なし。GEMの既存反応下限は維持','scope':'単一細胞・定常の供給可能性。共培養供給・動態・実測を保証しない','model_sha256':fingerprint,'threshold':THRESHOLD}
  result={'status':'unknown','reaction':rid,'direction':sign,'conditions':conditions}
  try:
   model=cobra.Model(short);model.solver='glpk';model.solver.configuration.timeout=25
   mets={mid:cobra.Metabolite(mid,compartment=m.get('compartment') or 'c') for mid,m in s['metabolites'].items()}
   reactions=[];mapping={};bounds={}
   for rec in s['reactions']:
    r=cobra.Reaction(rec['id']);r.bounds=tuple(rec['bounds']);r.add_metabolites({mets[m]:c for m,c in rec['stoich'].items()});reactions.append(r);bounds[r.id]=r.bounds
    if rec['exchange']:
     if len(rec['stoich'])!=1 or next(iter(rec['stoich'].values()))!=-1:raise ValueError('unsupported exchange direction')
     pool=canonical_metabolite_id(next(iter(rec['stoich'])))
     if pool in mapping:raise ValueError('ambiguous exchange pool '+pool)
     mapping[pool]=r.id
   model.add_reactions(reactions)
   limits=compute_uptake_limits(medium,{short:.21},{short:mapping},{short:bounds},.025,20)[short]
   for pool,ex in mapping.items():
    r=model.reactions.get_by_id(ex);original=max(0,-bounds[ex][0]);c=max(0,medium.get(pool,0))
    if pool=='h2o_e':limit=min(original,1000)
    elif pool=='h_e':limit=min(original,.1)
    elif pool=='o2_e':limit=min(original,20*c/(.01+c),c/(.21*.025))
    else:limit=limits.get(pool,0)
    r.bounds=(-limit,max(0,r.upper_bound))
   # Close all other boundary inflows; intracellular demand sinks may only consume.
   for r in model.boundary:
    if r.id in mapping.values():continue
    if len(r.metabolites)==1:
     c=next(iter(r.metabolites.values()))
     r.bounds=(max(0,r.lower_bound),r.upper_bound) if c<0 else (r.lower_bound,min(0,r.upper_bound))
   if ranking:
    from cobra.flux_analysis import pfba
    objective=s.get('objective',{})
    if not objective:raise ValueError('GEM objective is not exported')
    model.objective={model.reactions.get_by_id(k):v for k,v in objective.items()};model.objective.direction=s.get('objective_direction','max')
    primary=model.optimize()
    if primary.status!='optimal':raise ValueError('primary objective '+primary.status)
    solution=pfba(model)
    if solution.status!='optimal':raise ValueError('pFBA '+solution.status)
    flux=solution.fluxes
    residual=float(np.max(np.abs(create_stoichiometric_matrix(model)@np.array([flux[r.id] for r in model.reactions]))))
    violation=max(max(r.lower_bound-flux[r.id],flux[r.id]-r.upper_bound,0) for r in model.reactions)
    achieved=sum(v*flux[k] for k,v in objective.items())
    if residual>1e-7 or violation>1e-7 or abs(achieved-primary.objective_value)>1e-6*max(1,abs(primary.objective_value)):raise ValueError('pFBA certificate failed')
    conditions['growth']='GEMの元の目的関数を最適化後、総絶対流量を最小化'
    result.update(status='optimal',method='pFBA',objective=objective,objective_direction=model.objective.direction,objective_value=float(achieved),fluxes={k:float(v) for k,v in flux.items()},mass_balance_residual=residual,conditions=conditions,message='共通のpFBA解。流量ゼロは別の最適解でもゼロとは限りません。')
    cache[key]=result
    return result
   target=model.reactions.get_by_id(rid);model.objective=model.problem.Objective(sign*target.flux_expression,direction='max')
   solution=model.optimize()
   result['solver_status']=solution.status
   if solution.status=='infeasible':result.update(status='infeasible',message='この条件ではモデル全体が実行不能です。特定基質の不足とは断定できません。')
   elif solution.status!='optimal':result['message']='最適性を確認できませんでした'
   elif sign*solution.fluxes[rid]<=THRESHOLD:result.update(status='infeasible',message='指定条件では、この方向の反応流量が判定閾値を超えません',flux=float(solution.fluxes[rid]))
   else:
    # Do not freeze the maximized target objective during cycle removal.
    model.objective=model.problem.Objective(0)
    certificate=loopless_solution(model,fluxes=solution.fluxes.to_dict())
    if certificate.status!='optimal':raise ValueError('loop removal not optimal')
    flux=certificate.fluxes;v=np.array([flux[r.id] for r in model.reactions]);res=float(np.max(np.abs(create_stoichiometric_matrix(model)@v)));violation=max(max(r.lower_bound-flux[r.id],flux[r.id]-r.upper_bound,0) for r in model.reactions)
    result.update(flux=float(flux[rid]),mass_balance_residual=res,bound_violation=float(violation),method='FBA + CycleFreeFlux (COBRApy loopless_solution)')
    if sign*flux[rid]>THRESHOLD and res<1e-7 and violation<1e-7:
     substrates=[]
     for met,c in target.metabolites.items():
      if c*sign>=0:continue
      suppliers=[{'reaction':r.id,'production':float(r.metabolites[met]*flux[r.id])} for r in met.reactions if r.id!=rid and r.metabolites[met]*flux[r.id]>THRESHOLD]
      substrates.append({'metabolite':met.id,'required_flux':float(-c*flux[rid]),'suppliers':sorted(suppliers,key=lambda x:-x['production'])})
     result.update(status='feasible',message='参照条件で全基質を同時供給する循環除去後の定常解を確認',substrates=substrates)
    else:result['message']='循環除去後に反応成立を確認できませんでした（判定保留）'
  except Exception as exc:result.update(status='unknown',message='計算を完了できませんでした',error=str(exc))
  if result['status']!='unknown':cache[key]=result
  return result
