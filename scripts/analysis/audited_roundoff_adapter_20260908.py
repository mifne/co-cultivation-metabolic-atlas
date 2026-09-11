"""Precisely scoped C30 roundoff repair while an immutable audit run finishes.

Every previously successful trial passed a strict nonnegative C30 check,
so the added conditional is never entered along any successful old run.
The adapter is explicitly recorded and is not a license to mix teachers.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def install():
    from src.audited_dfba import AuditedDFBASimulator
    prepare=AuditedDFBASimulator._prepare_oxygen
    integrate=AuditedDFBASimulator._integrate
    def corrected_prepare(self,*args,**kwargs):
        prepare(self,*args,**kwargs)
        concentration=self.state.metabolites.get('C30_oligo_e',0.)
        if -1e-12<=concentration<0.:
            addition=-concentration
            self.state.metabolites['C30_oligo_e']=0.
            self.last_polymer_fluxes['carbon_c5_equivalent_error_mmol_l']+=6*addition
            self._oxygen_context['c30_roundoff_added']=addition
        # A larger negative value still fails the original nonnegative gate.
    def corrected_integrate(self,*args,**kwargs):
        integrate(self,*args,**kwargs)
        addition=self._oxygen_context.get('c30_roundoff_added',0.)
        if addition:
            ledger=self.accounting_audit['roundoff_added_mmol_l']
            ledger['C30_oligo_e']=ledger.get('C30_oligo_e',0.)+addition
    AuditedDFBASimulator._prepare_oxygen=corrected_prepare
    AuditedDFBASimulator._integrate=corrected_integrate

def main():
    import argparse
    from scripts.analysis.reassess_audited_symbiosis_20260908 import worker
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    provenance=dict(patch='c30_roundoff_20260908',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_source_sha256=hashlib.sha256((ROOT/'src/audited_dfba.py').read_bytes()).hexdigest(),
        threshold_mmol_l=1e-12,scope='negative C30 rounding only; committed corrections ledgered',
        successful_old_runs_equivalent='Every old trial strictly rejected C30 < 0, so this branch cannot affect any previously successful run.')
    (a.output/'numerical_patch.json').write_text(json.dumps(provenance,indent=2))
    install();worker(json.loads(a.case),a.output)

if __name__=='__main__':main()
