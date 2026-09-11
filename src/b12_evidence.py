"""Conservative B12 annotation audit; no external B12 dependency is invented."""
from pathlib import Path
import hashlib
import re

ROOT=Path(__file__).resolve().parents[1]
GENOMES={'Actinoplanes_sp_OR16_lcp':'AP019371.1.faa','Rhizobacter_gummiphilus_NS21':'NS21.faa'}


def inspect_and_curate(models):
    copies={n:m.copy() for n,m in models.items()}
    report={}
    for name,filename in GENOMES.items():
        if name not in copies:continue
        path=ROOT/'models/genome'/filename
        headers={}
        for line in path.read_text().splitlines():
            if not line.startswith('>'):continue
            fields=dict(re.findall(r'\[([^=\]]+)=([^\]]*)\]',line))
            if 'locus_tag' in fields:headers[fields['locus_tag']]=dict(fields,header=line)
        model=copies[name];record=dict(genome=str(path.relative_to(ROOT)),
            genome_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reactions=[],changes=[],
            external_b12_dependency_established=False)
        for reaction in model.reactions:
            if reaction.id!='METS' and 'methionine synthase' not in reaction.name.lower():continue
            gpr=reaction.gene_reaction_rule
            genes={g.id:headers.get(g.id) for g in reaction.genes}
            record['reactions'].append(dict(id=reaction.id,equation=reaction.reaction,gpr_before=gpr,genes=genes))
            # Only a simple OR with a retained annotated alternative can be
            # repaired from a pseudo=true annotation. Complex rules and sole
            # pseudo genes are reported for review, not silently disabled.
            terms=gpr.split(' or ')
            if len(terms)>1 and all(re.fullmatch(r'[A-Za-z0-9_]+',t) for t in terms):
                remove=[t for t in terms if headers.get(t,{}).get('pseudo')=='true']
                keep=[t for t in terms if t not in remove]
                if remove and keep and all(t in headers for t in keep):
                    reaction.gene_reaction_rule=' or '.join(keep)
                    record['changes'].append(dict(reaction=reaction.id,before=gpr,after=reaction.gene_reaction_rule,
                        removed_pseudogene_support=remove,flux_bounds_changed=False,
                        reason='local genome annotation pseudo=true; remaining alternative retained, not functionally validated'))
        record['b12_related_annotations']=[v for v in headers.values() if
            re.search(r'cobalamin|cobamide|vitamin B12',v.get('protein',''),re.I) or v.get('gene') in {'metH','metE','btuB','btuC','btuD','btuF'}]
        record['limitations']=['annotation and GPR inspection, not a sequence-domain or expression validation',
            'METS net stoichiometry need not contain its catalytic cofactor',
            'transport, synthesis, salvage and alternative pathways still need functional validation']
        report[name]=record
    return copies,report
