"""Host-only configuration wiring for zero-repair/hybrid benchmarks."""
from types import SimpleNamespace
import pytest

from scripts.benchmark_compact_gpu import make_hybrid_backend


def options(**overrides):
    values=dict(hybrid_rounds=0,repair_operators=None,cpu_workers=4,
        dispersion_threshold=1.,candidate_limit=1,restricted_columns=[],restricted_pivots=256,
        max_repair_batch=4,repair_cache_size=4,restricted_bucket=True,
        cohort_candidates=False,cohort_replay=False,cpu_basis_proposals=True,
        heterogeneous_candidates=True,heterogeneous_replay=True,candidate_diagnostics=False,
        candidate_oracle=False,hybrid_repair_policy='dispersion',compact_repair_workspace=False,
        gpu_stages=['maxmin'],full_feature_encoding=False,cpu_basis_handoff=False,
        cpu_exchange_support_updates=False,speculative_cpu=True,
        gpu_first_batch=False,gpu_cpu_fallback='exact')
    values.update(overrides)
    return SimpleNamespace(**values)


def test_zero_repair_hybrid_does_not_require_or_create_inverse_operators(monkeypatch):
    import src.gpu_hybrid_lp as module
    calls=[]
    monkeypatch.setattr(module,'HybridCertifiedBackend',lambda *a,**kw: calls.append((a,kw)) or 'hybrid')
    coords,bank=object(),object()
    assert make_hybrid_backend(options(),coords,bank)=='hybrid'
    assert calls[0][0]==(coords,bank)
    kwargs=calls[0][1]
    assert kwargs['repair_rounds']==0 and kwargs['speculative_cpu'] is True
    assert kwargs['gpu_stages']==['maxmin'] and kwargs['repair_columns']==64
    assert kwargs['gpu_first_batch'] is False and kwargs['gpu_cpu_fallback']=='exact'


def test_gpu_first_policy_is_forwarded_without_changing_other_hybrid_options(monkeypatch):
    import src.gpu_hybrid_lp as module
    calls=[]
    monkeypatch.setattr(module,'HybridCertifiedBackend',lambda *a,**kw: calls.append((a,kw)) or 'hybrid')
    args=options(speculative_cpu=False,gpu_first_batch=True,gpu_cpu_fallback='reject',
        gpu_stages=['maxmin','aggregate','exchange'])
    assert make_hybrid_backend(args,object(),object())=='hybrid'
    kwargs=calls[0][1]
    assert kwargs['gpu_first_batch'] is True
    assert kwargs['gpu_cpu_fallback']=='reject'
    assert kwargs['gpu_stages']==['maxmin','aggregate','exchange']
    assert kwargs['speculative_cpu'] is False


def test_positive_repair_without_operators_fails_before_construction(monkeypatch):
    import src.gpu_hybrid_lp as module
    def forbidden(*args,**kwargs):raise AssertionError('Constructed before validation')
    monkeypatch.setattr(module,'HybridCertifiedBackend',forbidden)
    with pytest.raises(ValueError,match='repair operators'):
        make_hybrid_backend(options(hybrid_rounds=1),None,None)


def test_cli_accepts_zero_repair_without_operators_before_existing_output_guard(tmp_path,monkeypatch):
    import sys
    from scripts.benchmark_compact_gpu import main
    output=tmp_path/'existing.json'
    output.touch()
    monkeypatch.setattr(sys,'argv',['benchmark','--bank',str(tmp_path/'not_loaded'),
        '--output',str(output),'--hybrid','--hybrid-rounds','0','--tie-policy','original3'])
    # Reaching this guard proves all CLI policy checks accepted the route;
    # the pre-existing output is deliberately never overwritten or loaded.
    with pytest.raises(FileExistsError):main()
