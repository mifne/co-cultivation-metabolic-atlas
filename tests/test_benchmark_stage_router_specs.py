from pathlib import Path
import pytest
from scripts.benchmark_compact_gpu import coverage_router_specs


def test_legacy_maxmin_and_additional_stages_have_distinct_pinned_paths():
    assert coverage_router_specs('old.npz','a'*64,[]) == {'maxmin':(Path('old.npz'),'a'*64)}
    specs=coverage_router_specs('old.npz','a'*64,
        [('aggregate','aggregate.npz','b'*64),('exchange','exchange.npz','c'*64)],gpu_first_batch=True)
    assert set(specs)=={'maxmin','aggregate','exchange'}
    assert specs['exchange']==(Path('exchange.npz'),'c'*64)


@pytest.mark.parametrize('primary,sha,rows,first',[
    ('x',None,[],False),(None,'a'*64,[],False),
    (None,None,[('aggregate','x','b'*64)],False),
    ('x','a'*64,[('maxmin','y','b'*64)],True),
    (None,None,[('bad','x','b'*64)],True),
    (None,None,[('aggregate','x','invalid')],True),
    (None,None,[('aggregate','x','b'*64),('aggregate','y','c'*64)],True),
])
def test_ambiguous_stale_or_unsupported_specifications_fail_before_gpu_setup(primary,sha,rows,first):
    with pytest.raises(ValueError):
        coverage_router_specs(primary,sha,rows,gpu_first_batch=first)
