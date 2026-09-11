from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.benchmark_compact_gpu import matched_actions, repeat_schedule, run_ordered_pair


def test_default_schedule_preserves_identical_seeds_and_cpu_first_order():
    schedule=repeat_schedule(10,3,2)
    assert schedule==[
        dict(repeat=0,seeds=[10,11,12],execution_order='cpu-first'),
        dict(repeat=1,seeds=[10,11,12],execution_order='cpu-first')]


def test_independent_seed_stride_and_alternating_execution_order():
    schedule=repeat_schedule(20289801,32,6,32,'alternate')
    assert [row['execution_order'] for row in schedule]==['cpu-first','gpu-first']*3
    assert schedule[-1]['seeds']==list(range(20289961,20289993))
    assert len({seed for row in schedule for seed in row['seeds']})==192


def test_training_guard_checks_later_repeats_before_any_execution():
    with pytest.raises(ValueError,match='Training/test overlap'):
        repeat_schedule(100,2,3,10,'alternate',train_seeds=[121])
    with pytest.raises(ValueError,match='nonnegative'):
        repeat_schedule(1,2,3,-1)
    assert repeat_schedule(100,2,2,10,'gpu-first')[1]['execution_order']=='gpu-first'


def test_action_generation_is_identical_to_old_draw_and_order_independent():
    seeds=[3,8]
    expected=[np.random.default_rng(seed).uniform(.05,.95,(120,5)).astype(np.float32)[:2] for seed in seeds]
    actual=matched_actions(seeds,2)
    for left,right in zip(actual,expected):
        np.testing.assert_array_equal(left,right)
        assert left.dtype==np.float32


@pytest.mark.parametrize('order,expected',[('cpu-first',['cpu','gpu']),('gpu-first',['gpu','cpu'])])
def test_execution_order_preserves_result_labels_and_side_local_patch_contexts(order,expected):
    state=SimpleNamespace(optimizer='original');calls=[]
    def cpu():
        assert state.optimizer=='original'
        calls.append('cpu');return 'cpu-result'
    def gpu():
        with patch.object(state,'optimizer','forbidden'):
            assert state.optimizer=='forbidden'
            calls.append('gpu')
        return 'gpu-result'
    assert run_ordered_pair(order,cpu,gpu)==('cpu-result','gpu-result')
    assert calls==expected and state.optimizer=='original'
