"""Lifecycle safety tests; fake native models, no timed GEM workload/GPU."""
import gc
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.cpu_repeated_lp import RepeatedCpuLP, _NUMERICAL_OPTIONS, _QUALITY_FIELDS


@pytest.fixture
def native_events(monkeypatch):
    import highspy
    events = []

    class FakeHighs:
        serial = 0

        def __init__(self):
            self.serial = FakeHighs.serial
            FakeHighs.serial += 1
            self.owner = threading.get_ident()
            self.options = {key:1e-7 for key in _NUMERICAL_OPTIONS}
            self.options['presolve'] = 'choose'
            self.valid = False
            self._event('create')

        def _event(self, operation):
            events.append((self.serial, operation, threading.get_ident()))
            assert threading.get_ident() == self.owner, operation+' crossed native owner threads'

        def setOptionValue(self, name, value):
            self._event('setOptionValue')
            self.options[name] = value
            return highspy.HighsStatus.kOk

        def passModel(self, lp):
            self._event('passModel')
            self.cost = np.array(lp.col_cost_, copy=True)
            self.lower = np.array(lp.col_lower_, copy=True)
            self.upper = np.array(lp.col_upper_, copy=True)
            assert lp.num_row_ == 0
            return highspy.HighsStatus.kError if self.cost[0] == 99. else highspy.HighsStatus.kOk

        def run(self):
            self._event('run')
            self.x = np.where(self.cost < 0., self.upper, self.lower)
            self.valid = True
            return highspy.HighsStatus.kOk

        def getModelStatus(self):
            self._event('getModelStatus')
            return highspy.HighsModelStatus.kOptimal

        def getSolution(self):
            self._event('getSolution')
            return SimpleNamespace(col_value=self.x, col_dual=self.cost,
                row_value=np.empty(0), row_dual=np.empty(0),
                value_valid=self.valid, dual_valid=self.valid)

        def getInfo(self):
            self._event('getInfo')
            return SimpleNamespace(simplex_iteration_count=0,
                **{key:0. for key in _QUALITY_FIELDS})

        def getOptions(self):
            self._event('getOptions')
            return SimpleNamespace(**self.options)

        def getBasis(self):
            self._event('getBasis')
            return SimpleNamespace(valid=self.valid,
                col_status=np.where(self.cost < 0., 2, 0), row_status=np.empty(0,dtype=int))

        def changeColsBounds(self, count, indices, lower, upper):
            self._event('changeColsBounds')
            self.lower[indices] = lower
            self.upper[indices] = upper
            return highspy.HighsStatus.kOk

        def changeColsCost(self, count, indices, cost):
            self._event('changeColsCost')
            self.cost[indices] = cost
            return highspy.HighsStatus.kOk

        def modelStatusToString(self, status):
            self._event('modelStatusToString')
            return 'Optimal'

        def disableCallbacks(self):
            self._event('disableCallbacks')

        def clear(self):
            self._event('clear')
            self.valid = False
            return highspy.HighsStatus.kOk

        def clearSolver(self):
            self._event('clearSolver')
            self.valid = False
            return highspy.HighsStatus.kOk

        def __del__(self):
            # Do not retain the instance itself in the event recorder.
            events.append((self.serial, 'destroy', threading.get_ident()))

    monkeypatch.setattr(highspy, 'Highs', FakeHighs)
    return events


def request(upper=2., cost=-1., stage='maxmin'):
    return np.array([cost]), dict(bounds=(0.,upper), _stage=stage)


def assert_owner_lifetimes(events):
    created = {serial:thread for serial, name, thread in events if name == 'create'}
    assert created
    for serial, owner in created.items():
        rows = [(name,thread) for item,name,thread in events if item == serial]
        assert all(thread == owner for _,thread in rows)
        operations = [name for name,_ in rows]
        assert operations[-3:] == ['disableCallbacks','clear','destroy']


def test_create_update_solve_snapshot_and_destroy_stay_on_stable_owner(native_events):
    main_thread = threading.get_ident()
    service = RepeatedCpuLP(2)
    first = service.solve_batch([request(),request(3.)],environment_ids=[100,200])
    assert [result.x[0] for result in first] == [2.,3.]
    original_lanes = dict(service._environment_lanes)
    # Reversed/subset requests must never migrate an existing model to a new worker.
    second = service.solve_batch([request(4.),request(5.)],environment_ids=[200,100])
    assert [result.x[0] for result in second] == [4.,5.]
    assert all(result.diagnostics['basis_reused'] for result in second)
    assert service._environment_lanes == original_lanes
    assert all(result.solution_snapshot.col_value.flags.owndata for result in second)
    first[0].solution_snapshot.col_value[:] = 123.
    first[0].solution_snapshot.col_dual[:] = 234.
    safe_view = service.models[(100,'maxmin',0,1,0)]['solver']
    detached = safe_view.getSolution()
    np.testing.assert_array_equal(detached.col_value,[5.])
    np.testing.assert_array_equal(detached.col_dual,[-1.])
    detached.col_value[:] = 999.
    np.testing.assert_array_equal(safe_view.getSolution().col_value,[5.])
    with pytest.raises(RuntimeError,match='owner thread'):
        safe_view.run()
    service.close()
    service.close()
    gc.collect()
    assert all(thread != main_thread for _,_,thread in native_events)
    assert_owner_lifetimes(native_events)
    assert not service.models and service.closed
    # Already copied results remain usable after native memory is released.
    np.testing.assert_array_equal(second[0].solution_snapshot.col_value,[4.])
    with pytest.raises(RuntimeError,match='closed'):
        safe_view.getSolution()


def test_legacy_pool_submission_is_routed_to_owner_not_dispatch_thread(native_events):
    service = RepeatedCpuLP(2)
    try:
        future = service.pool.submit(service._solve,(77,request()))
        first = future.result()
        second = service.solve_batch([request(7.)],environment_ids=[77])[0]
        assert first.success and second.success and second.diagnostics['basis_reused']
        assert len({thread for _,name,thread in native_events if name == 'run'}) == 1
    finally:
        service.close()
    assert_owner_lifetimes(native_events)


def test_models_clear_disposes_on_owner_and_rebuilds_without_changing_lane(native_events):
    service = RepeatedCpuLP(2)
    try:
        service.solve_batch([request()],environment_ids=['seed-A'])
        lane = service._environment_lanes['seed-A']
        service.models.clear()
        assert not service.models
        assert_owner_lifetimes(native_events)
        rebuilt = service.solve_batch([request(6.)],environment_ids=['seed-A'])[0]
        assert rebuilt.diagnostics['model_rebuilt']
        assert service._environment_lanes['seed-A'] == lane
    finally:
        service.close()
    assert_owner_lifetimes(native_events)


def test_model_creation_failure_is_released_on_owner(native_events):
    service = RepeatedCpuLP(1)
    try:
        with pytest.raises(RuntimeError,match='model creation'):
            service.solve_batch([request(cost=99.)])
        assert not service.models
        assert_owner_lifetimes(native_events)
    finally:
        service.close()


def test_cleanup_failure_still_releases_every_native_model_and_joins_lanes(native_events,monkeypatch):
    import highspy
    original = highspy.Highs.clear
    def failing_clear(native):
        original(native)
        return highspy.HighsStatus.kError
    monkeypatch.setattr(highspy.Highs,'clear',failing_clear)
    service = RepeatedCpuLP(2)
    service.solve_batch([request(),request()],environment_ids=[10,20])
    with pytest.raises(RuntimeError,match='cleanup failed'):
        service.close()
    assert service.closed and not service.models
    assert_owner_lifetimes(native_events)
    assert all(not thread.is_alive() for lane in service._lanes for thread in lane._threads)
