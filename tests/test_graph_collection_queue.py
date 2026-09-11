import json
import pytest
from scripts.queue_graph_collection import verify_dependency
from tests.test_graph_learning_curve import collection_fixture


@pytest.mark.parametrize('status',['running','failed'])
def test_dependency_failure_never_launches_next(tmp_path,status):
    c=collection_fixture(tmp_path);c['requested_job']=dict(status=status,split='selection',count=4)
    (tmp_path/'catalog.json').write_text(json.dumps(c))
    with pytest.raises(RuntimeError,match='dependency'):verify_dependency(tmp_path,4)


def test_completed_other_split_is_not_selection_success(tmp_path):
    c=collection_fixture(tmp_path);c['requested_job']=dict(status='completed',split='train',count=4)
    (tmp_path/'catalog.json').write_text(json.dumps(c))
    with pytest.raises(RuntimeError):verify_dependency(tmp_path,4)


def test_verified_selection_allows_next_job(tmp_path):
    c=collection_fixture(tmp_path);c['requested_job']=dict(status='completed',split='selection',count=4)
    (tmp_path/'catalog.json').write_text(json.dumps(c))
    data=verify_dependency(tmp_path,4)
    assert len(data.select('selection',4))==4
    with pytest.raises(RuntimeError):verify_dependency(tmp_path,16)
