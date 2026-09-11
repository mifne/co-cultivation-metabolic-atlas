"""Never consume final seeds before a passing, matching development run."""
import json
import sys
import pytest
from scripts import qualify_gpu_simplex


@pytest.mark.parametrize("steps,passed,method,message", [
    (3,True,"tableau","Full development"),
    (120,False,"tableau","Full development"),
    (120,True,"pdlp","configuration differs")])
def test_refuses_unqualified_development(tmp_path,monkeypatch,steps,passed,method,message):
    source=tmp_path/"development.json"
    source.write_text(json.dumps(dict(config=dict(steps=steps,method=method),runs=[dict(passed=passed)])))
    out=tmp_path/"final"
    monkeypatch.setattr(sys,"argv",["qualify","--development",str(source),"--output-dir",str(out)])
    def forbidden(*args,**kwargs): raise AssertionError("Final seed was consumed")
    monkeypatch.setattr(qualify_gpu_simplex.subprocess,"run",forbidden)
    with pytest.raises(RuntimeError,match=message): qualify_gpu_simplex.main()
    assert not out.exists()


@pytest.mark.parametrize("runs",[[],[dict(passed=True)]])
def test_refuses_empty_or_forged_pass(tmp_path,monkeypatch,runs):
    source=tmp_path/"development.json"
    config=dict(steps=120,method="tableau",host_presolve=True,rank_reduce=False,
        objective_scale=1.,quadratic_regularization=0.,reaction_support=None)
    source.write_text(json.dumps(dict(config=config,runs=runs)))
    out=tmp_path/"final"
    monkeypatch.setattr(sys,"argv",["qualify","--development",str(source),"--output-dir",str(out)])
    def forbidden(*args,**kwargs): raise AssertionError("Final seed was consumed")
    monkeypatch.setattr(qualify_gpu_simplex.subprocess,"run",forbidden)
    with pytest.raises(RuntimeError,match="Full development|Complete accepted"):
        qualify_gpu_simplex.main()
    assert not out.exists()
