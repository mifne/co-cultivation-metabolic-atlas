import tempfile, time, threading
from pathlib import Path
from atlas_store import JobQueue, SessionStore

def until(fn):
    for _ in range(200):
        if fn(): return
        time.sleep(.01)
    raise AssertionError('timed out')

def test_jobs():
    started=threading.Event(); release=threading.Event(); calls=[]
    def calc(body):
        calls.append(body['id']); started.set(); release.wait(2)
        return {'status':'optimal','value':body['id']}
    queue=JobQueue(calc,limit=3)
    a=queue.submit({'id':1}); until(started.is_set)
    b=queue.submit({'id':1}); c=queue.submit({'id':2})
    queue.cancel(a['id']); queue.cancel(c['id'])
    assert queue.get(a['id'])['status']=='cancelled'
    release.set(); until(lambda:queue.get(b['id'])['status']=='done')
    assert calls==[1],calls
    assert queue.get(b['id'])['result']['value']==1
    for i in range(3,9):
        d=queue.submit({'id':i});until(lambda:queue.get(d['id'])['status']=='done')
    assert len(queue.jobs)<=3
    queue.executor.shutdown()

def test_sessions():
    with tempfile.TemporaryDirectory() as folder:
        store=SessionStore(folder,lambda:'model-a')
        payload={'schema':1,'model':'model-a','view':{'mediumOverride':{'glu__L_e':.02},'hiddenReactions':['r_a']}}
        result=store.save(payload);assert store.load(result['id'])==payload
        assert len(store.list())==1
        for bad in ('../outside','f'*33):
            try:store.load(bad)
            except ValueError:pass
            else:raise AssertionError('invalid ID accepted')
        try:store.save({**payload,'model':'model-b'})
        except ValueError:pass
        else:raise AssertionError('wrong model accepted')
        assert len(list(Path(folder).glob('*.json')))==1

test_jobs();test_sessions();print('PASS job deduplication, independent cancellation, queued cancellation, bounded retention, session roundtrip and model checks')
