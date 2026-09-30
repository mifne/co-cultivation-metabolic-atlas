"""Bounded calculation jobs and explicit, model-bound viewer snapshots."""
import json
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class JobQueue:
    def __init__(self, calculate, limit=64):
        self.calculate = calculate
        self.limit = limit
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='atlas-fba')
        self.jobs = OrderedDict()
        self.tickets = OrderedDict()

    def submit(self, request):
        key = json.dumps(request, sort_keys=True, separators=(',', ':'))
        with self.lock:
            # Tickets are client subscriptions: cancelling one never cancels another.
            for ticket, entry in list(self.tickets.items()):
                if time.monotonic() - entry['created'] > 1800:
                    self._cancel(ticket)
                    del self.tickets[ticket]
            for k, job in list(self.jobs.items()):
                if job['status'] in ('done', 'cancelled', 'failed') and not job['tickets']:
                    del self.jobs[k]
            if len(self.jobs) >= self.limit:
                for old_key, old_job in list(self.jobs.items()):
                    if old_key != key and old_job['status'] in ('done', 'failed', 'cancelled'):
                        for ticket in old_job['tickets']:
                            self.tickets.pop(ticket, None)
                        del self.jobs[old_key]
                        break
            job = self.jobs.get(key)
            if job and job['status'] == 'cancelled':
                job = None
            if len(self.tickets) >= 256 or (not job and len(self.jobs) >= self.limit):
                raise ValueError('計算待ちが上限です。完了後に再試行してください。')
            if not job:
                job = {'status': 'queued', 'tickets': set(), 'result': None, 'request': request}
                self.jobs[key] = job
            ticket = uuid.uuid4().hex
            self.tickets[ticket] = {'job': job, 'created': time.monotonic(), 'cancelled': False}
            job['tickets'].add(ticket)
            if 'future' not in job:
                job['future'] = self.executor.submit(self._run, job)
            return {'id': ticket, 'status': job['status']}

    def _run(self, job):
        with self.lock:
            if not job['tickets']:
                job['status'] = 'cancelled'
                return
            job['status'] = 'running'
        try:
            result = self.calculate(job['request'])
            with self.lock:
                job['result'] = result
                job['status'] = 'done' if job['tickets'] else 'cancelled'
        except Exception:
            with self.lock:
                job['result'] = {'status': 'unknown', 'message': '計算サービスでエラーが発生しました'}
                job['status'] = 'failed'

    def get(self, ticket):
        with self.lock:
            entry = self.tickets.get(ticket)
            if not entry:
                raise KeyError(ticket)
            job = entry['job']
            status = 'cancelled' if entry['cancelled'] else job['status']
            result = {'id': ticket, 'status': status}
            if status in ('done', 'failed'):
                result['result'] = job['result']
            return result

    def _cancel(self, ticket):
        entry = self.tickets.get(ticket)
        if not entry:
            raise KeyError(ticket)
        entry['cancelled'] = True
        job = entry['job']
        job['tickets'].discard(ticket)
        if not job['tickets'] and job['status'] == 'queued':
            job['future'].cancel()
            job['status'] = 'cancelled'
        return {'status': 'cancelled', 'running_calculation_finishes': job['status'] == 'running'}

    def cancel(self, ticket):
        with self.lock:
            return self._cancel(ticket)

class SessionStore:
    def __init__(self, folder, fingerprint):
        self.folder = Path(folder)
        self.fingerprint = fingerprint
        self.lock = threading.Lock()

    def path(self, identifier):
        if len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('保存IDが不正です')
        return self.folder / (identifier + '.json')

    def save(self, payload):
        if not isinstance(payload, dict) or payload.get('schema') != 1 or payload.get('model') != self.fingerprint():
            raise ValueError('モデル指紋が異なるか、保存形式が不正です')
        if not isinstance(payload.get('view'), dict):
            raise ValueError('表示状態がありません')
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 2_000_000:
            raise ValueError('保存データが大きすぎます')
        with self.lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            if len(list(self.folder.glob('*.json'))) >= 100:
                raise ValueError('保存数の上限（100件）です。JSON書き出しを使用してください。')
            identifier = uuid.uuid4().hex
            file = self.path(identifier)
            temporary = file.with_suffix('.tmp')
            temporary.write_text(encoded, encoding='utf-8')
            temporary.replace(file)
        return {'id': identifier, 'status': 'saved'}

    def load(self, identifier):
        payload = json.loads(self.path(identifier).read_text(encoding='utf-8'))
        if payload.get('model') != self.fingerprint():
            raise ValueError('保存時と現在のモデルが異なります')
        return payload

    def list(self):
        return [{'id': p.stem, 'saved_at': p.stat().st_mtime} for p in sorted(self.folder.glob('*.json'), key=lambda p:p.stat().st_mtime, reverse=True)]
