"""Resource-bounded SQL execution in an isolated subprocess."""
import json
from pathlib import Path
import subprocess
import sys


def evaluate(types, rows, queries, timeout=4):
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
            input=json.dumps(dict(types=types, rows=rows, queries=queries)),
            text=True, capture_output=True, timeout=timeout, check=False)
        if result.returncode:
            return {'error': 'worker_exit', 'returncode': result.returncode}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}


def worker():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    from audit_sql_environment import execute
    payload = json.loads(sys.stdin.read(8 * 1024**2))
    if any(t.lower() not in ('real','text','integer','numeric') for t in payload['types']):
        raise ValueError('unsupported type')
    results=[]
    for query in payload['queries']:
        try:
            results.append({'rows':execute(payload['types'], payload['rows'], query)})
        except Exception as exc:
            results.append({'error':type(exc).__name__, 'detail':str(exc)[:200]})
    print(json.dumps({'results':results}))

if __name__ == '__main__':
    worker()
