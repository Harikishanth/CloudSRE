import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import httpx
c = httpx.Client(base_url='https://dardrax-cloudsre-environment.hf.space', timeout=120)
print('Testing adversarial scenarios...')
for i in range(8):
    r = c.post('/reset', json={'task_id': 'adversarial'})
    obs = r.json().get('observation', r.json())
    sid = obs.get('scenario_id','')
    h = obs.get('service_health', {})
    broken = [(s, h2.get('status')) for s, h2 in h.items() if h2.get('status') != 'healthy']
    vis = 'VISIBLE' if broken else 'INVISIBLE'
    print(f'  {sid:50s} | {vis} broken={broken}')
    if broken:
        svc = broken[0][0]
        err = h.get(svc, {}).get('error','')
        if 'queue' in err.lower():
            cmd = 'queue drain 200'
        else:
            cmd = f'restart_service {svc}'
        r2 = c.post('/step', json={'action': {'command': 'status'}})
        r3 = c.post('/step', json={'action': {'command': cmd}})
        d3 = r3.json()
        done = d3.get('done', d3.get('observation',d3).get('done', False))
        print(f'    fix: {cmd} -> done={done}')
c.close()
