"""Accept only a successful, hash-bound P0 reproduction."""
from pathlib import Path
import hashlib,json
b=Path(__file__).resolve().parent/'work/pilot44-fixed-20261110'
r=b/'p0-run/result.json';a=b/'p0-audit.json'
result=json.loads(r.read_text(encoding='utf-8'));audit=json.loads(a.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert result['status']=='CONSTRUCTION_GATE_PASS' and audit['status']=='AUDIT_PASS'
assert audit['result_sha256']==sha(r)
with (b/'acceptance.json').open('x',encoding='utf-8') as f:
 json.dump(dict(status='ACCEPTED_P0_ONLY',result_sha256=sha(r),audit_sha256=sha(a)),f,indent=2)
print('P0 accepted; P1 remains a separate command.')
