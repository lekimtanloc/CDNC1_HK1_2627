import sys,json,csv,hashlib
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=O.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert not (O/'test_started_week6.json').exists()
for n in ['completed_valid.json','completed_oracle.json','independent_valid_passed.json','pre_freeze_old_hash_check.json']:assert (O/n).is_file()
assert sha(O/'protocol_week6.md')==(O/'protocol_week6.sha256').read_text().split()[0]
rows=list(csv.DictReader((O/'operating_points.csv').open()));configs={}
for b in ['A','B','G2']:
 r=next(r for r in rows if r['base']==b and float(r['threshold_pp'])==2);p=float(r['wp']);g=int(r['gamma']);configs[b]=[[0.,2],[.3,2]]
 if p and [p,g] not in configs[b]:configs[b].append([p,g])
hashes=json.loads((O/'hash_cu_truoc_tuan06.json').read_text())
# Freeze direct inputs plus the full historical snapshot. Test verifies all before its marker.
for p in list(O.glob('*.py'))+list(O.glob('*.md'))+list((O/'valid').glob('*_top10.csv'))+[O/'operating_points.csv',O/'user_decision_week6.json',O/'protocol_week6.sha256']:
 hashes[str(p.relative_to(R))]=sha(p)
with (O/'freeze_week6.json').open('x') as f:json.dump({'timestamp':datetime.now(timezone.utc).isoformat(),'configs':configs,'main_base':'A','operating_threshold_pp':2,'named_P':'P của paper, đã bỏ category; wp=.3 gamma2','hashes':hashes,'test_has_prior_exposure':True,'oracle_not_used_for_selection':True},f,ensure_ascii=False,indent=2)
print(configs)
