import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import csv,json,hashlib,math
from pathlib import Path
from collections import Counter
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];D=R/'results/week2/FashionRecommender/data_protocol';F=R/'results/week5/FashionRecommender'
def read(p):return list(csv.DictReader(p.open()))
def dump(n,x):
 with (O/n).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
rows=read(D/'valid.csv');qs=[r for r in rows if r['role']=='query'];gs=[r for r in rows if r['role']=='gallery']
pop={r['product_id']:float(r['popularity']) for r in read(R/'results/week4/FashionRecommender/artifacts_completed/popularity.csv')};t={r['query_id']:float(r['t_q']) for r in read(F/'t_q_valid.csv')}
metadata=read(R/'FashionRecommender/repro_outputs_resnet50_full/metadata.csv');raw=np.random.default_rng(42).integers(0,101,len(metadata));normalized=(raw-raw.min())/(raw.max()-raw.min());assert all(pop[r['item_id']]==p for r,p in zip(metadata,normalized))
x=np.array([t[q['image_path']] for q in qs]);y=np.array([pop[q['item_id']] for q in qs]);gp=[pop[i] for i in sorted({r['item_id'] for r in gs})];freq=Counter(gp)
sourcehash={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'valid.csv',D/'ground_truth_valid.csv',F/'t_q_valid.csv',R/'results/week4/FashionRecommender/artifacts_completed/popularity.csv',R/'FashionRecommender/repro_outputs_resnet50_full/metadata.csv']}
dump('leakage_diagnostics_valid.json',{'gallery':len(gs),'queries':len(qs),'query_gallery_intersection':len({q['image_path'] for q in qs}&{g['image_path'] for g in gs}),'t_query_pop_pearson':float(np.corrcoef(x,y)[0,1]),'popularity_rng_lineage_exact':True,'gallery_products':len(gp),'unique_normpop_values':len(freq),'largest_pop_tie_products':max(freq.values()),'normpop_mean':float(np.mean(gp)),'normpop_std':float(np.std(gp)),'queries_each_unique_product':len({q['item_id'] for q in qs})==len(qs),'note':'Pearson descriptive only; exact deterministic provenance does not prove population independence. Official score code has no ground truth input except evaluation. Oracle stored separately.','hashes':sourcehash})
# Verify source immutability before freeze, including old artifacts not used directly.
before=json.loads((O/'hash_cu_truoc_tuan06.json').read_text());bad=[]
for name,h in before.items():
 p=R/name;digest=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):digest.update(b)
 if digest.hexdigest()!=h:bad.append(name)
assert not bad,bad
dump('pre_freeze_old_hash_check.json',{'files':len(before),'unchanged':True})
print('lineage and old hashes PASS')
