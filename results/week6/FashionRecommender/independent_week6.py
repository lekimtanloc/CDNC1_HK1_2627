"""Independent implementation: no imports from the primary or historical metric code."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='4'
import csv,json,math,hashlib,sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];W=R/'results/week4/FashionRecommender';F=R/'results/week5/FashionRecommender';C=F/'addendum_5b/frame_control_5c';D=R/'results/week2/FashionRecommender/data_protocol'
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def write(p,rs):
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main(split):
 folder=O/split;rows=read(D/f'{split}.csv');gs=[r for r in rows if r['role']=='gallery'];gt=defaultdict(set)
 for r in read(D/f'ground_truth_{split}.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
 queries=list(gt);product={r['image_path']:r['item_id'] for r in rows};pop={r['product_id']:float(r['popularity']) for r in read(W/'artifacts_completed/popularity.csv')};t={r['query_id']:float(r['t_q']) for r in read(F/f't_q_{split}.csv')}
 metrics=read(O/f'metrics_{split}.csv');paired=read(O/f'paired_{split}.csv');checks=[];rec=[];arrays={};n=len(queries)
 rng=np.random.default_rng(42);indices=rng.integers(0,n,(10000,n))
 def bs(x):return np.array([x[z].mean() for z in indices])
 for path in sorted(folder.glob('*_top10.csv')):
  tokens=path.stem.split('_');base=tokens[1];p=int(tokens[2][1:])/10;g=int(tokens[3][1:]);rank=defaultdict(list)
  for r in read(path):rank[r['query_id']].append(r)
  vals=defaultdict(list)
  for q in queries:
   ordered=sorted(rank[q],key=lambda x:int(x['rank']));ids=[r['candidate_id'] for r in ordered];assert len(ids)==len(set(ids))==10 and q not in ids
   positions=[j+1 for j,img in enumerate(ids) if img in gt[q]];nr=len(gt[q]);data={'Top-1':float(1 in positions),'MRR@10':1/min(positions) if positions else 0.}
   for k in (1,5,10):data[f'Recall@{k}']=sum(r<=k for r in positions)/nr
   for k in (5,10):
    data[f'NDCG@{k}']=sum(1/math.log2(r+1) for r in positions if r<=k)/sum(1/math.log2(r+1) for r in range(1,min(k,nr)+1))
    pp=set(product[i] for i in ids[:k]);data[f'MAE@{k}']=sum(abs(pop[i]-t[q]) for i in pp)/len(pp)
   for m,v in data.items():vals[m].append(v)
  vals={m:np.array(v) for m,v in vals.items()};arrays[(base,p,g)]=vals
  saved={r['query_id']:r for r in read(path.with_name(path.name.replace('_top10','_per_query')))}
  err=max(abs(float(saved[q][m])-vals[m][j]) for j,q in enumerate(queries) for m in vals);assert err<1e-12
  for m,v in vals.items():
   row=next(r for r in metrics if r['base']==base and float(r['wp'])==p and int(r['gamma'])==g and r['metric']==m)
   lo,hi=np.percentile(bs(v),[2.5,97.5]);diff=max(abs(v.mean()-float(row['value'])),abs(lo-float(row['ci_low'])),abs(hi-float(row['ci_high'])));assert diff<1e-12
   rec.append({'split':split,'base':base,'wp':p,'gamma':g,'metric':m,'independent_value':v.mean(),'independent_ci_low':lo,'independent_ci_high':hi,'max_error':diff,'queries':n,'gallery':len(gs)})
  full_rank=(split=='valid' and (p,g) in [(0.,2),(.3,2),(.5,4)]) or (split=='test' and base=='A')
  if full_rank:
   if base=='A':vp=W/'artifacts_completed/fixed_all.npy';ip=W/'artifacts_completed/fixed_all_index.csv';col='vector_row'
   elif base=='B':vp=W/'v2/artifacts_v2/fixed_all_v2.npy';ip=W/'v2/artifacts_v2/fixed_all_v2_index.csv';col='vector_row'
   else:vp=C/'vectors_G2_5c.npy';ip=F/'addendum_5b/vectors_index_5b.csv';col='local_vector_row'
   vectors=np.load(vp,mmap_mode='r');index={r['image_path']:int(r[col]) for r in read(ip)}
   dot=np.matmul(vectors[[index[q] for q in queries]],vectors[[index[r['image_path']] for r in gs]].T)
   maxscore=0.
   for j,q in enumerate(queries):
    # Python scalar scoring and stable tuple sort, independent of primary NumPy argsort/fusion.
    values=[float(dot[j,i]) if p==0 else (1-p)*float(dot[j,i])+p*(1-abs(pop[r['item_id']]-t[q])**g) for i,r in enumerate(gs)]
    order=sorted((i for i,r in enumerate(gs) if r['image_path']!=q),key=lambda i:(-values[i],i))[:10]
    old=sorted(rank[q],key=lambda r:int(r['rank']));assert [gs[i]['image_path'] for i in order]==[r['candidate_id'] for r in old],('STOP independent ranking',base,p,g,q)
    maxscore=max(maxscore,max(abs(values[i]-float(r['score'])) for i,r in zip(order,old)))
   assert maxscore<1e-12
  checks.append({'split':split,'base':base,'wp':p,'gamma':g,'queries':n,'gallery':len(gs),'per_query_max_error':err,'ranking_recomputed':full_rank,'metric_ci_verified':True})
  print('independent',split,base,p,g,flush=True)
 for (base,p,g),vals in arrays.items():
  for m,v in vals.items():
   delta=v-arrays[(base,0.,2)][m];lo,hi=np.percentile(bs(delta),[2.5,97.5]);r=next(r for r in paired if r['base']==base and float(r['wp'])==p and int(r['gamma'])==g and r['metric']==m)
   assert max(abs(delta.mean()-float(r['difference'])),abs(lo-float(r['ci_low'])),abs(hi-float(r['ci_high'])))<1e-12
 write(O/f'independent_{split}_checks.csv',checks);write(O/f'independent_{split}_numbers.csv',rec)
 with (O/f'independent_{split}_passed.json').open('x') as f:json.dump({'timestamp':datetime.now(timezone.utc).isoformat(),'all_metrics_and_paired_ci':True,'ranking_points':sum(r['ranking_recomputed'] for r in checks),'configs':len(checks)},f,indent=2)
if __name__=='__main__':main(sys.argv[1])
