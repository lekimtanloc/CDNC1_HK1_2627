"""Separate implementation from CSV ranks and truth; no import of metric/evaluation module."""
import csv,json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'results/week2/FashionRecommender/data_protocol';W=R/'results/week4/FashionRecommender'
def read(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
freeze=json.loads((O/'freeze.json').read_text());paired=json.loads((O/'paired_day_du.json').read_text());pop={r['product_id']:float(r['popularity']) for r in read(W/'artifacts_completed/popularity.csv')};checks=[];independent=[]
def check(split,config,scope,q,m,a,b):checks.append({'split':split,'config':config,'scope':scope,'query_id':q,'metric':m,'main':a,'independent':b,'match':abs(a-b)<1e-12})
for split in ('valid','test'):
 gt=defaultdict(set)
 for r in read(P/f'ground_truth_{split}.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
 qs=list(gt);scores={};boots={};rng=np.random.default_rng(42)
 # Independently regenerate the same bootstrap sample sequence in chunks.
 chunks=[rng.integers(0,len(qs),(100,len(qs))) for _ in range(100)]
 for c in ('B0','B2p','B2-v1','B2-v2'):
  path=O/f'valid_{c}_top10.csv' if split=='valid' else R/freeze['test_top10'][c]['path'];rankings=defaultdict(list)
  for r in read(path):rankings[r['query_id']].append(r)
  main={r['query_id']:r for r in read(O/(f'valid_{c}_per_query.csv' if split=='valid' else f'test_{c}_per_query_day_du.csv'))};rows=[]
  for q in qs:
   rr=rankings[q];positions=sorted(int(r['rank']) for r in rr if r['candidate_id'] in gt[q]);s={'Top-1':float(1 in positions),'MRR@10':1/positions[0] if positions else 0.}
   for k in (1,5,10):s[f'Recall@{k}']=len([v for v in positions if v<=k])/len(gt[q])
   for k in (5,10):s[f'NDCG@{k}']=sum(math.log(2)/math.log(v+1) for v in positions if v<=k)/sum(math.log(2)/math.log(v+1) for v in range(1,min(k,len(gt[q]))+1))
   tq=int(hashlib.sha256(('42:'+q).encode()).hexdigest()[:16],16)/(2**64);ps=set(r['product_id'] for r in rr);s['Popularity MAE']=math.fsum(abs(pop[p]-tq) for p in ps)/len(ps)
   rows.append(s);independent.append({'split':split,'config':c,'query_id':q,**s})
   for m,v in s.items():check(split,c,'per_query',q,m,float(main[q][m]),v)
  scores[c]={m:np.array([r[m] for r in rows]) for m in rows[0]};boots[c]={}
  summary=read(O/('metrics_valid.csv' if split=='valid' else 'metrics_test_day_du.csv'))
  for m,vals in scores[c].items():
   mean=math.fsum(vals)/len(vals);boot=np.concatenate([vals[ix].mean(axis=1) for ix in chunks]);boots[c][m]=boot;lo,hi=np.quantile(boot,[.025,.975]);r=next(r for r in summary if r['config']==c and r['metric']==m)
   for key,v in [('value',mean),('ci_low',lo),('ci_high',hi)]:check(split,c,'aggregate','',m+' '+key,float(r[key]),float(v))
 for a,b in [('B2-v2','B2-v1'),('B2-v1','B2p'),('B2-v2','B2p')]:
  label=a+' minus '+b;main=paired[split][label]
  for m in ('Recall@5','NDCG@5','MRR@10'):
   ci=np.quantile(boots[a][m]-boots[b][m],[.025,.975]);mean=math.fsum(scores[a][m]-scores[b][m])/len(qs)
   for key,v in [('difference',mean),('ci_low',ci[0]),('ci_high',ci[1])]:check(split,label,'paired','',m+' '+key,main[m][key],float(v))
  t=[[sum(x==i and y==j for x,y in zip(scores[b]['Top-1'],scores[a]['Top-1'])) for j in (0,1)] for i in (0,1)];n=t[0][1]+t[1][0];pv=binomtest(t[0][1],n,.5).pvalue if n else 1.
  check(split,label,'paired','','McNemar p',main['McNemar']['p_exact_two_sided'],pv)
  for i in (0,1):
   for j in (0,1):check(split,label,'paired','',f'McNemar cell {i},{j}',main['McNemar']['table_rows_subtrahend_cols_minuend_0_1'][i][j],t[i][j])
 rr=read(P/f'{split}.csv');gallery=[r for r in rr if r['role']=='gallery'];gprod=set(r['item_id'] for r in gallery);ip={r['image_path']:r['item_id'] for r in rr};ref=next(r for r in read(O/'reference_levels_day_du.csv') if r['split']==split);errs=[]
 for q in qs:
  tq=int(hashlib.sha256(('42:'+q).encode()).hexdigest()[:16],16)/2**64;pool=gprod-{ip[q]};errs.append(math.fsum(abs(pop[p]-tq) for p in pool)/len(pool))
 for m,v in [('random_Top-1',sum(len(gt[q]) for q in qs)/len(qs)/len(gallery)),('random_Recall@5',5/len(gallery)),('popularity_mae_reference',math.fsum(errs)/len(errs))]:check(split,'reference','reference','',m,float(ref[m]),v)
write(O/'independent_test_checks.csv',checks);write(O/'independent_scores_day_du.csv',independent)
bad=[r for r in checks if not r['match']]
with (O/'independent_result_day_du.json').open('x') as f:json.dump({'checks':len(checks),'test_checks':sum(r['split']=='test' for r in checks),'mismatches':len(bad),'tolerance':1e-12,'status':'PASS' if not bad else 'STOP'},f,indent=2)
assert not bad,f'STOP independent mismatch: {bad[:3]}'
print('PASS',len(checks),'checks; official test was not rerun')
