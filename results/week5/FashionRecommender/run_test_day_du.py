"""One official test evaluation, exclusive output creation, frozen inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import csv,json,math,hashlib,datetime,time,platform
from pathlib import Path
from collections import defaultdict
import numpy as np
from metrics import score
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'results/week2/FashionRecommender/data_protocol';W=R/'results/week4/FashionRecommender'
configs=['B0','B2p','B2-v1','B2-v2']
WARN='B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân.'
def read(p):return list(csv.DictReader(p.open()))
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def write(p,rs):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def truth(s):
 d=defaultdict(set)
 for r in read(P/f'ground_truth_{s}.csv'):d[r['query_image_path']].add(r['relevant_gallery_path'])
 return d
def pair(a,b,ix):
 out={}
 for m in ('Recall@5','NDCG@5','MRR@10'):
  d=a[m]-b[m];lo,hi=np.percentile(d[ix].mean(1),[2.5,97.5]);out[m]={'difference':float(d.mean()),'ci_low':float(lo),'ci_high':float(hi)}
 x=b['Top-1'];y=a['Top-1'];t=[[int(((x==i)&(y==j)).sum()) for j in (0,1)] for i in (0,1)];n=t[0][1]+t[1][0];k=min(t[0][1],t[1][0]);pv=min(1,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1
 out['McNemar']={'table_rows_subtrahend_cols_minuend_0_1':t,'p_exact_two_sided':pv};return out
if __name__=='__main__':
 start=time.monotonic();freeze=json.loads((O/'freeze.json').read_text());assert sha(O/'protocol_v1.md')==freeze['protocol_sha256'];assert sha(O/'user_decision.json')==freeze['user_decision_sha256'];assert sha(O/'category_rule_v2.json')==freeze['category_rule_sha256']
 dump(O/'test_run_started_day_du.json',{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(O/'freeze.json'),'official_run':1})
 gt=truth('test');queries=list(gt);assert len(queries)==586
 rr=read(P/'test.csv');prod={r['image_path']:r['item_id'] for r in rr};gallery={r['image_path'] for r in rr if r['role']=='gallery'}
 pop={r['product_id']:float(r['popularity']) for r in read(W/'artifacts_completed/popularity.csv')};t={q:int.from_bytes(hashlib.sha256(f'42:{q}'.encode()).digest()[:8],'big')/2**64 for q in queries};write(O/'t_q_test.csv',[{'query_id':q,'t_q':t[q]} for q in queries])
 ix=np.random.default_rng(42).integers(0,len(queries),(10000,len(queries)));summary=[]
 for c in configs:
  source=freeze['test_top10'][c];p=R/source['path'];assert sha(p)==source['sha256'];rows=read(p);assert len(rows)==source['rows']==5860
  rankings=defaultdict(list)
  for r in sorted(rows,key=lambda r:(r['query_id'],int(r['rank']))):rankings[r['query_id']].append(r)
  assert set(rankings)==set(gt);per=[]
  for q in queries:
   candidates=[r['candidate_id'] for r in rankings[q]];assert len(candidates)==len(set(candidates))==10 and set(candidates)<=gallery and q not in candidates
   assert [int(r['rank']) for r in rankings[q]]==list(range(1,11))
   assert all(int(r['is_correct'])==int(r['candidate_id'] in gt[q]) and r['product_id']==prod[r['candidate_id']] for r in rankings[q])
   s=score(candidates,gt[q]);products={prod[i] for i in candidates};s['Popularity MAE']=sum(abs(pop[p]-t[q]) for p in sorted(products))/len(products);per.append({'query_id':q,**s})
  write(O/f'test_{c}_per_query_day_du.csv',per)
  for m in per[0]:
   if m=='query_id':continue
   a=np.array([r[m] for r in per]);lo,hi=np.percentile(a[ix].mean(1),[2.5,97.5]);summary.append({'split':'test','config':c,'metric':m,'value':float(a.mean()),'ci_low':float(lo),'ci_high':float(hi),'semantic_note':WARN})
 write(O/'metrics_test_day_du.csv',summary)
 pairs={};baselines=[]
 for split in ('valid','test'):
  gt=truth(split);queries=list(gt);ix=np.random.default_rng(42).integers(0,len(queries),(10000,len(queries)));scores={}
  for c in configs:
   p=O/(f'valid_{c}_per_query.csv' if split=='valid' else f'test_{c}_per_query_day_du.csv');data={r['query_id']:r for r in read(p)};scores[c]={m:np.array([float(data[q][m]) for q in queries]) for m in data[queries[0]] if m!='query_id'}
  pairs[split]={'semantic_note':WARN}
  for a,b in [('B2-v2','B2-v1'),('B2-v1','B2p'),('B2-v2','B2p')]:pairs[split][a+' minus '+b]=pair(scores[a],scores[b],ix)
  rr=read(P/f'{split}.csv');products={r['item_id'] for r in rr if r['role']=='gallery'};imageprod={r['image_path']:r['item_id'] for r in rr};G=sum(r['role']=='gallery' for r in rr);Q=len(gt);errors=[]
  for q in queries:
   tq=int.from_bytes(hashlib.sha256(f'42:{q}'.encode()).digest()[:8],'big')/2**64;ps=products-{imageprod[q]};errors.append({'query_id':q,'gallery_products_excluding_query_product':len(ps),'mae_reference':sum(abs(pop[p]-tq) for p in sorted(ps))/len(ps)})
  write(O/f'popularity_reference_{split}_day_du.csv',errors)
  baselines.append({'split':split,'queries':Q,'gallery_images':G,'ground_truth_pairs':sum(map(len,gt.values())),'random_Top-1':sum(map(len,gt.values()))/(Q*G),'random_Recall@5':5/G,'B0_Top-1':float(scores['B0']['Top-1'].mean()),'B0_Recall@5':float(scores['B0']['Recall@5'].mean()),'popularity_mae_reference':sum(r['mae_reference'] for r in errors)/Q,'B0_Popularity_MAE':float(scores['B0']['Popularity MAE'].mean()),'comparison':'descriptive, no significance test against random/reference','semantic_note':WARN})
 dump(O/'paired_day_du.json',pairs);write(O/'reference_levels_day_du.csv',baselines)
 import torch,torchvision,PIL
 dump(O/'test_run_completed_day_du.json',{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'official_test_runs':1,'versions':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'torchvision':torchvision.__version__,'Pillow':PIL.__version__},'peak_ram':'chưa kiểm tra được','semantic_note':WARN})
 print('Official test complete; independent check required before proceeding')
