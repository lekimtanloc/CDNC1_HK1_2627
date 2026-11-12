import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import sys,time,math
from collections import defaultdict
import numpy as np
from common_5b import *
sys.path.insert(0,str(PRIOR));from metrics import score
METRICS=['Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','NDCG@10','MRR@10']
PAIRS=[('C','B'),('A','D'),('C','A'),('B','D'),('F','A')]
def rows_gt(split):
 rr=read(O/'dev_split_5b.csv' if split=='dev' else P/f'{split}.csv');gt=defaultdict(set)
 for r in read(O/'ground_truth_dev_5b.csv' if split=='dev' else P/f'ground_truth_{split}.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
 return rr,gt

def source(c):
 if c=='A':return W/'artifacts_completed/fixed_all.npy',W/'artifacts_completed/fixed_all_index.csv','vector_row'
 if c=='B':return W/'v2/artifacts_v2/fixed_all_v2.npy',W/'v2/artifacts_v2/fixed_all_v2_index.csv','vector_row'
 return O/f'vectors_{c}_5b.npy',O/'vectors_index_5b.csv','local_vector_row'

def rankings(split,c,rr,gt):
 # Rebuild rankings from cached normalized vectors for every variant.
 p,index,key=source(c);a=np.load(p,mmap_mode='r');idx={r['image_path']:int(r[key]) for r in read(index)};qs=[r for r in rr if r['role']=='query'];gs=[r for r in rr if r['role']=='gallery'];sim=a[[idx[r['image_path']] for r in qs]]@a[[idx[r['image_path']] for r in gs]].T;result=[]
 for qi,q in enumerate(qs):
  order=[int(i) for i in np.argsort(-sim[qi],kind='stable') if gs[int(i)]['image_path']!=q['image_path']][:10]
  for rank,i in enumerate(order,1):result.append({'query_id':q['image_path'],'rank':rank,'candidate_id':gs[i]['image_path'],'product_id':gs[i]['item_id'],'score':float(sim[qi,i]),'is_correct':int(gs[i]['image_path'] in gt[q['image_path']])})
 out=O/f'{split}_{c}_top10_5b.csv';write(out,[dict(r,analysis_note=NOTE) for r in result]);d=defaultdict(list)
 for r in sorted(result,key=lambda r:(r['query_id'],int(r['rank']))):d[r['query_id']].append(r['candidate_id'])
 assert set(d)==set(gt) and all(len(x)==10 and len(set(x))==10 for x in d.values());return d

def bootstrap(a):
 rng=np.random.default_rng(42);out=[]
 for i in range(0,10000,100):
  ix=rng.integers(0,len(a),(100,len(a)));out.append(a[ix].mean(1))
 return np.concatenate(out)
def paired(a,b):
 out={}
 for m in ('Recall@5','NDCG@5','MRR@10'):
  d=a[m]-b[m];ci=np.percentile(bootstrap(d),[2.5,97.5]);out[m]={'difference':float(d.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])}
 t=[[int(((b['Top-1']==i)&(a['Top-1']==j)).sum()) for j in (0,1)] for i in (0,1)];n=t[0][1]+t[1][0];k=min(t[0][1],t[1][0]);pv=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.
 out['McNemar']={'table_rows_subtrahend_cols_minuend':t,'p_exact_two_sided':pv};return out

def effect_rule(p,pairs):
 d=[p[k]['Recall@5'] for k in pairs];direction=['positive' if x['ci_low']>0 else ('negative' if x['ci_high']<0 else 'contains_zero') for x in d];present=direction[0]==direction[1] and direction[0]!='contains_zero';return {'comparisons':pairs,'conclusion':'có' if present else 'không kết luận','direction':direction[0] if present else None,'interpretation':'paired retrieval association only, not a causal conclusion','note':NOTE}

def run(split):
 start=time.monotonic();assert json.loads((O/'A_byte_result_5b.json').read_text())['passed']
 if split=='test':
  frozen=json.loads((O/'freeze_5b.json').read_text())
  for p,h in frozen['hashes'].items():assert sha(R/p)==h,p
  dump(O/'test_started_5b.json',{'timestamp':now(),'freeze_sha256':sha(O/'freeze_5b.json'),'note':NOTE})
 rr,gt=rows_gt(split);qs=list(gt);scores={};summary=[];configs=['A','B'] if split=='dev' else ['A','B','C','D','F']
 for c in configs:
  rank=rankings(split,c,rr,gt);per=[{'query_id':q,**{m:v for m,v in score(rank[q],gt[q]).items() if m in METRICS}} for q in qs];write(O/f'{split}_{c}_per_query_5b.csv',[dict(r,analysis_note=NOTE) for r in per]);scores[c]={m:np.array([r[m] for r in per]) for m in METRICS}
  for m,a in scores[c].items():
   ci=np.percentile(bootstrap(a),[2.5,97.5]);summary.append({'split':split,'variant':c,'metric':m,'value':float(a.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'analysis_note':NOTE})
  print(split,c,'complete',flush=True)
 write(O/f'metrics_{split}_5b.csv',summary);pairs=[('B','A')] if split=='dev' else PAIRS;p={a+'-'+b:paired(scores[a],scores[b]) for a,b in pairs};dump(O/f'paired_{split}_5b.json',{'note':NOTE,'pairs':p})
 if split!='dev':dump(O/f'effects_{split}_5b.json',{'face_hair':effect_rule(p,['C-B','A-D']),'pants_shoes':effect_rule(p,['C-A','B-D']),'F':'report only; no threshold, not evidence of model identity recognition'})
 Q=len(qs);G=sum(r['role']=='gallery' for r in rr);dump(O/f'summary_{split}_5b.json',{'timestamp':now(),'note':NOTE,'queries':Q,'gallery':G,'truth_pairs':sum(map(len,gt.values())),'random_Top-1':sum(map(len,gt.values()))/(Q*G),'random_Recall@5':5/G,'seconds':time.monotonic()-start,'test_evaluation_passes':1 if split=='test' else 0,'random_comparison':'descriptive only; no hypothesis test'})
if __name__=='__main__':run(sys.argv[1])
