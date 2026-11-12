import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import sys,math,time
from collections import defaultdict
import numpy as np
from common_5c import *
sys.path.insert(0,str(PRIOR));from metrics import score
METRICS=['Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','NDCG@10','MRR@10']
def bootstrap(a):
 rng=np.random.default_rng(42);return np.concatenate([a[rng.integers(0,len(a),(100,len(a)))].mean(1) for _ in range(100)])
def pair(a,b):
 out={}
 for m in ('Recall@5','NDCG@5','MRR@10'):
  d=a[m]-b[m];lo,hi=np.percentile(bootstrap(d),[2.5,97.5]);out[m]={'difference':float(d.mean()),'ci_low':float(lo),'ci_high':float(hi)}
 t=[[int(((b['Top-1']==i)&(a['Top-1']==j)).sum()) for j in (0,1)] for i in (0,1)];n=t[0][1]+t[1][0];k=min(t[0][1],t[1][0]);p=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.;out['McNemar']={'table_rows_subtrahend_cols_minuend':t,'p_exact_two_sided':p};return out
def interpret(p,labels):
 signs=[1 if p[k]['Recall@5']['ci_low']>0 else (-1 if p[k]['Recall@5']['ci_high']<0 else 0) for k in labels];yes=signs[0]!=0 and signs[0]==signs[1];return {'comparisons':labels,'conclusion':'có' if yes else 'không kết luận','direction':('positive' if signs[0]>0 else 'negative') if yes else None,'note':NOTE}
def run(split):
 assert split in ('valid','test');assert json.loads((O/'G0_byte_result_5c.json').read_text())['passed'];start=time.monotonic()
 if split=='test':
  frozen=json.loads((O/'freeze_5c.json').read_text())
  for p,h in frozen['hashes'].items():assert sha(R/p)==h,p
  dump(O/'test_started_5c.json',{'timestamp':now(),'freeze_sha256':sha(O/'freeze_5c.json'),'note':NOTE})
 rows=read(P/f'{split}.csv');qs=[r for r in rows if r['role']=='query'];gs=[r for r in rows if r['role']=='gallery'];gt=defaultdict(set)
 for r in read(P/f'ground_truth_{split}.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
 queries=list(gt);assert len({r['item_id'] for r in qs})==len(qs)==len(gt);idx={r['image_path']:int(r['local_vector_row']) for r in read(BASE/'vectors_index_5b.csv')};all_scores={};summary=[]
 for c in VARIANTS+['B','F']:
  if c in ('B','F'):rankrows=read(BASE/f'{split}_{c}_top10_5b.csv')
  else:
   a=np.load(BASE/'vectors_C_5b.npy' if c=='G0' else O/f'vectors_{c}_5c.npy',mmap_mode='r');sim=a[[idx[r['image_path']] for r in qs]]@a[[idx[r['image_path']] for r in gs]].T;rankrows=[]
   for qi,q in enumerate(qs):
    order=[int(i) for i in np.argsort(-sim[qi],kind='stable') if gs[int(i)]['image_path']!=q['image_path']][:10]
    for k,i in enumerate(order,1):rankrows.append({'query_id':q['image_path'],'rank':k,'candidate_id':gs[i]['image_path'],'product_id':gs[i]['item_id'],'score':float(sim[qi,i]),'is_correct':int(gs[i]['image_path'] in gt[q['image_path']])})
  rank=defaultdict(list)
  for r in sorted(rankrows,key=lambda r:(r['query_id'],int(r['rank']))):rank[r['query_id']].append(r['candidate_id'])
  assert set(rank)==set(gt) and all(len(a)==len(set(a))==10 for a in rank.values());write(O/f'{split}_{c}_top10_5c.csv',[dict(r,analysis_note=NOTE) for r in rankrows]);per=[{'query_id':q,**{m:v for m,v in score(rank[q],gt[q]).items() if m in METRICS}} for q in queries];write(O/f'{split}_{c}_per_query_5c.csv',[dict(r,analysis_note=NOTE) for r in per]);all_scores[c]={m:np.array([r[m] for r in per]) for m in METRICS}
  for m,arr in all_scores[c].items():
   lo,hi=np.percentile(bootstrap(arr),[2.5,97.5]);summary.append({'split':split,'variant':c,'metric':m,'value':float(arr.mean()),'ci_low':float(lo),'ci_high':float(hi),'analysis_note':NOTE})
  print(split,c,'done',flush=True)
 write(O/f'metrics_{split}_5c.csv',summary);paired={a+'-'+b:pair(all_scores[a],all_scores[b]) for a,b in PAIRS};dump(O/f'paired_{split}_5c.json',{'note':NOTE,'pairs':paired});dump(O/f'effects_{split}_5c.json',{'face_hair':interpret(paired,['G0-G1','G2-G4']),'background':interpret(paired,['G2-G0','G4-G1']),'frame':'G1-B and G3-F numbers/CIs only, no threshold','G3':'descriptive vsG0 and random; no person identity claims','note':NOTE})
 dump(O/f'summary_{split}_5c.json',{'timestamp':now(),'queries':len(qs),'gallery':len(gs),'truth_pairs':sum(map(len,gt.values())),'random_Top-1':sum(map(len,gt.values()))/(len(qs)*len(gs)),'random_Recall@5':5/len(gs),'seconds':time.monotonic()-start,'test_passes':1 if split=='test' else 0,'note':NOTE})
if __name__=='__main__':run(sys.argv[1])
