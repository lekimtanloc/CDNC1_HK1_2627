"""Independent rank-based metrics, bootstrap and exact binomial checks. No main metric import."""
import sys,math
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from common_5b import *
split=sys.argv[1];gt=defaultdict(set)
for r in read(O/'ground_truth_dev_5b.csv' if split=='dev' else P/f'ground_truth_{split}.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
qs=list(gt);summary=read(O/f'metrics_{split}_5b.csv');mainpair=json.loads((O/f'paired_{split}_5b.json').read_text())['pairs'];checks=[];scores={};boot={}
def check(c,scope,q,m,x,y):checks.append({'split':split,'variant':c,'scope':scope,'query_id':q,'metric':m,'main':x,'independent':y,'match':abs(x-y)<1e-12,'analysis_note':NOTE})
for c in sorted({r['variant'] for r in summary}):
 rankings=defaultdict(list)
 for r in read(O/f'{split}_{c}_top10_5b.csv'):rankings[r['query_id']].append(r)
 main={r['query_id']:r for r in read(O/f'{split}_{c}_per_query_5b.csv')};per=[]
 for q in qs:
  hits=sorted(int(r['rank']) for r in rankings[q] if r['candidate_id'] in gt[q]);s={'Top-1':float(1 in hits),'MRR@10':1/hits[0] if hits else 0.}
  for k in (1,5,10):s[f'Recall@{k}']=sum(p<=k for p in hits)/len(gt[q])
  for k in (5,10):s[f'NDCG@{k}']=math.fsum(math.log(2)/math.log(p+1) for p in hits if p<=k)/math.fsum(math.log(2)/math.log(j+1) for j in range(1,min(k,len(gt[q]))+1))
  per.append(s)
  for m,v in s.items():check(c,'query',q,m,float(main[q][m]),v)
 scores[c]={m:np.array([r[m] for r in per]) for m in per[0]};boot[c]={}
 for m,vals in scores[c].items():
  rng=np.random.default_rng(42);boot[c][m]=np.concatenate([vals[rng.integers(0,len(qs),(100,len(qs)))].mean(axis=1) for _ in range(100)]);lo,hi=np.quantile(boot[c][m],[.025,.975]);r=next(r for r in summary if r['variant']==c and r['metric']==m)
  for key,v in [('value',math.fsum(vals)/len(qs)),('ci_low',lo),('ci_high',hi)]:check(c,'aggregate','',m+' '+key,float(r[key]),float(v))
for label,p in mainpair.items():
 a,b=label.split('-')
 for m in ('Recall@5','NDCG@5','MRR@10'):
  d=scores[a][m]-scores[b][m];lo,hi=np.quantile(boot[a][m]-boot[b][m],[.025,.975])
  for key,val in [('difference',math.fsum(d)/len(d)),('ci_low',lo),('ci_high',hi)]:check(label,'paired','',m+' '+key,p[m][key],float(val))
 t=[[sum(x==i and y==j for x,y in zip(scores[b]['Top-1'],scores[a]['Top-1'])) for j in (0,1)] for i in (0,1)];n=t[0][1]+t[1][0];pv=binomtest(t[0][1],n,.5).pvalue if n else 1.
 check(label,'paired','','McNemar p',p['McNemar']['p_exact_two_sided'],pv)
 for i in (0,1):
  for j in (0,1):check(label,'paired','',f'McNemar cell {i},{j}',p['McNemar']['table_rows_subtrahend_cols_minuend'][i][j],t[i][j])
write(O/f'independent_{split}_checks_5b.csv',checks);bad=sum(not r['match'] for r in checks);dump(O/f'independent_{split}_result_5b.json',{'queries_checked':len(qs),'checks':len(checks),'mismatches':bad,'note':NOTE});assert not bad,'STOP independent checks mismatch'
print('PASS',split,len(qs),'queries',len(checks),'checks',flush=True)
