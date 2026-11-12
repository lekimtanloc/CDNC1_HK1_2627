import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import math,datetime,sys
import numpy as np
from prepare import O,R,P,W,read,write,dump,sha
from sanity import truth,ranks,paths
from metrics import score
from week3_retrieval_metrics import evaluate as legacy
A=W/'artifacts_completed';V=W/'v2/artifacts_v2'
configs=['B0','B2p','B2-v1','B2-v2'];primary=['Top-1','Recall@5','NDCG@5','MRR@10']
pop={r['product_id']:float(r['popularity']) for r in read(A/'popularity.csv')}
def rank_validation():
 rr=read(P/'valid.csv');qs=[r for r in rr if r['role']=='query'];gs=[r for r in rr if r['role']=='gallery'];gt=truth('valid')
 sources={'B2p':(O/'old_valid.npy',O/'old_valid_index.csv'),'B2-v1':(A/'fixed_all.npy',A/'fixed_all_index.csv'),'B2-v2':(V/'fixed_all_v2.npy',V/'fixed_all_v2_index.csv')}
 for c in configs:
  if c=='B0':scores=np.tile([pop[r['item_id']] for r in gs],(len(qs),1))
  else:
   p,index=sources[c];a=np.load(p,mmap_mode='r');idx={r['image_path']:int(r['vector_row']) for r in read(index)}
   scores=a[[idx[r['image_path']] for r in qs]]@a[[idx[r['image_path']] for r in gs]].T
  out=[]
  for j,q in enumerate(qs):
   order=sorted(range(len(gs)),key=lambda i:(-scores[j,i],__import__('hashlib').sha256(f"42:{gs[i]['image_path']}".encode()).hexdigest())) if c=='B0' else np.argsort(-scores[j],kind='stable')
   order=[i for i in order if gs[i]['image_path']!=q['image_path']][:10]
   for k,i in enumerate(order,1):out.append({'query_id':q['image_path'],'rank':k,'candidate_id':gs[i]['image_path'],'product_id':gs[i]['item_id'],'score':float(scores[j,i]),'is_correct':int(gs[i]['image_path'] in gt[q['image_path']])})
  write(O/f'valid_{c}_top10.csv',out)
def compute(split):
 gt=truth(split);queries=list(gt);rr=read(P/f'{split}.csv');imageprod={r['image_path']:r['item_id'] for r in rr}
 assert len({imageprod[q] for q in queries})==len(queries)
 t={q:int.from_bytes(__import__('hashlib').sha256(f'42:{q}'.encode()).digest()[:8],'big')/2**64 for q in queries};write(O/f't_q_{split}.csv',[{'query_id':q,'t_q':t[q]} for q in queries])
 rng=np.random.default_rng(42);ix=rng.integers(0,len(queries),size=(10000,len(queries)));summary=[];all_scores={};checks=[]
 for c in configs:
  rp=paths[c] if split=='test' else O/f'valid_{c}_top10.csv';rankings=ranks(rp);assert set(rankings)==set(gt)
  per=[]
  for q in queries:
   s=score(rankings[q],gt[q]);products={imageprod[i] for i in rankings[q][:10]};s['Popularity MAE']=sum(abs(pop[p]-t[q]) for p in products)/len(products);per.append(dict(query_id=q,**s))
  write(O/f'{split}_{c}_per_query.csv',per);all_scores[c]={m:np.array([p[m] for p in per]) for m in per[0] if m!='query_id'}
  for m,a in all_scores[c].items():
   low,high=np.percentile(a[ix].mean(axis=1),[2.5,97.5]);summary.append({'split':split,'config':c,'metric':m,'value':float(a.mean()),'ci_low':float(low),'ci_high':float(high)})
  for k in (1,5,10):
   l=legacy(rankings,gt,k);checks.append({'config':c,'k':k,'recall_difference':float(all_scores[c][f'Recall@{k}'].mean()-l['recall'])})
   assert abs(checks[-1]['recall_difference'])<1e-12
  assert abs(all_scores[c]['Top-1'].mean()-legacy(rankings,gt,1)['reciprocal_rank'])<1e-12
 write(O/f'metrics_{split}.csv',summary);dump(O/f'legacy_check_{split}.json',checks)
 paired={}
 for m in ('Recall@5','NDCG@5','MRR@10'):
  d=all_scores['B2-v2'][m]-all_scores['B2-v1'][m];lo,hi=np.percentile(d[ix].mean(axis=1),[2.5,97.5]);paired[m]={'difference_v2_minus_v1':float(d.mean()),'ci_low':float(lo),'ci_high':float(hi)}
 a=all_scores['B2-v1']['Top-1'];b=all_scores['B2-v2']['Top-1'];table=[[int(((a==i)&(b==j)).sum()) for j in (0,1)] for i in (0,1)];n=table[0][1]+table[1][0];small=min(table[0][1],table[1][0]);pv=min(1.,2*sum(math.comb(n,i) for i in range(small+1))/2**n) if n else 1.
 paired['McNemar']={'table_rows_v1_cols_v2_0_1':table,'p_exact_two_sided':pv};dump(O/f'paired_{split}.json',paired)
 if split=='valid':
  d=paired['Recall@5'];chosen='B2-v2' if d['ci_low']<=0<=d['ci_high'] or d['difference_v2_minus_v1']>0 else 'B2-v1'
  dump(O/'baseline_chinh.json',{'selected':chosen,'validation_only':True,'rule':'CI Recall@5 v2-v1 excludes zero: choose higher Recall; otherwise v2 semantic mapping','evidence':d,'requires_user_decision':chosen=='B2-v1','time':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  if chosen=='B2-v1':print('STOP: B2-v1 selected; user decision required',flush=True);return False
 return True
if __name__=='__main__':
 if sys.argv[1]=='valid':rank_validation();compute('valid')
 elif sys.argv[1]=='test':
  assert (O/'freeze.json').exists();assert not (O/'metrics_test.csv').exists();compute('test')
