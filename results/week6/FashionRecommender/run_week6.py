import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='4'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import csv,json,hashlib,math,sys,platform
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];W=R/'results/week4/FashionRecommender';F=R/'results/week5/FashionRecommender';C=F/'addendum_5b/frame_control_5c';D=R/'results/week2/FashionRecommender/data_protocol'
MET=['Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','NDCG@10','MRR@10','MAE@5','MAE@10'];GRID=[(0.,2)]+[(p/10,g) for p in range(1,6) for g in (2,4)]
SOURCES={'A':(W/'artifacts_completed/fixed_all.npy',W/'artifacts_completed/fixed_all_index.csv','vector_row'),'B':(W/'v2/artifacts_v2/fixed_all_v2.npy',W/'v2/artifacts_v2/fixed_all_v2_index.csv','vector_row'),'G2':(C/'vectors_G2_5c.npy',F/'addendum_5b/vectors_index_5b.csv','local_vector_row')}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def dump(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def write(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def key(p,g):return f'p{round(p*10)}_g{g}'
def check_protocol():assert sha(O/'protocol_week6.md')==(O/'protocol_week6.sha256').read_text().split()[0]
def load(split):
 rows=read(D/f'{split}.csv');qs=[r for r in rows if r['role']=='query'];gs=[r for r in rows if r['role']=='gallery'];gt=defaultdict(set)
 for r in read(D/f'ground_truth_{split}.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
 byq={r['image_path']:r for r in qs};qs=[byq[q] for q in gt]
 assert (len(gs),len(qs))==((1597,568) if split=='valid' else (1543,586))
 pop={r['product_id']:float(r['popularity']) for r in read(W/'artifacts_completed/popularity.csv')}
 tq={r['query_id']:float(r['t_q']) for r in read(F/f't_q_{split}.csv')}
 for q in qs:assert tq[q['image_path']]==int.from_bytes(hashlib.sha256(f"42:{q['image_path']}".encode()).digest()[:8],'big')/2**64
 ip={r['image_path']:r['item_id'] for r in rows}
 for q in qs:assert all(ip[i]==q['item_id'] for i in gt[q['image_path']])
 return qs,gs,gt,pop,np.array([tq[q['image_path']] for q in qs])
def cosine(base,qs,gs):
 p,idx,col=SOURCES[base];a=np.load(p,mmap_mode='r');lookup={r['image_path']:int(r[col]) for r in read(idx)}
 return a[[lookup[q['image_path']] for q in qs]]@a[[lookup[r['image_path']] for r in gs]].T
BS={}
def bootstrap(a):
 n=len(a)
 if n not in BS:BS[n]=np.random.default_rng(42).integers(0,n,(10000,n))
 return np.concatenate([a[ix].mean(axis=1) for ix in np.array_split(BS[n],100)])
def stats(a):
 low,high=np.percentile(bootstrap(a),[2.5,97.5]);return float(a.mean()),float(low),float(high)
def evaluate(sim,qs,gs,gt,pop,t,p,g):
 gp=np.array([pop[r['item_id']] for r in gs]);score=sim if p==0 else (1-p)*sim.astype(np.float64)+p*(1-np.abs(gp[None,:]-t[:,None])**g)
 rr=[];per=[];ranks=[]
 for j,q in enumerate(qs):
  order=[int(i) for i in np.argsort(-score[j],kind='stable') if gs[i]['image_path']!=q['image_path']][:10];ranks.append(order)
  hit=np.array([gs[i]['image_path'] in gt[q['image_path']] for i in order],dtype=float);n=len(gt[q['image_path']]);v={'Top-1':float(hit[0])}
  for k in (1,5,10):v[f'Recall@{k}']=float(hit[:k].sum()/n)
  for k in (5,10):v[f'NDCG@{k}']=float((hit[:k]/np.log2(np.arange(2,k+2))).sum()/sum(1/math.log2(i+2) for i in range(min(k,n))))
  v['MRR@10']=next((1/(i+1) for i,h in enumerate(hit) if h),0.)
  for k in (5,10):v[f'MAE@{k}']=float(np.mean([abs(pop[item]-t[j]) for item in sorted({gs[i]['item_id'] for i in order[:k]})]))
  per.append({'query_id':q['image_path'],**v})
  for rank,i in enumerate(order,1):rr.append({'query_id':q['image_path'],'rank':rank,'candidate_id':gs[i]['image_path'],'product_id':gs[i]['item_id'],'score':float(score[j,i]),'is_correct':int(hit[rank-1])})
 return rr,per,np.array([[r[m] for m in MET] for r in per]),ranks

def legacy(base,split,rr,a):
 if base=='G2':path=C/f'{split}_G2_top10_5c.csv';metric=C/f'metrics_{split}_5c.csv';label='G2';col='variant'
 else:
  label={'A':'B2-v1','B':'B2-v2'}[base];col='config'
  path=F/f'valid_{label}_top10.csv' if split=='valid' else (W/'artifacts_completed/visual_only_top10.csv' if base=='A' else W/'v2/artifacts_v2/visual_only_v2_top10.csv')
  metric=F/('metrics_valid.csv' if split=='valid' else 'metrics_test_day_du.csv')
 old={(r['query_id'],int(r['rank'])):r for r in read(path)};diff=[]
 for r in rr:
  s=old[(r['query_id'],r['rank'])]
  for c in r:
   same=float(r[c])==float(s[c]) if c in ('rank','score','is_correct') else r[c]==s[c]
   if not same:diff.append((r['query_id'],r['rank'],c,r[c],s[c]))
 assert not diff,('STOP baseline rows',base,split,diff[:3])
 checks=[]
 for r in read(metric):
  if r[col]!=label:continue
  m='MAE@10' if r['metric']=='Popularity MAE' else r['metric']
  if m not in MET:continue
  val,lo,hi=stats(a[:,MET.index(m)]);err=max(abs(val-float(r['value'])),abs(lo-float(r['ci_low'])),abs(hi-float(r['ci_high'])))
  checks.append({'metric':m,'max_error':err});assert err<1e-12,('STOP baseline metric',base,m,err)
 return {'base':base,'split':split,'gallery':1597 if split=='valid' else 1543,'queries':len(a),'rows_matched':len(rr),'all_six_columns_exact':True,'metrics':checks}
def references(qs,gs,pop,t,split):
 prods=sorted({r['item_id'] for r in gs});gp=np.array([pop[p] for p in prods]);imagepop=np.array([pop[r['item_id']] for r in gs]);per=[]
 for j,q in enumerate(qs):
  allerr=np.abs(gp-t[j]);ex=np.array([abs(pop[p]-t[j]) for p in prods if p!=q['item_id']])
  per.append({'query_id':q['image_path'],'gallery_image_mean':float(np.abs(imagepop-t[j]).mean()),'gallery_product_mean':float(allerr.mean()),'gallery_product_excluding_query':float(ex.mean()),'closest10_products_including_query':float(np.sort(allerr)[:10].mean()),'closest10_products_excluding_query':float(np.sort(ex)[:10].mean()),'absolute_lower_bound_one_product':float(allerr.min())})
 write(O/f'reference_{split}_per_query.csv',per)
 return [{'split':split,'gallery':len(gs),'queries':len(qs),'reference':m,'value':v,'ci_low':lo,'ci_high':hi} for m in per[0] if m!='query_id' for v,lo,hi in [stats(np.array([r[m] for r in per]))]]
def run(split,oracle=False):
 check_protocol();qs,gs,gt,pop,t=load(split);summary=[];paired=[];mcn=[];legacychecks=[];scale=[];mono=[];all_results={};refs=[]
 if split=='test':
  assert not oracle;freeze=json.loads((O/'freeze_week6.json').read_text())
  for path,h in freeze['hashes'].items():assert sha(R/path)==h,path
  dump(O/'test_started_week6.json',{'timestamp':now(),'freeze_sha256':sha(O/'freeze_week6.json'),'single_pass':True})
 elif not oracle:dump(O/'validation_started_week6.json',{'timestamp':now(),'protocol_sha256':sha(O/'protocol_week6.md')})
 else:dump(O/'oracle_started_week6.json',{'timestamp':now(),'not_for_selection':True})
 folder=O/('oracle' if oracle else split);folder.mkdir(exist_ok=False)
 for sigma in ([.1,.3] if oracle else [None]):
  tq=t.copy();scenario='official'
  if sigma:
   scenario=f'oracle_sigma{sigma}'
   for j,q in enumerate(qs):
    seed=int.from_bytes(hashlib.sha256(f"42:{q['image_path']}:{sigma:.1f}".encode()).digest()[:8],'big')
    tq[j]=np.clip(pop[q['item_id']]+np.random.default_rng(seed).normal(0,sigma),0,1)
   write(folder/f't_{scenario}.csv',[{'query_id':q['image_path'],'t_oracle':tq[j],'sigma':sigma,'warning':'ORACLE, validation only'} for j,q in enumerate(qs)])
  else:refs=references(qs,gs,pop,tq,split)
  for base in (['A'] if oracle else ['A','B','G2']):
   sim=cosine(base,qs,gs);points=GRID if split=='valid' else [tuple(x) for x in freeze['configs'][base]];scores={}
   for p,g in points:
    name=f'{scenario}_{base}_{key(p,g)}';rr,per,a,ranks=evaluate(sim,qs,gs,gt,pop,tq,p,g);scores[(p,g)]=a
    write(folder/f'{name}_top10.csv',rr);write(folder/f'{name}_per_query.csv',per)
    if p==0 and not oracle:legacychecks.append(legacy(base,split,rr,a))
    if split=='valid' and not oracle and p==0:
     for k in (1,10):
      vals=np.array([sim[j,ids[:k]] for j,ids in enumerate(ranks)]).ravel()
      scale.append({'base':base,'gallery':len(gs),'queries':len(qs),'quantity':f'cos_visual_top{k}','min':float(vals.min()),'median':float(np.median(vals)),'max':float(vals.max()),'std':float(vals.std())})
    for k,m in enumerate(MET):
     v,lo,hi=stats(a[:,k]);summary.append({'split':split,'scenario':scenario,'base':base,'wp':p,'gamma':g,'metric':m,'value':v,'ci_low':lo,'ci_high':hi,'gallery':len(gs),'queries':len(qs)})
    b=scores[(0.,2)]
    for k,m in enumerate(MET):
     v,lo,hi=stats(a[:,k]-b[:,k]);paired.append({'split':split,'scenario':scenario,'base':base,'wp':p,'gamma':g,'metric':m,'difference':v,'ci_low':lo,'ci_high':hi,'gallery':len(gs),'queries':len(qs)})
    n01=int(((b[:,0]==0)&(a[:,0]==1)).sum());n10=int(((b[:,0]==1)&(a[:,0]==0)).sum());n=n01+n10
    pv=min(1.,2*sum(math.comb(n,i) for i in range(min(n01,n10)+1))/2**n) if n else 1.
    mcn.append({'split':split,'scenario':scenario,'base':base,'wp':p,'gamma':g,'n01':n01,'n10':n10,'p_exact':pv,'gallery':len(gs),'queries':len(qs)})
    if not oracle and split=='valid':
     popterm=1-np.abs(np.array([pop[r['item_id']] for r in gs])[None,:]-tq[:,None])**g
     scale.append({'base':base,'gallery':len(gs),'queries':len(qs),'quantity':f'pop_term_{key(p,g)}','min':float(popterm.min()),'median':float(np.median(popterm)),'max':float(popterm.max()),'std':float(popterm.std()),'weighted_std':float(p*popterm.std())})
   if split=='valid' and not oracle:
    for g in (2,4):
     prev=scores[(0.,2)]
     for p in [.1,.2,.3,.4,.5]:
      a=scores[(p,g)];d=a[:,2]-prev[:,2];v,lo,hi=stats(d);bv,bl,bh=stats(a[:,2]-scores[(0.,2)][:,2]);mono.append({'base':base,'wp':p,'gamma':g,'adjacent_difference':v,'ci_low':lo,'ci_high':hi,'vs_zero_difference':bv,'vs_zero_low':bl,'vs_zero_high':bh,'clear_increase':bool(lo>0 or bl>0),'gallery':len(gs),'queries':len(qs)});prev=a
   all_results[(scenario,base)]=scores
   print(split,scenario,base,'complete',flush=True)
 suffix='oracle' if oracle else split
 write(O/f'metrics_{suffix}.csv',summary);write(O/f'paired_{suffix}.csv',paired);write(O/f'mcnemar_{suffix}.csv',mcn)
 if not oracle:
  write(O/f'reference_{split}.csv',refs);dump(O/f'baseline_reproduction_{split}.json',legacychecks)
 if split=='valid' and not oracle:
  # Uniform fieldnames for optional weighted SD.
  for r in scale:r.setdefault('weighted_std','')
  write(O/'scale_valid.csv',scale);write(O/'monotonicity_valid.csv',mono)
  assert not any(r['clear_increase'] for r in mono),'STOP: clear Recall increase'
  selected=[]
  for (_,base),scores in all_results.items():
   baseline=scores[(0.,2)][:,2].mean()
   for threshold in [.01,.02,.03]:
    eligible=[(p,g,a) for (p,g),a in scores.items() if p>0 and a[:,2].mean()>=baseline-threshold]
    eligible.sort(key=lambda x:(-x[0],x[2][:,8].mean(),x[1]))
    if eligible:p,g,a=eligible[0]
    else:p,g,a=0.,2,scores[(0.,2)]
    selected.append({'base':base,'threshold_pp':threshold*100,'wp':p,'gamma':g,'has_operating_point':bool(eligible),'recall5':float(a[:,2].mean()),'mae10':float(a[:,8].mean()),'gallery':len(gs),'queries':len(qs)})
  write(O/'operating_points.csv',selected)
 dump(O/f'completed_{suffix}.json',{'timestamp':now(),'configs_evaluated':len(summary)//len(MET),'gallery':len(gs),'queries':len(qs),'protocol_sha256':sha(O/'protocol_week6.md')})
if __name__=='__main__':run(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='oracle')
