"""Post-freeze descriptive audit; never ranks or selects test configurations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import csv,json,math,hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];D=R/'results/week2/FashionRecommender/data_protocol';F=R/'results/week5/FashionRecommender'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(n,rows):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bs(a):
 rng=np.random.default_rng(42)
 return np.concatenate([a[rng.integers(0,len(a),(100,len(a)))].mean(1) for _ in range(100)])
pop={r['product_id']:float(r['popularity']) for r in read(R/'results/week4/FashionRecommender/artifacts_completed/popularity.csv')}
# Independently verify oracle t, Recall5 and both MAEs for every query/config, including CIs.
rows=read(D/'valid.csv');product={r['image_path']:r['item_id'] for r in rows};gt=defaultdict(set)
for r in read(D/'ground_truth_valid.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
queries=list(gt);sensitivity=[];oraclechecks=[]
for sigma in [.1,.3]:
 ts={r['query_id']:float(r['t_oracle']) for r in read(O/f'oracle/t_oracle_sigma{sigma}.csv')}
 for q in queries:
  seed=int.from_bytes(hashlib.sha256(f'42:{q}:{sigma:.1f}'.encode()).digest()[:8],'big');expected=float(np.clip(pop[product[q]]+np.random.default_rng(seed).normal(0,sigma),0,1));assert ts[q]==expected
 baseline=None
 for path in sorted((O/'oracle').glob(f'oracle_sigma{sigma}_A_*_top10.csv')):
  tokens=path.stem.split('_');p=int(tokens[3][1:])/10;g=int(tokens[4][1:]);ranks=defaultdict(list)
  for r in read(path):ranks[r['query_id']].append(r['candidate_id'])
  vals={m:[] for m in ['Recall@5','MAE@5','MAE@10']}
  for q in queries:
   ids=ranks[q];vals['Recall@5'].append(len(set(ids[:5])&gt[q])/len(gt[q]))
   for k in [5,10]:
    items=set(product[i] for i in ids[:k]);vals[f'MAE@{k}'].append(sum(abs(pop[i]-ts[q]) for i in items)/len(items))
  vals={m:np.array(a) for m,a in vals.items()}
  if p==0:baseline=vals
  saved=read(path.with_name(path.name.replace('_top10','_per_query')));saved={r['query_id']:r for r in saved}
  for m,a in vals.items():
   assert max(abs(a[i]-float(saved[q][m])) for i,q in enumerate(queries))<1e-12
   lo,hi=np.percentile(bs(a),[2.5,97.5]);d=a-baseline[m];dl,dh=np.percentile(bs(d),[2.5,97.5]);r=next(r for r in read(O/'paired_oracle.csv') if r['scenario']==f'oracle_sigma{sigma}' and float(r['wp'])==p and int(r['gamma'])==g and r['metric']==m)
   assert max(abs(d.mean()-float(r['difference'])),abs(dl-float(r['ci_low'])),abs(dh-float(r['ci_high'])))<1e-12
   sensitivity.append({'scenario':f'oracle_sigma{sigma}','sigma':sigma,'base':'A','wp':p,'gamma':g,'metric':m,'value':a.mean(),'ci_low':lo,'ci_high':hi,'difference':d.mean(),'delta_low':dl,'delta_high':dh,'gallery':1597,'queries':568})
  oraclechecks.append({'sigma':sigma,'wp':p,'gamma':g,'queries_verified':len(queries),'gallery':1597,'t_hash_verified':True,'metric_and_paired_ci_verified':True})
for r in read(O/'metrics_valid.csv'):
 if r['base']=='A' and r['metric'] in ['Recall@5','MAE@5','MAE@10']:
  pair=next(s for s in read(O/'paired_valid.csv') if all(s[k]==r[k] for k in ['base','wp','gamma','metric']))
  sensitivity.append({'scenario':'official','sigma':'infinity_label_only','base':'A','wp':r['wp'],'gamma':r['gamma'],'metric':r['metric'],'value':r['value'],'ci_low':r['ci_low'],'ci_high':r['ci_high'],'difference':pair['difference'],'delta_low':pair['ci_low'],'delta_high':pair['ci_high'],'gallery':1597,'queries':568})
write('sensitivity_summary.csv',sensitivity);write('independent_oracle_checks.csv',oraclechecks)
# Reference MAE recalculation without using primary implementation.
refs=[]
for split in ['valid','test']:
 rows=read(D/f'{split}.csv');gs=[r for r in rows if r['role']=='gallery'];qs=[r for r in rows if r['role']=='query'];ts={r['query_id']:float(r['t_q']) for r in read(F/f't_q_{split}.csv')};products=set(r['item_id'] for r in gs);arrays=defaultdict(list)
 for q in qs:
  t=ts[q['image_path']];errors=[abs(pop[p]-t) for p in products];excluded=[abs(pop[p]-t) for p in products if p!=q['item_id']]
  arrays['gallery_image_mean'].append(sum(abs(pop[r['item_id']]-t) for r in gs)/len(gs));arrays['gallery_product_mean'].append(sum(errors)/len(errors));arrays['gallery_product_excluding_query'].append(sum(excluded)/len(excluded));arrays['closest10_products_including_query'].append(sum(sorted(errors)[:10])/10);arrays['closest10_products_excluding_query'].append(sum(sorted(excluded)[:10])/10);arrays['absolute_lower_bound_one_product'].append(min(errors))
 for m,a in arrays.items():
  row=next(r for r in read(O/f'reference_{split}.csv') if r['reference']==m);err=abs(sum(a)/len(a)-float(row['value']));assert err<1e-12;refs.append({'split':split,'reference':m,'independent_value':sum(a)/len(a),'max_error':err,'gallery':len(gs),'queries':len(qs)})
write('independent_references.csv',refs)
# Family-wide Bonferroni. Oracle omitted from main family by protocol.
mcn=[r for split in ['valid','test'] for r in read(O/f'mcnemar_{split}.csv') if float(r['wp'])>0];family=len(mcn)
for r in mcn:r.update(family_comparisons=family,p_bonferroni=min(1.,float(r['p_exact'])*family),significant_bonferroni=float(r['p_exact'])*family<.05)
write('mcnemar_bonferroni.csv',mcn)
metrics=['Recall@5','NDCG@5','MRR@10','MAE@5','MAE@10'];nc=family*len(metrics);corrected=[]
for r in mcn:
 split=r['split'];base=r['base'];p=float(r['wp']);g=int(r['gamma']);name=f'official_{base}_p{round(p*10)}_g{g}';a=read(O/split/f'{name}_per_query.csv');b=read(O/split/f'official_{base}_p0_g2_per_query.csv');bd={r['query_id']:r for r in b}
 for metric in metrics:
  delta=np.array([float(s[metric])-float(bd[s['query_id']][metric]) for s in a]);lo,hi=np.percentile(bs(delta),[100*.025/nc,100*(1-.025/nc)])
  corrected.append({'split':split,'base':base,'wp':p,'gamma':g,'metric':metric,'difference':delta.mean(),'ci_low_bonferroni':lo,'ci_high_bonferroni':hi,'family_metric_comparisons':nc,'bootstrap_draws':10000,'expected_draws_per_tail':10000*.025/nc,'gallery':r['gallery'],'queries':r['queries']})
write('paired_bonferroni.csv',corrected)
print('oracle, references, multiplicity verified',family,nc)
