"""Independent validation verification; does not import the main metric module."""
import csv,json,math,hashlib
from pathlib import Path
from collections import defaultdict
O=Path(__file__).resolve().parent;R=O.parents[2]
def read(p):return list(csv.DictReader(p.open()))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
gt=defaultdict(set)
for r in read(R/'results/week2/FashionRecommender/data_protocol/ground_truth_valid.csv'):gt[r['query_image_path']].add(r['relevant_gallery_path'])
pop={r['product_id']:float(r['popularity']) for r in read(R/'results/week4/FashionRecommender/artifacts_completed/popularity.csv')}
checks=[];aggregate=[]
for c in ('B0','B2p','B2-v1','B2-v2'):
 rankings=defaultdict(list)
 for r in read(O/f'valid_{c}_top10.csv'):rankings[r['query_id']].append(r)
 main={r['query_id']:r for r in read(O/f'valid_{c}_per_query.csv')};sums=defaultdict(float)
 for q,relevant in gt.items():
  rows=sorted(rankings[q],key=lambda x:int(x['rank']));positions=[int(r['rank']) for r in rows if r['candidate_id'] in relevant]
  metrics={'Top-1':float(1 in positions),'MRR@10':1/min(positions) if positions else 0}
  for k in (1,5,10):metrics[f'Recall@{k}']=len([r for r in positions if r<=k])/len(relevant)
  for k in (5,10):metrics[f'NDCG@{k}']=sum(math.log(2)/math.log(r+1) for r in positions if r<=k)/sum(math.log(2)/math.log(i+1) for i in range(1,min(k,len(relevant))+1))
  t=int(hashlib.sha256(('42:'+q).encode()).hexdigest()[:16],16)/(2**64);products=set(r['product_id'] for r in rows)
  metrics['Popularity MAE']=sum(abs(pop[p]-t) for p in products)/len(products)
  for m,v in metrics.items():
   expected=float(main[q][m]);checks.append({'config':c,'query_id':q,'metric':m,'main':expected,'independent':v,'match':abs(v-expected)<1e-12});sums[m]+=v
 for m,v in sums.items():
  saved=next(float(r['value']) for r in read(O/'metrics_valid.csv') if r['config']==c and r['metric']==m)
  aggregate.append({'source':'metrics_valid.csv','config':c,'metric':m,'reported':saved,'independent':v/len(gt),'match':abs(saved-v/len(gt))<1e-12})
write(O/'independent_validation_checks.csv',checks);write(O/'doi_chieu_so_lieu.csv',aggregate)
assert all(r['match'] for r in checks+aggregate)
print(f'Independent validation: {len(checks)} query/metric comparisons and {len(aggregate)} aggregate comparisons PASS')
