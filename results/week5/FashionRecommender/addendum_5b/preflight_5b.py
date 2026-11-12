import random,sys,io,csv
from collections import defaultdict,Counter
from common_5b import *

def build_split(records,seed=42):
 """Extracted build_rows logic for ONE fixed split; ignores stored role/query/GT."""
 by=defaultdict(list)
 for r in records:by[r['item_id']].append(r)
 rows=[];gt=[]
 for item in sorted(by,key=lambda v:(len(v),v)):
  rr=sorted(by[item],key=lambda r:r['image_path']);shuffled=list(rr);random.Random(f'{seed}:{item}:views').shuffle(shuffled);qp=shuffled[0]['image_path'] if len(shuffled)>=2 else None
  counts=Counter(r['view_category'] for r in rr if r['view_category']!='Unknown');category=min(counts,key=lambda v:(-counts[v],v)) if counts else 'Unknown'
  for r in rr:
   row={'role':'query' if r['image_path']==qp else 'gallery','item_id':item}
   for k in ('image_path','mask_path','image_bytes','mask_bytes','width','height','mask_label','view_category'):row[k]=r[k]
   row['item_category']=category;rows.append(row)
  if qp:
   for r in shuffled[1:]:gt.append({'query_image_path':qp,'relevant_gallery_path':r['image_path'],'item_id':item,'item_category':category})
 return rows,gt

def csv_bytes(rows):
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return s.getvalue().encode()
if __name__=='__main__':
 files=[p for i in range(1,6) for p in (R/f'results/week{i}').rglob('*') if p.is_file() and O not in p.parents]+list((R/'FashionRecommender').glob('*.py'))+[R/'archive.zip']
 dump(O/'hash_cu_truoc_5b.json',{str(p.relative_to(R)):sha(p) for p in sorted(set(files))})
 dump(O/'design_5b.json',{'timestamp':now(),'note':NOTE,'labels':LABELS,'effect_rule':'effect present only if BOTH paired Recall@5 CIs exclude0 in same direction; else no conclusion','face_hair':['C-B','A-D'],'pants_shoes':['C-A','B-D'],'F_comparison':'F-A plus expected random; no threshold, no human identity labels','metrics':['Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','MRR@10'],'bootstrap':{'replicates':10000,'seed':42,'unit':'query','ci_percentile':[2.5,97.5]},'crop':'week4/v2/common.py crop and load, black224 fallback on empty selected labels','sampling_A':'25 complete original batches chosen by random.Random(42); 200 images; original batch geometry retained','dev':'fixed train item assignments; rebuild roles and GT with original query selection; no model training','test_policy':'embeddings may be extracted beforehand; test ranking and metrics only after freeze_5b.json; one official pass','A_B_reuse':'A fixed_all.npy; B fixed_all_v2.npy','E':'not defined by user; only A,B,C,D,F evaluated','source_hashes':{'split_code':sha(R/'FashionRecommender/prepare_week2_splits.py'),'crop_code':sha(W/'v2/common.py'),'protocol':sha(PRIOR/'protocol_v1.md')}})
 inventory=read(P/'inventory.csv');checks=[]
 for split in ('valid','test','train'):
  # Input omits role and item_category deliberately; infer both independently.
  inp=[{k:v for k,v in r.items() if k not in ('role','item_category')} for r in inventory if r['split']==split];rr,gt=build_split(inp)
  for kind,built,p in [('split',rr,P/f'{split}.csv'),('ground_truth',gt,P/f'ground_truth_{split}.csv')]:
   saved=read(p);eq=built==saved;byteeq=csv_bytes(built)==p.read_bytes();checks.append({'split':split,'kind':kind,'rows':len(built),'row_by_row_equal':eq,'byte_equal':byteeq});assert eq and byteeq,f'STOP split reconstruction mismatch {split} {kind}'
  for role in ('query','gallery'):
   checks.append({'split':split,'kind':role,'rows':sum(r['role']==role for r in rr),'row_by_row_equal':[r for r in rr if r['role']==role]==[r for r in read(P/f'{split}.csv') if r['role']==role],'byte_equal':True})
  if split=='train':
   write(O/'dev_split_5b.csv',rr);write(O/'dev_queries_5b.csv',[r for r in rr if r['role']=='query']);write(O/'dev_gallery_5b.csv',[r for r in rr if r['role']=='gallery']);write(O/'ground_truth_dev_5b.csv',gt)
 dump(O/'split_reconstruction_5b.json',checks)
 print(json.dumps(checks,indent=2))
