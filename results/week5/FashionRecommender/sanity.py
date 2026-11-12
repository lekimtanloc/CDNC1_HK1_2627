import csv,json,sys
from collections import defaultdict
from initialize import O,R,dump
from metrics import score
sys.path.insert(0,str(R/'FashionRecommender'))
from week3_retrieval_metrics import evaluate
W=R/'results/week4/FashionRecommender'; A=W/'artifacts_completed'; V=W/'v2/artifacts_v2'
def read(p):return list(csv.DictReader(p.open()))
paths={'B0':A/'popularity_only_top10.csv','B2p':A/'visual_before_top10.csv','B2-v1':A/'visual_only_top10.csv','B2-v2':V/'visual_only_v2_top10.csv'}
def truth(split):
 d=defaultdict(set)
 for r in read(R/f'results/week2/FashionRecommender/data_protocol/ground_truth_{split}.csv'):d[r['query_image_path']].add(r['relevant_gallery_path'])
 return d
def ranks(p):
 d=defaultdict(list)
 for r in sorted(read(p),key=lambda x:(x['query_id'],int(x['rank']))):d[r['query_id']].append(r['candidate_id'])
 return d
if __name__=='__main__':
 gt=truth('test')
 saved=json.loads((A/'sanity_checks.json').read_text())['results']; saved_v2=json.loads((V/'sanity_checks_v2.json').read_text())['results']
 expected={c:tuple(d[k] for k in ('top1','recall_at_5')) for c,d in {'B0':saved['popularity_only'],'B2p':saved['visual_before'],'B2-v1':saved['visual_only'],'B2-v2':saved_v2['v2_visual']}.items()}
 results={}
 for c,p in paths.items():
  rr=ranks(p); assert len(read(p))==5860 and set(rr)==set(gt)
  scores=[score(rr[q],gt[q]) for q in gt];a=sum(s['Top-1'] for s in scores)/len(gt);b=sum(s['Recall@5'] for s in scores)/len(gt)
  legacy=(evaluate(rr,gt,1)['reciprocal_rank'],evaluate(rr,gt,5)['recall'])
  results[c]={'top1':a,'recall_at_5':b,'legacy':legacy,'expected':expected[c],'match':all(abs(x-y)<1e-12 for x,y in zip((a,b),expected[c])) and (a,b)==legacy}
 dump(O/'sanity_reproduction.json',results)
 print(json.dumps(results,indent=2));assert all(r['match'] for r in results.values()),'STOP sanity mismatch'
