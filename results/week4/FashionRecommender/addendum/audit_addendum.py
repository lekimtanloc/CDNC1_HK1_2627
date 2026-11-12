import csv,io,json,hashlib,sys,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from PIL import Image
BASE=Path(__file__).resolve().parent.parent; OUT=BASE/'addendum'; ROOT=BASE.parents[2]; ART=BASE/'artifacts_completed'
sys.path.insert(0,str(ROOT/'FashionRecommender'))
from week3_retrieval_metrics import evaluate

def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with open(p,'x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,obj):
 with open(p,'x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 frozen=[p for p in BASE.rglob('*') if p.is_file() and OUT not in p.parents];dump(OUT/'original_files_sha256.json',{str(p.relative_to(BASE)):sha(p) for p in frozen})
 rows=read(ART/'fixed_all_index.csv'); audit=read(ART/'crop_audit.csv');pix=Counter();freq=Counter();details=[];modes=Counter(); affected=[]
 empty={r['image_path'] for r in audit if r['mode']=='fixed_all' and r['empty_garment_mask']=='1'}
 with zipfile.ZipFile(ROOT/'archive.zip') as z:
  assert len([n for n in z.namelist() if n.startswith('masks/') and n.endswith('.png')])==len(rows)==13752
  for i,r in enumerate(rows):
   im=Image.open(io.BytesIO(z.read(r['mask_path'])));modes[im.mode]+=1;a=np.asarray(im); assert a.ndim==2
   labs,counts=np.unique(a,return_counts=True); vals=[int(v) for v in labs]; pix.update(dict(zip(vals,map(int,counts))));freq.update(vals)
   if any(v>=16 for v in vals):affected.append({'image_path':r['image_path'],'split':r['split'],'labels_outside_1_12':json.dumps([v for v in vals if v>=16])})
   if r['image_path'] in empty:
    kind='i_toan_nen' if vals==[0] else ('ii_chi_nhan_ngoai_1_12' if not any(1<=v<=12 for v in vals) else 'iii_khac')
    details.append({'item_id':r['item_id'],'image_path':r['image_path'],'mask_path':r['mask_path'],'split':r['split'],'role':r['role'],'labels':json.dumps(vals),'background_ratio':float(np.mean(a==0)),'classification':kind})
   if i%2500==0:print('mask',i,flush=True)
  (OUT/'images').mkdir(exist_ok=True)
  for n in ['1687_011.jpg','3445_011.jpg']:(OUT/'images'/n).write_bytes(z.read('images/'+n))
 write(OUT/'mask_label_frequency.csv',[{'label':v,'pixel_count':pix[v],'image_count':freq[v]} for v in sorted(pix)])
 write(OUT/'mask_rong_chi_tiet.csv',details)
 if affected:write(OUT/'nhan_ngoai_1_12.csv',affected)
 oldidx=read(ART/'old_test_index.csv');old=np.load(ART/'old_test.npy');new=np.load(ART/'fixed_all.npy');index={r['image_path']:i for i,r in enumerate(rows)};test=[r for r in oldidx];q=[r for r in test if r['role']=='query'];g=[r for r in test if r['role']=='gallery'];gi={r['image_path']:i for i,r in enumerate(g)}
 protocol=ROOT/'results/week2/FashionRecommender/data_protocol';truth=defaultdict(set)
 for r in read(protocol/'ground_truth_test.csv'):truth[r['query_image_path']].add(r['relevant_gallery_path'])
 newtest=new[[index[r['image_path']] for r in test]]; sims=newtest[:len(q)]@newtest[len(q):].T
 blacktest=[r for r in test if r['image_path'] in empty]; bg=[r for r in blacktest if r['role']=='gallery'];bq=[r for r in blacktest if r['role']=='query'];bt=[];ties=[]
 for r in blacktest:
  bt.append({'image_path':r['image_path'],'product_id':r['item_id'],'role':r['role'],'has_gt_pair':int(bool(truth[r['image_path']]) if r['role']=='query' else any(r['image_path'] in t for t in truth.values())),'gt_pair_count':len(truth[r['image_path']]) if r['role']=='query' else sum(r['image_path'] in t for t in truth.values())})
 for j,r in enumerate(q):
  pos=np.argsort(-sims[j],kind='stable'); ranks={g[int(i)]['image_path']:rank for rank,i in enumerate(pos,1)}
  for b in bg:ties.append({'query_id':r['image_path'],'query_black':int(r['image_path'] in empty),'candidate_id':b['image_path'],'score':float(sims[j,gi[b['image_path']]]),'full_rank':ranks[b['image_path']],'is_correct':int(b['image_path'] in truth[r['image_path']])})
 write(OUT/'anh_den_test.csv',bt);write(OUT/'thu_hang_ung_vien_den.csv',ties)
 grouped={};recalc={}
 for name in ['visual_before','visual_only','popularity_only']:
  rank=defaultdict(list)
  for r in read(ART/f'{name}_top10.csv'):rank[r['query_id']].append(r['candidate_id'])
  grouped[name]=rank;recalc[name]={'top1':evaluate(rank,truth,1)['reciprocal_rank'],'recall_at_5':evaluate(rank,truth,5)['recall']}
 legacy=defaultdict(list)
 for r in read(ROOT/'results/week3/FashionRecommender/xep_hang_test_top5.csv'):legacy[r['query_image_path']].append(r['gallery_image_path'])
 recalc['week3_saved_top5']={'top1':evaluate(legacy,truth,1)['reciprocal_rank'],'recall_at_5':evaluate(legacy,truth,5)['recall']}
 oldblack={r['image_path'] for r in audit if r['mode']=='old_test' and r['all_black']=='1'};trans=[]
 for r in q:
  p=r['image_path'];a=grouped['visual_before'][p][0] in truth[p];b=grouped['visual_only'][p][0] in truth[p]
  trans.append({'query_id':p,'old_black':int(p in oldblack),'empty_1_12':int(p in empty),'old_hit':int(a),'fixed_hit':int(b),'transition':f'{int(a)}->{int(b)}'})
 write(OUT/'top1_query_transitions.csv',trans)
 splits={'all':trans,'old_black':[r for r in trans if r['old_black']],'old_nonblack':[r for r in trans if not r['old_black']],'empty_1_12':[r for r in trans if r['empty_1_12']]}
 transitions={k:{'n':len(v),'counts':dict(Counter(r['transition'] for r in v))} for k,v in splits.items()}
 cache=ROOT/'FashionRecommender/repro_outputs_resnet50_full';meta=read(cache/'metadata.csv');cached=torch.load(cache/'features_resnet50.pt',weights_only=True).numpy();ci={r['image_path']:i for i,r in enumerate(meta)};inter=[(i,ci[r['image_path']]) for i,r in enumerate(test) if r['image_path'] in ci];diff=np.abs(np.stack([old[i]-cached[j] for i,j in inter]));exact=sum(np.array_equal(old[i],cached[j]) for i,j in inter)
 def dupe(a,rr):
  d=defaultdict(list)
  for i,v in enumerate(a):d[v.tobytes()].append(i)
  dd=[v for v in d.values() if len(v)>1];return {'vectors':sum(map(len,dd)),'products':len({rr[i]['item_id'] for v in dd for i in v})}
 selected=new[[index[r['image_path']] for r in meta]];pi=[ci[p] for p in ['images/1687_011.jpg','images/3445_011.jpg']]
 manifest=json.load(open(ART/'manifest.json'));saved=json.load(open(ART/'sanity_checks.json'))['results']
 checks=[('Ảnh đen test cũ',100,len(oldblack)),('Ảnh đen test sửa',9,len(blacktest)),('Vector đại diện trùng cũ',162,dupe(cached,meta)['vectors']),('Vector đại diện trùng sửa',11,dupe(selected,meta)['vectors']),('Ảnh đen toàn split',48,sum(int(r['all_black']) for r in audit if r['mode']=='fixed_all')),('Rỗng nhãn 1–12',48,len(details)),('Sản phẩm vector trùng toàn split',39,dupe(new,rows)['products']),('Cosine 1687–3445',0.49510920,float(selected[pi[0]]@selected[pi[1]])),('Thời gian (s)',915.47,manifest['elapsed_seconds']),('RSS (MiB)',815.61,manifest['peak_rss_bytes']/1024**2)]
 for k in recalc:
  for metric in ['top1','recall_at_5']:checks.append((k+'/'+metric,saved[k][metric],recalc[k][metric]))
 write(OUT/'doi_chieu_so_lieu.csv',[{'quantity':k,'reported':a,'recalculated':b,'match':abs(a-b)<(0.0051 if k in ['Thời gian (s)','RSS (MiB)'] else 1e-8),'source':'manifest.json (không đo lại)' if k in ['Thời gian (s)','RSS (MiB)'] else 'CSV/vector nguồn'} for k,a,b in checks])
 bvec=new[[index[r['image_path']] for r in blacktest]]
 result={'mask_modes':dict(modes),'mask_classification':dict(Counter(r['classification'] for r in details)),'outside_16_plus_images':len(affected),'black_queries':len(bq),'black_gallery':len(bg),'black_cosine_min':float((bvec@bvec.T).min()),'black_cosine_max':float((bvec@bvec.T).max()),'black_vectors_equal':all(np.array_equal(bvec[0],v) for v in bvec),'black_gallery_scores_equal_each_query':bool(np.all(sims[:,[gi[r['image_path']] for r in bg]]==sims[:,gi[bg[0]['image_path']]][:,None])),'chance':{'mean_relevant':sum(map(len,truth.values()))/len(q),'top1':sum(map(len,truth.values()))/len(q)/len(g),'recall5':5/len(g)},'transitions':transitions,'cache_comparison':{'old_cache_rows':len(meta),'old_test_rows':len(test),'intersection_images':len(inter),'bitwise_equal_rows':exact,'max_abs_diff':float(diff.max()),'mean_abs_diff':float(diff.mean()),'week3_top5_rankings_equal_queries':sum(legacy[p]==grouped['visual_before'][p][:5] for p in truth),'week3_top1_hit_flags_equal_queries':sum((legacy[p][0] in truth[p])==(grouped['visual_before'][p][0] in truth[p]) for p in truth)},'sanity_recomputed':recalc,'single_view_gallery_products':sum(v==1 for v in Counter(r['item_id'] for r in test).values())}
 dump(OUT/'audit_results.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
