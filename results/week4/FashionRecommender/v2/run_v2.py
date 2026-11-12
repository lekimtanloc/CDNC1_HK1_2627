import argparse,random,time,platform,resource,shlex,zipfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
import PIL,torch,torchvision
import torch.nn.functional as F
from torchvision import models,transforms
from common import *
sys.path.insert(0,str(ROOT/'FashionRecommender'))
from week3_retrieval_metrics import evaluate

def duplicates(a,rows):
 groups=defaultdict(list)
 for i,v in enumerate(a):groups[v.tobytes()].append(i)
 groups=[g for g in groups.values() if len(g)>1];cross=[g for g in groups if len({rows[i]['item_id'] for i in g})>1]
 return {'groups':len(groups),'images_in_groups':sum(map(len,groups)),'products_in_cross_product_groups':len({rows[i]['item_id'] for g in cross for i in g}),'members':[[rows[i]['image_path'] for i in g] for g in groups]}
def rankings(p):
 result=defaultdict(list)
 for r in read(p):result[r['query_id']].append(r['candidate_id'])
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=BASE/'artifacts_v2');args=ap.parse_args();out=args.output_dir.resolve()
 decision=json.load(open(BASE/'mapping/decision.json'));assert decision['proceed'] and decision['keep_labels'];keep=decision['keep_labels']
 out.mkdir(exist_ok=False);start=time.monotonic();seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 rows=read(OLD/'fixed_all_index.csv');idx={r['image_path']:i for i,r in enumerate(rows)};test=read(PROTOCOL/'test.csv');queries=[r for r in test if r['role']=='query'];gallery=[r for r in test if r['role']=='gallery'];n=len(queries);assert (n,len(gallery),len(rows))==(586,1543,13752)
 validation=json.load(open(PROTOCOL/'validation_report.json'));assert sha(ARCHIVE)==validation['archive_sha256']
 for p in PROTOCOL.glob('*.csv'):assert sha(p)==next(r['sha256'] for r in validation['outputs'] if r['path']==p.name)
 oldman=json.load(open(OLD/'manifest.json'));versions={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL.__version__,'platform':platform.platform()}
 assert versions==oldman['versions'],(versions,oldman['versions'])
 weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth';assert sha(weights)==oldman['weights_sha256']
 model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);model.fc=torch.nn.Identity();model.eval();prep=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 truth=defaultdict(set)
 for r in read(PROTOCOL/'ground_truth_test.csv'):truth[r['query_image_path']].add(r['relevant_gallery_path'])
 assert len(truth)==n and sum(map(len,truth.values()))==1439
 audit=[];timings={}
 def extract(z,rr,name,record):
  t=time.monotonic();vv=[]
  for off in range(0,len(rr),8):
   batch=[]
   for r in rr[off:off+8]:
    im,mask=load(z,r);_,fixed,bbox,count=crop(im,mask,keep);black=int(not np.asarray(fixed).any());labs=np.unique(mask).tolist()
    if record:audit.append({'image_path':r['image_path'],'mask_path':r['mask_path'],'product_id':r['item_id'],'split':r['split'],'role':r['role'],'labels':json.dumps(labs),'kept_pixels':count,'kept_area_fraction':count/mask.size,'bbox':json.dumps(bbox),'bbox_area_fraction':((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(im.width*im.height)) if bbox else 0.,'empty_keep':int(count==0),'empty_type':'all_background' if labs==[0] else ('outside_keep_only' if count==0 else 'not_empty'),'all_black':black})
    batch.append(prep(fixed))
   with torch.inference_mode():vv.append(F.normalize(model(torch.stack(batch)),dim=1).numpy())
   if off%800==0:print(name,off,len(rr),flush=True)
  a=np.concatenate(vv);assert a.shape==(len(rr),2048) and np.isfinite(a).all();assert np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5);np.save(out/f'{name}.npy',a);timings[name]=time.monotonic()-t;return a
 with zipfile.ZipFile(ARCHIVE) as z:
  a=extract(z,rows,'fixed_all_v2',True);write(out/'fixed_all_v2_index.csv',rows);write(out/'crop_audit_v2.csv',audit)
  special=['images/1687_011.jpg','images/3445_011.jpg','images/1000_031.jpg'];rng=random.Random(42);chosen=[idx[p] for p in special];chosen+=rng.sample([i for i in range(len(rows)) if i not in chosen],512-len(chosen));rr=[rows[i] for i in chosen];write(out/'replay_index.csv',[dict(replay_row=j,**r) for j,r in enumerate(rr)])
  b=extract(z,rr,'replay_1',False);c=extract(z,rr,'replay_2',False);dump(out/'reproducibility.json',{'images':len(rr),'seed':42,'special':special,'byte_equal':b.tobytes()==c.tobytes(),'equal_rows':sum(np.array_equal(x,y) for x,y in zip(b,c)),'max_abs_diff':float(np.abs(b-c).max()),'replay_1_sha256':sha(out/'replay_1.npy'),'replay_2_sha256':sha(out/'replay_2.npy'),'vs_full_run_equal_rows':sum(np.array_equal(x,y) for x,y in zip(b,a[chosen])),'vs_full_run_max_abs_diff':float(np.abs(b-a[chosen]).max())})
  # Separate special-image evidence: original | selected pixels | resized/padded.
  panel=Image.new('RGB',(1350,1250),'white');text(panel,(20,10),'Ba ảnh kiểm tra: gốc | vùng được giữ v2 | crop v2',29,bold=True)
  for j,p in enumerate(special):
   r=rows[idx[p]];im,mask=load(z,r);masked,fixed,_,_=crop(im,mask,keep)
   for col,img in enumerate([im,masked,fixed]):put(panel,img,(col*450+20,65+j*385,405,325))
   text(panel,(30,395+j*385),p,22)
  save(panel,out/'special_crop_v2.png')
 olda=np.load(OLD/'fixed_all.npy',mmap_mode='r');meta=read(ROOT/'FashionRecommender/repro_outputs_resnet50_full/metadata.csv');mi=[idx[r['image_path']] for r in meta];testidx=[idx[r['image_path']] for r in queries+gallery]
 black={r['image_path'] for r in audit if r['all_black']};oldblack={r['image_path'] for r in read(OLD/'crop_audit.csv') if r['mode']=='fixed_all' and r['all_black']=='1'}
 checks={'keep_labels':keep,'black_all':len(black),'black_by_split':{s:sum(r['all_black'] for r in audit if r['split']==s) for s in ['train','valid','test']},'empty_count':sum(r['empty_keep'] for r in audit),'empty_classification':dict(Counter(r['empty_type'] for r in audit if r['empty_keep'])),'duplicates_all':duplicates(a,rows),'duplicates_test':duplicates(a[testidx],queries+gallery),'duplicates_representatives':duplicates(a[mi],meta),'v1_duplicates_all':duplicates(olda,rows),'v1_duplicates_test':duplicates(olda[testidx],queries+gallery),'v1_duplicates_representatives':duplicates(olda[mi],meta),'v1_black_all':len(oldblack),'v1_black_test':sum(r['split']=='test' and r['image_path'] in oldblack for r in rows),'special_cosine_v1':float(olda[idx[special[0]]]@olda[idx[special[1]]]),'special_cosine_v2':float(a[idx[special[0]]]@a[idx[special[1]]]),'special_equal_v2':bool(np.array_equal(a[idx[special[0]]],a[idx[special[1]]]))}
 bbox=np.array([r['bbox_area_fraction'] for r in audit]);checks['bbox_quantiles_all']=dict(zip(['min','p01','p05','p50','p95','p99','max'],map(float,np.quantile(bbox,[0,.01,.05,.5,.95,.99,1]))));checks['bbox_positive_below_1pct']=int(((bbox>0)&(bbox<.01)).sum());checks['bbox_above_95pct']=int((bbox>.95).sum());dump(out/'crop_checks_v2.json',checks)
 write(out/'bbox_extremes.csv',sorted(audit,key=lambda r:r['bbox_area_fraction'])[:20]+sorted(audit,key=lambda r:r['bbox_area_fraction'],reverse=True)[:20])
 sim=a[[idx[r['image_path']] for r in queries]]@a[[idx[r['image_path']] for r in gallery]].T;rank={};result=[];blackrank=[]
 for qi,q in enumerate(queries):
  order=np.argsort(-sim[qi],kind='stable');order=[int(i) for i in order if gallery[int(i)]['image_path']!=q['image_path']];rank[q['image_path']]=[gallery[i]['image_path'] for i in order[:10]]
  for k,i in enumerate(order,1):
   r={'query_id':q['image_path'],'rank':k,'candidate_id':gallery[i]['image_path'],'product_id':gallery[i]['item_id'],'score':float(sim[qi,i]),'is_correct':int(gallery[i]['image_path'] in truth[q['image_path']])}
   if k<=10:result.append(r)
   if gallery[i]['image_path'] in black:blackrank.append(dict(query_black=int(q['image_path'] in black),**r))
 write(out/'visual_only_v2_top10.csv',result);write(out/'black_gallery_ranks.csv',blackrank,fields=['query_black','query_id','rank','candidate_id','product_id','score','is_correct'])
 blacktest=[dict(image_path=r['image_path'],product_id=r['item_id'],role=r['role'],gt_pairs=len(truth[r['image_path']]) if r['role']=='query' else sum(r['image_path'] in t for t in truth.values())) for r in queries+gallery if r['image_path'] in black];write(out/'black_test.csv',blacktest,fields=['image_path','product_id','role','gt_pairs'])
 for name in ['popularity_only_top10.csv','popularity.csv','popularity_lineage.json']:
  with open(out/name,'xb') as f:f.write((OLD/name).read_bytes())
 dump(out/'popularity_reference.json',{'note':'Exact byte copies of v1; no dependence on image vectors','files':[{'name':n,'v1_sha256':sha(OLD/n),'v2_sha256':sha(out/n),'equal':sha(OLD/n)==sha(out/n)} for n in ['popularity_only_top10.csv','popularity.csv','popularity_lineage.json']]})
 oldrank=rankings(OLD/'visual_only_top10.csv');poprank=rankings(out/'popularity_only_top10.csv');sanity={}
 for name,rrr in [('v1_visual',oldrank),('v2_visual',rank),('popularity_only',poprank)]:sanity[name]={'top1':evaluate(rrr,truth,1)['reciprocal_rank'],'recall_at_5':evaluate(rrr,truth,5)['recall']}
 dump(out/'sanity_checks_v2.json',{'note':'Chỉ kiểm tra pipeline; cấp ảnh, macro Recall@5 theo số đáp án; không đánh giá chính thức.','results':sanity})
 transitions=[]
 for q in queries:
  p=q['image_path'];x=int(oldrank[p][0] in truth[p]);y=int(rank[p][0] in truth[p]);group='black_both' if p in oldblack and p in black else ('black_v1_only' if p in oldblack else ('black_v2_only' if p in black else 'neither_black'));transitions.append({'query_id':p,'v1_black':int(p in oldblack),'v2_black':int(p in black),'group':group,'v1_top1_correct':x,'v2_top1_correct':y,'transition':f'{x}->{y}'})
 write(BASE/'top1_chuyen_trang_thai_v1_v2.csv',transitions);tr={}
 for name,rr in [('all',transitions),('black_v1',[r for r in transitions if r['v1_black']]),('black_v2',[r for r in transitions if r['v2_black']]),('neither_black',[r for r in transitions if r['group']=='neither_black'])]+[(g,[r for r in transitions if r['group']==g]) for g in ['black_both','black_v1_only','black_v2_only']]:tr[name]={'n':len(rr),'counts':dict(Counter(r['transition'] for r in rr))}
 dump(out/'transition_counts.json',tr)
 inputs=[ARCHIVE,*PROTOCOL.glob('*.csv'),PROTOCOL/'validation_report.json',OLD/'fixed_all.npy',OLD/'fixed_all_index.csv',OLD/'crop_audit.csv',OLD/'visual_only_top10.csv',OLD/'popularity_only_top10.csv',OLD/'popularity.csv',OLD/'popularity_lineage.json',OLD/'manifest.json',BASE/'mapping/decision.json',BASE/'mapping/mapping_de_xuat.csv',ROOT/'FashionRecommender/week3_retrieval_metrics.py',Path(__file__),BASE/'common.py',weights]
 manifest={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':42,'device':'cpu','threads':4,'batch_size':8,'deterministic_algorithms':True,'weights':'ResNet50_Weights.IMAGENET1K_V2','weights_sha256':sha(weights),'versions':versions,'command':'python3 -B '+shlex.join(sys.argv),'elapsed_seconds':time.monotonic()-start,'stage_seconds':timings,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'inputs':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs],'outputs':[{'path':str(p.relative_to(BASE)),'sha256':sha(p),'bytes':p.stat().st_size,'rows':len(read(p)) if p.suffix=='.csv' else (int(np.load(p,mmap_mode='r').shape[0]) if p.suffix=='.npy' else None)} for p in sorted(out.iterdir())]+[{'path':'top1_chuyen_trang_thai_v1_v2.csv','sha256':sha(BASE/'top1_chuyen_trang_thai_v1_v2.csv'),'rows':586}]}
 dump(out/'manifest_v2.json',manifest);print(json.dumps({'black':checks['black_by_split'],'sanity':sanity,'seconds':manifest['elapsed_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
