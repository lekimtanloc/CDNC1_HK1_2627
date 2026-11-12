#!/usr/bin/env python3
"""Frozen Week 4 baselines. No training, fusion, or formal evaluation."""
import argparse, csv, hashlib, io, json, platform, random, resource, sqlite3, sys, time, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import PIL
from PIL import Image, ImageOps, ImageDraw
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'FashionRecommender'))
from week3_retrieval_metrics import evaluate


def read(p):
    with open(p, newline='') as f: return list(csv.DictReader(f))


def write(p, rows):
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def crops(z, row):
    im = Image.open(io.BytesIO(z.read(row['image_path']))).convert('RGB')
    mask = np.asarray(Image.open(io.BytesIO(z.read(row['mask_path']))).convert('L'))
    assert im.size == (mask.shape[1], mask.shape[0])
    garment = (mask >= 1) & (mask <= 12)
    y, x = np.where(garment)
    masked = Image.fromarray(np.where(garment[...,None], np.asarray(im), 0).astype(np.uint8))
    old = transforms.CenterCrop(224)(transforms.Resize(256)(masked))
    # Empty masks cannot recover garment content; retain black and audit explicitly.
    if not len(x): return im, old, Image.new('RGB', (224,224))
    bbox = (int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1)
    fitted = ImageOps.contain(masked.crop(bbox), (224,224), Image.Resampling.BILINEAR)
    fixed = Image.new('RGB', (224,224)); fixed.paste(fitted, ((224-fitted.width)//2,(224-fitted.height)//2))
    return im, old, fixed


def duplicate_stats(a, rows):
    groups = defaultdict(list)
    for i,v in enumerate(a): groups[v.tobytes()].append(i)
    groups = [g for g in groups.values() if len(g)>1]
    cross = [g for g in groups if len({rows[i]['item_id'] for i in g})>1]
    return {'groups':len(groups), 'images_in_groups':sum(map(len,groups)), 'excess_vectors':sum(len(g)-1 for g in groups),
            'products_in_cross_product_groups':len({rows[i]['item_id'] for g in cross for i in g}),
            'members':[[rows[i]['image_path'] for i in g] for g in groups]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'artifacts'); ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    protocol=ROOT/'results/week2/FashionRecommender/data_protocol'; archive=ROOT/'archive.zip'; cache=ROOT/'FashionRecommender/repro_outputs_resnet50_full'; prior=ROOT/'results/week3/FashionRecommender'
    inputs=[archive,*sorted(protocol.glob('*.csv')),protocol/'validation_report.json',protocol/'tracking.sqlite3',cache/'metadata.csv',cache/'features_resnet50.pt',cache/'run_manifest.json',prior/'ket_qua_kiem_thu.json',prior/'ket_qua_test_nhieu_goc_nhin.json',prior/'xep_hang_test_top5.csv',ROOT/'FashionRecommender/week3_verify_and_smoke.py',ROOT/'FashionRecommender/week3_retrieval_metrics.py',Path(__file__).resolve()]
    validation=json.loads((protocol/'validation_report.json').read_text()); hashes={str(p):sha(p) for p in inputs}
    assert hashes[str(archive)]==validation['archive_sha256']
    for p in protocol.glob('*.csv'):
        expected=next(r['sha256'] for r in validation['outputs'] if r['path']==p.name); assert hashes[str(p)]==expected
    parts={s:read(protocol/f'{s}.csv') for s in ('train','valid','test')}
    rows=[dict(r,split=s) for s,rr in parts.items() for r in rr]
    assert len({r['image_path'] for r in rows})==len(rows)==13752
    assert [len({r['item_id'] for r in parts[s]}) for s in parts]==[3219,691,690]
    queries=[r for r in parts['test'] if r['role']=='query']; gallery=[r for r in parts['test'] if r['role']=='gallery']; oldrows=queries+gallery
    truth=defaultdict(set)
    for r in read(protocol/'ground_truth_test.csv'): truth[r['query_image_path']].add(r['relevant_gallery_path'])
    assert (len(queries),len(gallery),sum(map(len,truth.values())))==(586,1543,1439)
    assert set(truth)=={r['image_path'] for r in queries}
    assert not set(truth)&{r['image_path'] for r in gallery}
    conn=sqlite3.connect(f'file:{protocol / "tracking.sqlite3"}?mode=ro',uri=True)
    schema={n:[r[1] for r in conn.execute(f'PRAGMA table_info("{n}")')] for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}; conn.close()
    dump(out/'popularity_lineage.json',{'inventory_columns':list(read(protocol/'inventory.csv')[0]),'sqlite_columns':schema,'source':'week3_verify_and_smoke.py smoke(): default_rng(42).integers(0,101,len(metadata)), min-max normalized; metadata order preserved. Not purchases-derived, not trendiness compatibility.','seed':seed})
    meta=read(cache/'metadata.csv'); rng=np.random.default_rng(seed); raw=rng.integers(0,101,len(meta)); pop=(raw-raw.min())/(raw.max()-raw.min()); popularity={r['item_id']:float(p) for r,p in zip(meta,pop)}
    assert set(popularity)=={r['item_id'] for r in rows}
    write(out/'popularity.csv',[{'product_id':r['item_id'],'raw_synthetic':int(v),'popularity':float(p),'metadata_row':i} for i,(r,v,p) in enumerate(zip(meta,raw,pop))])
    model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); model.fc=torch.nn.Identity(); model.eval()
    prep=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    audit=[]; timing={}
    def extract(z, rr, name, fixed):
        t=time.monotonic(); vectors=[]
        for off in range(0,len(rr),args.batch_size):
            tensors=[]
            for r in rr[off:off+args.batch_size]:
                _,before,after=crops(z,r); im=after if fixed else before
                mask_array=np.asarray(Image.open(io.BytesIO(z.read(r['mask_path']))))
                empty_mask=not bool(((mask_array>=1)&(mask_array<=12)).any())
                audit.append({'empty_garment_mask':int(empty_mask),'mode':name,'image_path':r['image_path'],'product_id':r['item_id'],'all_black':int(not np.asarray(im).any())})
                tensors.append(prep(im))
            with torch.inference_mode(): vectors.append(F.normalize(model(torch.stack(tensors)),dim=1).numpy())
            if off%800==0: print(name,off,len(rr),flush=True)
        a=np.concatenate(vectors); assert np.isfinite(a).all(); np.save(out/f'{name}.npy',a); write(out/f'{name}_index.csv',[dict(vector_row=i,**r) for i,r in enumerate(rr)]); timing[name]=time.monotonic()-t; return a
    with zipfile.ZipFile(archive) as z:
        fixed=extract(z,rows,'fixed_all',True)
        old=extract(z,oldrows,'old_test',False)
        write(out/'crop_audit.csv',audit)
        idx={r['image_path']:i for i,r in enumerate(rows)}; testfixed=fixed[[idx[r['image_path']] for r in oldrows]]
        oldcache=torch.load(cache/'features_resnet50.pt',map_location='cpu',weights_only=True).numpy()
        selected=fixed[[idx[r['image_path']] for r in meta]]
        check={'black_fixed_all':sum(r['all_black'] for r in audit if r['mode']=='fixed_all'),'black_fixed_test':sum(not np.asarray(crops(z,r)[2]).any() for r in oldrows),'black_old_test':sum(r['all_black'] for r in audit if r['mode']=='old_test'), 'duplicates_old_product_cache':duplicate_stats(oldcache,meta),'duplicates_fixed_same_representatives':duplicate_stats(selected,meta),'duplicates_old_test':duplicate_stats(old,oldrows),'duplicates_fixed_test':duplicate_stats(testfixed,oldrows),'duplicates_fixed_all':duplicate_stats(fixed,rows)}
        pair=[next(i for i,r in enumerate(meta) if r['item_id']==p) for p in ('1687','3445')]
        check['items_1687_3445']={'paths':[meta[i]['image_path'] for i in pair],'old_exact_equal':bool(np.array_equal(oldcache[pair[0]],oldcache[pair[1]])),'fixed_exact_equal':bool(np.array_equal(selected[pair[0]],selected[pair[1]])),'fixed_cosine':float(selected[pair[0]]@selected[pair[1]]),'all_views':{p:[{'image_path':r['image_path'],'black_old':not bool(np.asarray(crops(z,r)[1]).any()),'black_fixed':not bool(np.asarray(crops(z,r)[2]).any())} for r in rows if r['item_id']==p] for p in ('1687','3445')}}
        dump(out/'crop_checks.json',check)
        sanity={}; rankings_out={}
        def rank(name, matrix):
            result=[]; rankings={}
            for qi,q in enumerate(queries):
                scores=matrix[qi] if matrix.ndim==2 else matrix
                if name=='popularity_only':
                    order=sorted(range(len(gallery)),key=lambda i:(-scores[i],hashlib.sha256(f'{seed}:{gallery[i]["image_path"]}'.encode()).hexdigest(),gallery[i]['image_path']))
                else: order=np.argsort(-scores,kind='stable')
                order=[int(i) for i in order if gallery[int(i)]['image_path']!=q['image_path']][:10]
                rankings[q['image_path']]=[gallery[i]['image_path'] for i in order]
                for k,i in enumerate(order,1): result.append({'query_id':q['image_path'],'rank':k,'candidate_id':gallery[i]['image_path'],'product_id':gallery[i]['item_id'],'score':float(scores[i]),'is_correct':int(gallery[i]['image_path'] in truth[q['image_path']])})
            write(out/f'{name}_top10.csv',result)
            # Existing Week 3 metric implementation only. MRR@1 is exactly Top-1 hit rate.
            sanity[name]={'top1':evaluate(rankings,truth,1)['reciprocal_rank'],'recall_at_5':evaluate(rankings,truth,5)['recall']}; rankings_out[name]=result
        n=len(queries)
        rank('visual_before',old[:n]@old[n:].T); rank('visual_only',testfixed[:n]@testfixed[n:].T); rank('popularity_only',np.array([popularity[r['item_id']] for r in gallery]))
        legacy=read(prior/'xep_hang_test_top5.csv'); legacyrank=defaultdict(list)
        for r in legacy: legacyrank[r['query_image_path']].append(r['gallery_image_path'])
        sanity['week3_saved_top5']={'top1':evaluate(legacyrank,truth,1)['reciprocal_rank'],'recall_at_5':evaluate(legacyrank,truth,5)['recall']}
        dump(out/'sanity_checks.json',{'note':'Chỉ để kiểm tra pipeline; không phải kết quả đánh giá chính thức tuần 5.','results':sanity})
        def panel(images,labels,path):
            canvas=Image.new('RGB',(224*len(images),270),'white'); draw=ImageDraw.Draw(canvas)
            for j,(im,label) in enumerate(zip(images,labels)):
                im=ImageOps.contain(im,(224,224)); canvas.paste(im,(224*j+(224-im.width)//2,0)); draw.text((224*j+5,228),label,fill='black')
            canvas.save(path)
        r=next(r for r in oldrows if not np.asarray(crops(z,r)[1]).any()); ims=crops(z,r)
        panel(ims,[r['image_path'],'OLD: black CenterCrop','FIXED: mask bbox + pad'],out/'crop_before_after.png')
        byq=defaultdict(list)
        for r in rankings_out['visual_only']:
            if r['rank']<=5: byq[r['query_id']].append(r)
        mixed=[q for q,rr in byq.items() if 0<sum(r['is_correct'] for r in rr)<5][:3]
        assert len(mixed)==3
        for j,q in enumerate(mixed,1):
            rr=byq[q]; paths=[q]+[r['candidate_id'] for r in rr]
            ims=[Image.open(io.BytesIO(z.read(p))).convert('RGB') for p in paths]
            panel(ims,['QUERY '+q]+[f"#{r['rank']} {'DUNG' if r['is_correct'] else 'SAI'} item {r['product_id']}\n{r['score']:.4f}" for r in rr],out/f'visual_top5_{j}.png')
    weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    manifest={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'device':'cpu','threads':4,'batch_size':args.batch_size,'deterministic_algorithms':True,'weights':'ResNet50_Weights.IMAGENET1K_V2','weights_sha256':sha(weights),'versions':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL.__version__,'platform':platform.platform()},'elapsed_seconds':time.monotonic()-start,'stage_seconds':timing,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'command':'python3 '+ ' '.join(sys.argv),'rerun':'Use --output-dir with a new directory; existing output is never overwritten.','inputs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashes[str(p)],'rows':len(read(p)) if p.suffix=='.csv' else None} for p in inputs],'outputs':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p),'rows':len(read(p)) if p.suffix=='.csv' else (int(np.load(p,mmap_mode='r').shape[0]) if p.suffix=='.npy' else None)} for p in sorted(out.iterdir())]}
    dump(out/'manifest.json',manifest); print(json.dumps({'sanity':sanity,'seconds':manifest['elapsed_seconds'],'checks':{k:v for k,v in check.items() if k.startswith('black')}},indent=2),flush=True)

if __name__=='__main__': main()
