import random,time,zipfile
from collections import defaultdict,Counter
from common import *
def main():
 out=BASE/'mapping';out.mkdir(exist_ok=False);start=time.monotonic()
 dump(BASE/'hash_cu_truoc.json',{str(p.relative_to(ROOT)): {'sha256':sha(p),'bytes':p.stat().st_size} for p in old_files()})
 rows=read(OLD/'fixed_all_index.csv');stats=[];perlabel=defaultdict(list);co=np.zeros((16,16),dtype=np.int64);coords={}
 with zipfile.ZipFile(ARCHIVE) as z:
  names=z.namelist();dump(out/'archive_inspection.json',{'members':len(names),'non_image_mask_members':[n for n in names if not n.endswith(('.jpg','.png','/'))],'sha256':sha(ARCHIVE)})
  for i,r in enumerate(rows):
   mask=np.asarray(Image.open(io.BytesIO(z.read(r['mask_path']))));assert mask.ndim==2;h,w=mask.shape;flat=mask.ravel();counts=np.bincount(flat,minlength=16);assert len(counts)==16
   if (h,w) not in coords:
    coords[(h,w)]=(np.broadcast_to(np.arange(w,dtype=float)[None,:]/(w-1),(h,w)).ravel().copy(),np.broadcast_to(np.arange(h,dtype=float)[:,None]/(h-1),(h,w)).ravel().copy())
   xx,yy=coords[(h,w)];sx=np.bincount(flat,weights=xx,minlength=16);sy=np.bincount(flat,weights=yy,minlength=16);labs=np.flatnonzero(counts);co[np.ix_(labs,labs)]+=1
   for v in labs:
    v=int(v);d={'image_path':r['image_path'],'split':r['split'],'label':v,'pixels':int(counts[v]),'area_fraction':float(counts[v]/flat.size),'centroid_x':float(sx[v]/counts[v]),'centroid_y':float(sy[v]/counts[v])};stats.append(d);perlabel[v].append((r,d))
   if i%2500==0:print('mapping masks',i,flush=True)
  write(out/'thong_ke_tung_anh.csv',stats)
  summary=[]
  for v in range(16):
   for sp in ['all','train','valid','test']:
    ss=[d for r,d in perlabel[v] if sp=='all' or r['split']==sp];summary.append({'nhan':v,'split':sp,'so_anh':len(ss),'tong_pixel':sum(d['pixels'] for d in ss),'dien_tich_tb_pixel':np.mean([d['pixels'] for d in ss]),'dien_tich_tb_ty_le':np.mean([d['area_fraction'] for d in ss]),'tam_x_tb':np.mean([d['centroid_x'] for d in ss]),'tam_y_tb':np.mean([d['centroid_y'] for d in ss])})
  write(out/'nhan_thong_ke.csv',summary);write(out/'dong_xuat_hien.csv',[{'nhan_a':a,'nhan_b':b,'so_anh_cung_co':int(co[a,b]),'p_b_given_a':float(co[a,b]/co[a,a])} for a in range(16) for b in range(16)])
  selections=[];overview=Image.new('RGB',(1600,1600),'white')
  for v in range(16):
   rng=random.Random(42);chosen=[]
   for sp in ['train','valid','test']:
    pool=[r for r,d in perlabel[v] if r['split']==sp];take=min(8,len(pool));chosen+=rng.sample(pool,take)
   if len(chosen)<24:chosen+=rng.sample([r for r,d in perlabel[v] if r not in chosen],24-len(chosen))
   assert len(chosen)==24
   canvas=Image.new('RGB',(1600,2460),'white');text(canvas,(20,10),f'Nhãn {v:02d} · 24 mẫu ngẫu nhiên · seed 42 · gốc | vùng tô đỏ',27,bold=True)
   for j,r in enumerate(chosen):
    selections.append({'label':v,'kind':'random24','image_path':r['image_path'],'split':r['split'],'cell':j+1});im,mask=load(z,r);arr=np.asarray(im).copy();sel=mask==v;arr[sel]=(0.45*arr[sel]+0.55*np.array([255,20,30])).astype('uint8');overlay=Image.fromarray(arr);x=(j%4)*400;y=60+(j//4)*400
    put(canvas,im,(x+3,y,192,335));put(canvas,overlay,(x+203,y,192,335));text(canvas,(x+8,y+342),f'{j+1:02d} · {Path(r["image_path"]).name}\n{r["split"]}',19)
    if j==0:put(overview,overlay,((v%4)*400,(v//4)*400,390,345));text(overview,((v%4)*400+12,(v//4)*400+350),f'Nhãn {v:02d}',24,bold=True)
   save(canvas,out/f'atlas_nhan_{v:02d}.png')
   # Additional balanced random examples, cropped around label for rare/suspect labels.
   if v in [3,4,6,7,8,9,11,12,13,14,15]:
    extra=[]
    for sp in ['train','valid','test']:
     pool=[r for r,d in perlabel[v] if r['split']==sp and r not in chosen];extra+=rng.sample(pool,min(4,len(pool)))
    panel=Image.new('RGB',(1600,1240),'white');text(panel,(20,10),f'Nhãn {v:02d} · mẫu bổ sung, phóng vùng nhãn · seed 42',28,bold=True)
    for j,r in enumerate(extra):
     selections.append({'label':v,'kind':'extra_roi','image_path':r['image_path'],'split':r['split'],'cell':j+1});im,mask=load(z,r);sel=mask==v;y,x=np.where(sel);bbox=(max(0,int(x.min())-20),max(0,int(y.min())-20),min(im.width,int(x.max())+21),min(im.height,int(y.max())+21));masked=Image.fromarray(np.where(sel[...,None],np.asarray(im),0).astype('uint8'));xx=(j%4)*400;yy=60+(j//4)*390;put(panel,im.crop(bbox),(xx,yy,195,320));put(panel,masked.crop(bbox),(xx+200,yy,195,320));text(panel,(xx+5,yy+330),f'{Path(r["image_path"]).name} · {r["split"]}',18)
    save(panel,out/f'bo_sung_nhan_{v:02d}.png')
   print('atlas',v,flush=True)
  save(overview,out/'atlas_tong_quan.png');write(out/'atlas_samples.csv',selections)
 dump(out/'evidence_manifest.json',{'seed':42,'random_main_per_label':24,'sampling':'8 per split where possible; fill shortages from remaining images; Random(42) per label','centroid':'mean of pixel x/(W-1), y/(H-1) per present image, then equal image average','area':'mean conditioned on label present; no area cutoff in selection','elapsed_seconds':time.monotonic()-start,'source_archive_sha256':sha(ARCHIVE),'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())]})
if __name__=='__main__':main()
