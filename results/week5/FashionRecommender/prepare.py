import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,csv,json,random,io,zipfile,time,platform,hashlib
from collections import Counter,defaultdict
from initialize import O,R,dump,sha
from sanity import read
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
P=R/'results/week2/FashionRecommender/data_protocol'; W=R/'results/week4/FashionRecommender'
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def categories():
 rule={'category_rule_version':'v2','main':{'1':'áo thân trên','2':'chân váy','5':'váy liền','8':'áo khoác','13':'quần'},'accessories':{'6':'túi','7':'trang sức','10':'mũ','11':'kính','12':'thắt lưng','15':'giày','3':'tất'},'image':'main label with largest pixel count; tie smallest numeric label; absent accessory_or_unknown','product':'majority image category; tie smallest numeric label; unknown ordered after numeric labels','review':'100 products seed42, allocate evenly across split/category strata, then distribute remainder to strata with capacity; representative image smallest image_path; compare visible dominant main garment with assigned product category','pass_threshold':90}
 dump(O/'category_rule_v2.json',rule);dump(O/'category_rule_v2_hash.json',{'sha256':sha(O/'category_rule_v2.json')})
 rows=[dict(r,split=s) for s in ('train','valid','test') for r in read(P/f'{s}.csv')]; byim=defaultdict(list)
 for r in read(W/'v2/mapping/thong_ke_tung_anh.csv'):
  if r['label'] in rule['main']:byim[r['image_path']].append((int(r['pixels']),int(r['label'])))
 byprod=defaultdict(list);out=[]
 for r in rows:
  labels=byim[r['image_path']]; cat=str(sorted(labels,key=lambda t:(-t[0],t[1]))[0][1]) if labels else 'accessory_or_unknown'
  byprod[r['item_id']].append(dict(r,category=cat))
 for item,rr in byprod.items():
  counts=Counter(r['category'] for r in rr);cat=min(counts,key=lambda x:(-counts[x],int(x) if x.isdigit() else 999))
  out.append({'item_id':item,'split':rr[0]['split'],'category_v2':cat,'category_old':rr[0]['item_category'],'n_images':len(rr)})
 assert len(out)==4600;write(O/'category_v2.csv',out)
 counts=Counter((r['split'],r['category_v2']) for r in out);write(O/'category_counts.csv',[{'split':s,'category':c,'count':n} for (s,c),n in sorted(counts.items())])
 strata=defaultdict(list)
 for r in sorted(out,key=lambda r:int(r['item_id'])):strata[(r['split'],r['category_v2'])].append(r)
 rng=random.Random(42)
 for ss in strata.values():rng.shuffle(ss)
 sample=[]
 while len(sample)<100:
  for key in sorted(strata):
   if strata[key] and len(sample)<100:sample.append(strata[key].pop())
 check=O/'category_check';check.mkdir(exist_ok=True)
 font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',17)
 with zipfile.ZipFile(R/'archive.zip') as z:
  for page in range(10):
   canvas=Image.new('RGB',(1500,650),'white');d=ImageDraw.Draw(canvas)
   for j,r in enumerate(sample[page*10:page*10+10]):
    rep=min(byprod[r['item_id']],key=lambda x:x['image_path']); im=Image.open(io.BytesIO(z.read(rep['image_path']))).convert('RGB');m=np.array(Image.open(io.BytesIO(z.read(rep['mask_path']))).convert('L'));masked=Image.fromarray(np.where(np.isin(m,[1,2,3,5,6,7,8,10,11,12,13,15])[...,None],np.array(im),0).astype('uint8'))
    x=(j%5)*300;y=(j//5)*325
    for k,img in enumerate((im,masked)):
     img=ImageOps.contain(img,(148,270));canvas.paste(img,(x+k*150,y))
    d.text((x+2,y+273),f"{page*10+j+1}. ID {r['item_id']} {r['split']}",font=font,fill='black');d.text((x+2,y+295),rule['main'].get(r['category_v2'],r['category_v2']),font=font,fill='black')
   canvas.save(check/f'page_{page+1:02d}.png',dpi=(180,180))
 write(check/'sample_100.csv',sample)
 print('category ready',flush=True)
def extract():
 import torch,torchvision,PIL
 import torch.nn.functional as F
 from torchvision import models,transforms
 import importlib.util
 spec=importlib.util.spec_from_file_location('old',W/'run_week4.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 rows=read(P/'valid.csv');rows=[r for r in rows if r['role']=='query']+[r for r in rows if r['role']=='gallery'];assert len(rows)==2165
 model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);model.fc=torch.nn.Identity();model.eval()
 prep=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);a=[];start=time.monotonic()
 with zipfile.ZipFile(R/'archive.zip') as z:
  for off in range(0,len(rows),8):
   batch=torch.stack([prep(old.crops(z,r)[1]) for r in rows[off:off+8]])
   with torch.inference_mode():a.append(F.normalize(model(batch),dim=1).numpy())
   if off%160==0:print('extract',off,'/',len(rows),flush=True)
  a=np.concatenate(a);np.save(O/'old_valid.npy',a);write(O/'old_valid_index.csv',[dict(vector_row=i,**r) for i,r in enumerate(rows)])
  # Reproduce 25 complete original batches to retain identical floating point batch geometry.
  batches=random.Random(42).sample(list(range(len(rows)//8)),25);equal=[]
  for b in batches:
   off=b*8;batch=torch.stack([prep(old.crops(z,r)[1]) for r in rows[off:off+8]])
   with torch.inference_mode():again=F.normalize(model(batch),dim=1).numpy()
   equal.extend({'image_path':rows[off+j]['image_path'],'byte_equal':a[off+j].tobytes()==again[j].tobytes()} for j in range(8))
 write(O/'reproducibility_200.csv',equal);assert all(r['byte_equal'] for r in equal)
 dump(O/'extraction.json',{'seconds':time.monotonic()-start,'versions':{'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'Pillow':PIL.__version__},'reproduced':len(equal),'equal':sum(r['byte_equal'] for r in equal),'sampling':'25 random full batches, seed42; 200 images','source_crop_sha256':sha(W/'run_week4.py'),'weights_sha256':sha(__import__('pathlib').Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth')})
if __name__=='__main__':
 if sys.argv[1]=='categories':categories()
 else:extract()
