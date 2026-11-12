import csv,zipfile,io,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
B=Path(__file__).resolve().parent;R=B.parents[3];rows=list(csv.DictReader(open(B.parent/'artifacts_completed/fixed_all_index.csv')));out=[];examples={};union=Counter();font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
with zipfile.ZipFile(R/'archive.zip') as z:
 for i,r in enumerate(rows):
  a=np.asarray(Image.open(io.BytesIO(z.read(r['mask_path']))));counts=np.bincount(a.ravel(),minlength=16);labs=np.flatnonzero(counts).tolist()
  out.append({'image_path':r['image_path'],'split':r['split'],'labels':json.dumps(labs),'has_13_or_15':int(13 in labs or 15 in labs)})
  union['13_or_15']+=int(13 in labs or 15 in labs);union['4_or_7_or_14']+=int(any(v in labs for v in [4,7,14]))
  for v in labs:
   if v not in examples and counts[v]>1000:examples[v]=r['image_path']
  if i%4000==0:print(i,flush=True)
 canvas=Image.new('RGB',(1600,1600),'white');d=ImageDraw.Draw(canvas)
 for v,p in sorted(examples.items()):
  r=next(r for r in rows if r['image_path']==p);a=np.asarray(Image.open(io.BytesIO(z.read(r['mask_path']))));im=Image.open(io.BytesIO(z.read(p))).convert('RGB');masked=Image.fromarray(np.where((a==v)[...,None],np.asarray(im),0).astype('uint8'));x=(v%4)*400;y=(v//4)*400
  for off,img in [(0,im),(200,masked)]:
   thumb=ImageOps.contain(img,(195,345));canvas.paste(thumb,(x+off,y))
  d.text((x+5,y+350),f'Nhãn {v} · {Path(p).name}',font=font,fill='black')
 canvas.save(B/'label_regions_all.png',dpi=(180,180))
with open(B/'mask_image_labels.csv','x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
with open(B/'label_inspection.json','x') as f:json.dump({'union_counts':dict(union),'examples':examples},f,indent=2)
