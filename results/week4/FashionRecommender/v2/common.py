import csv,json,hashlib,io,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent; W4=BASE.parent; ROOT=W4.parents[2]; OLD=W4/'artifacts_completed'; PROTOCOL=ROOT/'results/week2/FashionRecommender/data_protocol'; ARCHIVE=ROOT/'archive.zip'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'; BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rows,fields=None):
 with open(p,'x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,o):
 with open(p,'x') as f:json.dump(o,f,ensure_ascii=False,indent=2)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def load(z,r):
 im=Image.open(io.BytesIO(z.read(r['image_path']))).convert('RGB');mask=np.asarray(Image.open(io.BytesIO(z.read(r['mask_path']))));assert mask.ndim==2 and im.size==(mask.shape[1],mask.shape[0]);return im,mask
def crop(im,mask,keep):
 selected=np.isin(mask,keep);y,x=np.where(selected);masked=Image.fromarray(np.where(selected[...,None],np.asarray(im),0).astype(np.uint8));fixed=Image.new('RGB',(224,224));bbox=None
 if len(x):
  bbox=(int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1);fitted=ImageOps.contain(masked.crop(bbox),(224,224),Image.Resampling.BILINEAR);fixed.paste(fitted,((224-fitted.width)//2,(224-fitted.height)//2))
 return masked,fixed,bbox,int(selected.sum())
def text(im,xy,s,size=22,bold=False,color='#183045'):
 ImageDraw.Draw(im).multiline_text(xy,s,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color,spacing=5)
def put(im,img,box,border=None):
 x,y,w,h=box;t=ImageOps.contain(img,(w-6,h-6));im.paste(t,(x+(w-t.width)//2,y+(h-t.height)//2))
 if border:ImageDraw.Draw(im).rectangle((x,y,x+w,y+h),outline=border,width=4)
def save(im,p):
 if p.exists():raise FileExistsError(p)
 im.save(p,dpi=(180,180))
def old_files():
 roots=[ROOT/'results'/f'week{i}' for i in [1,2,3]]+[ROOT/'FashionRecommender',ROOT/'scripts',W4]
 files={p for root in roots for p in root.rglob('*') if p.is_file() and BASE not in p.parents};files.add(ARCHIVE)
 return sorted(files)
