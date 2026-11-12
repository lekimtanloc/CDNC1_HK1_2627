import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,random,time,zipfile,importlib.util,platform
import numpy as np
import torch,torchvision,PIL
import torch.nn.functional as F
from torchvision import models,transforms
from PIL import Image,ImageOps
from common_5c import *
spec=importlib.util.spec_from_file_location('reference_crop_5c',W/'v2/common.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
versions={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL.__version__};expected=json.loads((BASE/'extraction_complete_5b.json').read_text());assert versions==expected['versions']
weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth';assert sha(weights)==expected['weights_sha256']
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);model.fc=torch.nn.Identity();model.eval();prep=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
rows=read(BASE/'vectors_index_5b.csv');assert len(rows)==4294 and {r['split'] for r in rows}=={'valid','test'}
def fixed_frame(im,mask,variant):
 _,reference,bbox,_=v2.crop(im,mask,C)
 if variant=='G0':return reference,bbox
 if bbox is None:return Image.new('RGB',(224,224)),None
 if variant=='G1':keep=np.isin(mask,B)
 elif variant=='G2':keep=np.ones(mask.shape,dtype=bool)
 elif variant=='G3':keep=np.isin(mask,FH)
 elif variant=='G4':keep=~np.isin(mask,FH)
 else:raise ValueError(variant)
 masked=Image.fromarray(np.where(keep[...,None],np.asarray(im),0).astype(np.uint8));fitted=ImageOps.contain(masked.crop(bbox),(224,224),Image.Resampling.BILINEAR);fixed=Image.new('RGB',(224,224));fixed.paste(fitted,((224-fitted.width)//2,(224-fitted.height)//2));return fixed,bbox

def batch(z,rr,c):
 tensors=[];audit=[]
 for r in rr:
  im,mask=v2.load(z,r);fixed,bbox=fixed_frame(im,mask,c);tensors.append(prep(fixed));audit.append({'image_path':r['image_path'],'split':r['split'],'bbox':json.dumps(bbox),'fallback':int(bbox is None),'all_black':int(not np.asarray(fixed).any()),'note':NOTE})
 with torch.inference_mode():a=F.normalize(model(torch.stack(tensors)),dim=1).numpy()
 return a,audit
if __name__=='__main__':
 start=time.monotonic()
 if sys.argv[1]=='check':
  old=np.load(BASE/'vectors_C_5b.npy',mmap_mode='r');chosen=random.Random(42).sample(range(len(rows)//8),25);checks=[]
  with zipfile.ZipFile(R/'archive.zip') as z:
   for b in chosen:
    off=b*8;a,_=batch(z,rows[off:off+8],'G0')
    for j in range(8):checks.append({'vector_row':off+j,'image_path':rows[off+j]['image_path'],'byte_equal':a[j].tobytes()==old[off+j].tobytes(),'max_abs_diff':float(np.abs(a[j]-old[off+j]).max()),'note':NOTE})
  write(O/'G0_byte_checks_5c.csv',checks);dump(O/'G0_byte_result_5c.json',{'images':len(checks),'equal':sum(r['byte_equal'] for r in checks),'passed':all(r['byte_equal'] for r in checks),'seconds':time.monotonic()-start,'versions':versions,'weights_sha256':sha(weights),'note':NOTE});assert all(r['byte_equal'] for r in checks),'STOP G0 bytes differ from C'
  print('G0 replay PASS 200/200',flush=True)
 elif sys.argv[1]=='extract':
  assert json.loads((O/'G0_byte_result_5c.json').read_text())['passed'];cp=O/'checkpoints_5c';cp.mkdir(exist_ok=True);index=O/'vectors_index_5c.csv'
  if not index.exists():write(index,rows)
  else:assert read(index)==rows
  with zipfile.ZipFile(R/'archive.zip') as z:
   for c in ('G1','G2','G3','G4'):
    completed=O/f'extraction_{c}_5c.json'
    if completed.exists():
     meta=json.loads(completed.read_text());assert sha(O/f'vectors_{c}_5c.npy')==meta['vector_sha256'];print(c,'completed reuse',flush=True);continue
    chunks=[];audit=[];seconds=0
    for off in range(0,len(rows),256):
     p=cp/f'{c}_{off:05d}_5c.npy';ap=cp/f'{c}_{off:05d}_audit_5c.csv';mp=cp/f'{c}_{off:05d}_5c.json'
     if mp.exists():
      meta=json.loads(mp.read_text());assert meta['variant']==c and meta['source_code_sha256']==sha(Path(__file__)) and meta['design_sha256']==sha(O/'design_5c.json') and sha(p)==meta['sha256'] and sha(ap)==meta['audit_sha256'];a=np.load(p);aa=read(ap);print(c,off,'resume checkpoint',flush=True)
     else:
      assert not p.exists() and not ap.exists(),'Incomplete checkpoint: inspect before recovery';tt=time.monotonic();vv=[];aa=[]
      for j in range(off,min(off+256,len(rows)),8):
       a,au=batch(z,rows[j:min(j+8,off+256,len(rows))],c);vv.append(a);aa.extend(au)
      a=np.concatenate(vv);np.save(p,a);write(ap,aa);meta={'variant':c,'offset':off,'rows':len(a),'seconds':time.monotonic()-tt,'sha256':sha(p),'audit_sha256':sha(ap),'source_code_sha256':sha(Path(__file__)),'design_sha256':sha(O/'design_5c.json')};dump(mp,meta);print(c,off+len(a),'/',len(rows),'checkpoint saved',flush=True)
     chunks.append(a);audit.extend(aa);seconds+=meta['seconds']
    dest=O/f'vectors_{c}_5c.npy';assert not dest.exists();a=np.concatenate(chunks);assert a.shape==(4294,2048) and np.isfinite(a).all();np.save(dest,a);write(O/f'crop_audit_{c}_5c.csv',audit)
    dump(completed,{'variant':c,'images':len(rows),'seconds':seconds,'vector_sha256':sha(dest),'counts':{s:{'images':sum(r['split']==s for r in audit),'fallback':sum(int(r['fallback']) for r in audit if r['split']==s),'all_black':sum(int(r['all_black']) for r in audit if r['split']==s)} for s in ('valid','test')},'note':NOTE})
  dump(O/'extraction_complete_5c.json',{'timestamp':now(),'seconds_this_process':time.monotonic()-start,'versions':versions,'weights_sha256':sha(weights),'threads':4,'batch':8,'deterministic':torch.are_deterministic_algorithms_enabled(),'note':NOTE});print('G1–G4 extraction complete; G0 reused C after byte checks',flush=True)
