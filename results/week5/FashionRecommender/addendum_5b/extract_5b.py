import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,random,time,zipfile,importlib.util,platform
import numpy as np
import torch,torchvision,PIL
import torch.nn.functional as F
from torchvision import models,transforms
from common_5b import *
spec=importlib.util.spec_from_file_location('v2_crop_5b',W/'v2/common.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
versions={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL.__version__}
expected=json.loads((W/'artifacts_completed/manifest.json').read_text());assert all(versions[k]==expected['versions'][k] for k in versions),(versions,expected['versions'])
weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth';assert sha(weights)==expected['weights_sha256']
model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2);model.fc=torch.nn.Identity();model.eval();prep=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
allrows=read(W/'artifacts_completed/fixed_all_index.csv')
def batch(z,rr,labels):
 tensors=[];audit=[]
 for r in rr:
  im,mask=v2.load(z,r);_,fixed,bbox,count=v2.crop(im,mask,labels);tensors.append(prep(fixed));audit.append({'image_path':r['image_path'],'split':r['split'],'fallback':int(count==0),'all_black':int(not np.asarray(fixed).any()),'bbox':json.dumps(bbox),'kept_pixels':count})
 with torch.inference_mode():a=F.normalize(model(torch.stack(tensors)),dim=1).numpy()
 return a,audit
if __name__=='__main__':
 start=time.monotonic()
 if sys.argv[1]=='check':
  assert all(r['row_by_row_equal'] for r in json.loads((O/'split_reconstruction_5b.json').read_text()))
  old=np.load(W/'artifacts_completed/fixed_all.npy',mmap_mode='r');chosen=random.Random(42).sample(list(range(len(allrows)//8)),25);checks=[]
  with zipfile.ZipFile(R/'archive.zip') as z:
   for b in chosen:
    off=b*8;a,_=batch(z,allrows[off:off+8],LABELS['A'])
    for j in range(8):checks.append({'vector_row':off+j,'image_path':allrows[off+j]['image_path'],'byte_equal':a[j].tobytes()==old[off+j].tobytes(),'max_abs_diff':float(np.abs(a[j]-old[off+j]).max())})
  write(O/'A_byte_check_5b.csv',checks);dump(O/'A_byte_result_5b.json',{'images':len(checks),'equal':sum(r['byte_equal'] for r in checks),'passed':all(r['byte_equal'] for r in checks),'seconds':time.monotonic()-start,'versions':versions,'weights_sha256':sha(weights),'note':NOTE})
  assert all(r['byte_equal'] for r in checks),'STOP: vector A not byte equal'
  print('PASS A byte comparison 200/200',flush=True)
 elif sys.argv[1]=='extract':
  assert json.loads((O/'A_byte_result_5b.json').read_text())['passed']
  rows=[r for r in allrows if r['split'] in ('valid','test')];assert len(rows)==4294
  cp=O/'checkpoints_5b';cp.mkdir(exist_ok=True)
  index=O/'vectors_index_5b.csv'
  if not index.exists():write(index,[dict(local_vector_row=i,**r) for i,r in enumerate(rows)])
  else:assert [r['image_path'] for r in read(index)]==[r['image_path'] for r in rows]
  with zipfile.ZipFile(R/'archive.zip') as z:
   for c in ('C','D','F'):
    completed=O/f'extraction_{c}_5b.json'
    if completed.exists():
     prior=json.loads(completed.read_text());assert prior['labels']==LABELS[c] and prior['vector_sha256']==sha(O/f'vectors_{c}_5b.npy');assert len(read(O/f'crop_audit_{c}_5b.csv'))==len(rows);print(c,'completed variant reused',flush=True);continue
    chunks=[];audit=[];seconds=0
    for off in range(0,len(rows),256):
     path=cp/f'{c}_{off:05d}_5b.npy';meta=cp/f'{c}_{off:05d}_5b.json';ap=cp/f'{c}_{off:05d}_audit_5b.csv'
     if meta.exists():
      m=json.loads(meta.read_text());assert m['labels']==LABELS[c] and m['crop_code_sha256']==sha(W/'v2/common.py') and m['weights_sha256']==sha(weights) and sha(path)==m['sha256'] and sha(ap)==m['audit_sha256'];a=np.load(path);aa=read(ap);print(c,off,'checkpoint reused',flush=True)
     else:
      assert not path.exists() and not ap.exists(),'Incomplete checkpoint: inspect before resuming'
      tt=time.monotonic();vs=[];aa=[]
      for j in range(off,min(off+256,len(rows)),8):
       a,au=batch(z,rows[j:min(j+8,off+256,len(rows))],LABELS[c]);vs.append(a);aa.extend(au)
      a=np.concatenate(vs);np.save(path,a);write(ap,aa);m={'labels':LABELS[c],'offset':off,'rows':len(a),'seconds':time.monotonic()-tt,'sha256':sha(path),'audit_sha256':sha(ap),'crop_code_sha256':sha(W/'v2/common.py'),'weights_sha256':sha(weights)};dump(meta,m);print(c,off+len(a),'/',len(rows),'checkpoint saved',flush=True)
     chunks.append(a);audit.extend(aa);seconds+=m['seconds']
    path=O/f'vectors_{c}_5b.npy';assert not path.exists();a=np.concatenate(chunks);assert a.shape==(len(rows),2048) and np.isfinite(a).all();np.save(path,a);write(O/f'crop_audit_{c}_5b.csv',audit)
    dump(O/f'extraction_{c}_5b.json',{'note':NOTE,'labels':LABELS[c],'images':len(rows),'seconds':seconds,'versions':versions,'counts':{s:{'images':sum(r['split']==s for r in audit),'fallback':sum(int(r['fallback']) for r in audit if r['split']==s),'black':sum(int(r['all_black']) for r in audit if r['split']==s)} for s in ('valid','test')},'checkpoint_chunks':len(chunks),'vector_sha256':sha(path)})
  dump(O/'extraction_complete_5b.json',{'timestamp':now(),'elapsed_seconds_this_process':time.monotonic()-start,'versions':versions,'weights_sha256':sha(weights),'cpu_threads':torch.get_num_threads(),'batch':8,'deterministic':torch.are_deterministic_algorithms_enabled(),'note':NOTE});print('Extraction C,D,F complete',flush=True)
