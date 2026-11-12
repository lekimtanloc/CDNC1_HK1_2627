"""Synthetic spatial checks of actual fixed_frame source, without loading a second model."""
import ast,importlib.util
import numpy as np
from PIL import Image,ImageOps
from common_5c import *
spec=importlib.util.spec_from_file_location('reference_crop_for_tests',W/'v2/common.py');v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
node=next(n for n in ast.parse((O/'extract_5c.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fixed_frame');ns={'np':np,'Image':Image,'ImageOps':ImageOps,'v2':v2,'C':C,'B':B,'FH':FH};exec(compile(ast.Module(body=[node],type_ignores=[]),str(O/'extract_5c.py'),'exec'),ns);crop=ns['fixed_frame'];checks=[]
# C at opposite corners forces224x224 bbox; no resize interpolation. Explicit semantic expectations.
a=np.full((224,224,3),[71,129,203],dtype=np.uint8);mask=np.zeros((224,224),dtype=np.uint8);mask[0,0]=1;mask[-1,-1]=1
for j,label in enumerate([1,4,9,13,14,0]):mask[20:40,20+j*30:40+j*30]=label
im=Image.fromarray(a)
expectations={'G0':lambda lab:lab in (1,4,9,13),'G1':lambda lab:lab in (1,13),'G2':lambda lab:True,'G3':lambda lab:lab in (4,9),'G4':lambda lab:lab not in (4,9)}
for c,rule in expectations.items():
 out,bbox=crop(im,mask,c);assert bbox==(0,0,224,224);got=np.asarray(out)
 for label in (0,1,4,9,13,14):assert np.all(got[mask==label]==(np.array([71,129,203]) if rule(label) else 0)),(c,label)
 checks.append({'check':'exact known pixels '+c,'passed':True,'note':NOTE})
mask=np.full((16,12),14,dtype=np.uint8);im=Image.new('RGB',(12,16),(71,129,203))
for c in VARIANTS:
 out,bbox=crop(im,mask,c);assert bbox is None and not np.asarray(out).any();checks.append({'check':'empty C bbox black fallback '+c,'passed':True,'note':NOTE})
mask[:]=9
for c in ('G1','G4'):
 out,bbox=crop(im,mask,c);assert bbox is not None and not np.asarray(out).any();checks.append({'check':'black retained content with NONempty C bbox '+c,'passed':True,'note':NOTE})
out,bbox=crop(im,mask,'G0');ref=v2.crop(im,mask,C)[1];assert np.asarray(out).tobytes()==np.asarray(ref).tobytes();checks.append({'check':'G0 nonsquare contain/pad equals original C','passed':True,'note':NOTE})
dump(O/'crop_tests_5c.json',{'source_sha256':sha(O/'extract_5c.py'),'checks':checks,'passed':True});print('PASS',len(checks),'synthetic crop checks')
