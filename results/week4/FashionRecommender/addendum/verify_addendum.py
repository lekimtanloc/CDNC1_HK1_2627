import csv,json,hashlib,subprocess,sys
from pathlib import Path
from PIL import Image
B=Path(__file__).resolve().parent;BASE=B.parent;A=BASE/'artifacts_completed'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def read(p):return list(csv.DictReader(open(p)))
original=json.load(open(B/'original_files_sha256.json'))
for p,h in original.items():assert sha(BASE/p)==h,p
freq=read(B/'mask_label_frequency.csv');labels=read(B/'mask_image_labels.csv');empty=read(B/'mask_rong_chi_tiet.csv')
assert len(labels)==13752 and len(empty)==48
for r in freq:assert int(r['image_count'])==sum(int(r['label']) in json.loads(x['labels']) for x in labels)
assert sum(r['classification']=='i_toan_nen' for r in empty)==13
assert sum(r['classification']=='ii_chi_nhan_ngoai_1_12' for r in empty)==35
assert all(r['match']=='True' for r in read(B/'doi_chieu_so_lieu.csv'))
assert len(read(B/'anh_den_test.csv'))==9
assert len(read(B/'top1_query_transitions.csv'))==586
for p in (B/'figures').glob('hinh_*.png'):
 im=Image.open(p);assert im.width<=1600;assert 150<=im.info['dpi'][0]<=200
assert len(list((B/'figures').glob('hinh_*.png')))==6
assert sum(1 for l in (B/'ket_qua_tuan_04_ban_sua.md').read_text().splitlines() if l.startswith('## '))==8
# Cross-check direct duplicate recomputation against saved checks.
c=json.load(open(A/'crop_checks.json'));d={r['quantity']:float(r['recalculated']) for r in read(B/'doi_chieu_so_lieu.csv')}
assert d['Vector đại diện trùng cũ']==c['duplicates_old_product_cache']['images_in_groups']
assert d['Vector đại diện trùng sửa']==c['duplicates_fixed_same_representatives']['images_in_groups']
assert d['Sản phẩm vector trùng toàn split']==c['duplicates_fixed_all']['products_in_cross_product_groups']
res=subprocess.run([sys.executable,'-B',str(BASE/'verify_week4.py'),str(A)],check=True,capture_output=True,text=True)
print(res.stdout.strip());print(f'PASS addendum: {len(original)} original files unchanged; masks/frequencies, A6 comparisons, transitions, six PNGs and eight report sections validated.')
