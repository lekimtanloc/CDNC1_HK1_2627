import csv,json,hashlib,datetime
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[3];W=R/'results/week4/FashionRecommender';P=R/'results/week2/FashionRecommender/data_protocol';PRIOR=O.parent
NOTE='BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).'
LABELS={'A':list(range(1,13)),'B':[1,2,3,5,6,7,8,10,11,12,13,15],'C':list(range(1,14))+[15],'D':[1,2,3,5,6,7,8,10,11,12],'F':[4,9]}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return list(csv.DictReader(p.open(newline='')))
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def write(p,rs,fields=None):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
