import csv,json,hashlib,datetime
from pathlib import Path
O=Path(__file__).resolve().parent;BASE=O.parent;PRIOR=BASE.parent;R=O.parents[4];W=R/'results/week4/FashionRecommender';P=R/'results/week2/FashionRecommender/data_protocol'
NOTE='BỔ SUNG HẬU NGHIỆM 5c; không đổi baseline B2-v1 (bộ lọc sai ngữ nghĩa); không kết luận nguyên nhân/nhận diện người.'
C=[1,2,3,4,5,6,7,8,9,10,11,12,13,15];B=[1,2,3,5,6,7,8,10,11,12,13,15];FH=[4,9]
VARIANTS=['G0','G1','G2','G3','G4'];PAIRS=[('G1','B'),('G0','G1'),('G2','G4'),('G2','G0'),('G4','G1'),('G3','F'),('G3','G0')]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return list(csv.DictReader(p.open(newline='')))
def dump(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def write(p,rs):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
