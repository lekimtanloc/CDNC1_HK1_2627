import os,sys,csv,json,io,zipfile
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageOps,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent.parent; OUT=BASE/'addendum'; FIG=OUT/'figures'; ROOT=BASE.parents[2]; ART=BASE/'artifacts_completed'
sys.dont_write_bytecode=True;sys.path.insert(0,str(BASE));from run_week4 import crops
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'; BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def f(n=22,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def txt(im,xy,s,n=22,color='#172a3a',bold=False):ImageDraw.Draw(im).multiline_text(xy,s,font=f(n,bold),fill=color,spacing=7)
def read(p):
 with open(p,newline='') as h:return list(csv.DictReader(h))
def save(im,n):
 p=FIG/n
 if p.exists():raise FileExistsError(p)
 im.save(p,dpi=(180,180))
def image_at(canvas,im,box,border=None):
 x,y,w,h=box;thumb=ImageOps.contain(im,(w-8,h-8));canvas.paste(thumb,(x+(w-thumb.width)//2,y+(h-thumb.height)//2))
 if border:ImageDraw.Draw(canvas).rectangle((x,y,x+w,y+h),outline=border,width=5)
def main():
 FIG.mkdir(exist_ok=False);idx={r['image_path']:r for r in read(ART/'fixed_all_index.csv')};san=json.load(open(ART/'sanity_checks.json'))['results'];checks=json.load(open(ART/'crop_checks.json'))
 byq=defaultdict(list)
 for r in read(ART/'visual_only_top10.csv'):
  if int(r['rank'])<=5:byq[r['query_id']].append(r)
 qs=[q for q,rs in byq.items() if 0<sum(int(r['is_correct']) for r in rs)<5][:3]
 selection={'visual_queries':qs,'popularity_query':qs[0]}
 with zipfile.ZipFile(ROOT/'archive.zip') as z:
  def load(p):return Image.open(io.BytesIO(z.read(p))).convert('RGB')
  im=Image.new('RGB',(1350,1000),'white');txt(im,(25,15),'Sửa vùng cắt: giữ nội dung có nhãn 1–12',32,bold=True)
  for col,label in enumerate(['Ảnh gốc','CenterCrop cũ: ảnh đen','BBox + giữ tỉ lệ + pad']):txt(im,(col*450+30,70),label,25,bold=True)
  for row,p in enumerate(checks['items_1687_3445']['paths']):
   for col,img in enumerate(crops(z,idx[p])):image_at(im,img,(col*450+25,120+row*380,395,325))
   txt(im,(30,450+row*380),f"Sản phẩm {idx[p]['item_id']} · {p}",22)
  txt(im,(30,905),f"Cosine sau sửa giữa hai ảnh: {checks['items_1687_3445']['fixed_cosine']:.4f}".replace('.',','),27,bold=True)
  txt(im,(30,950),'Đối chiếu hai ảnh cụ thể; không phải mẫu ngẫu nhiên.',21)
  save(im,'hinh_01_crop_truoc_sau.png')
  def gallery_panel(queries,group,title,name):
   im=Image.new('RGB',(1560,110+len(queries)*390),'white');txt(im,(20,15),title,30,bold=True);txt(im,(20,59),'Viền xanh = ĐÚNG · Viền đỏ = SAI · Relevance theo ảnh cùng sản phẩm',22)
   for row,q in enumerate(queries):
    y=108+row*390; entries=[None]+group[q]
    for col,r in enumerate(entries):
     path=q if r is None else r['candidate_id'];x=col*260+8
     image_at(im,load(path),(x,y,242,280),None if r is None else ('#00834b' if int(r['is_correct']) else '#cc3344'))
     if r is None:label=f"Ảnh hỏi · SP {idx[q]['item_id']}\n{Path(q).name}"
     else:label=f"#{r['rank']} · {'ĐÚNG' if int(r['is_correct']) else 'SAI'} · SP {r['product_id']}\n{'Cosine' if 'visual' in name else 'Popularity'} {float(r['score']):.4f}"
     txt(im,(x+3,y+290),label,21)
   save(im,name)
  gallery_panel(qs,byq,'Visual-only: ba ảnh hỏi có cả kết quả đúng và sai','hinh_02_top5_visual_only.png')
  pop=defaultdict(list)
  for r in read(ART/'popularity_only_top10.csv'):
   if int(r['rank'])<=5:pop[r['query_id']].append(r)
  gallery_panel([qs[0]],pop,'Popularity-only: xếp theo độ phổ biến giả lập','hinh_04_vi_du_popularity_only.png')
  empty=read(OUT/'mask_rong_chi_tiet.csv')[:12];selection['empty_examples']=[r['image_path'] for r in empty]
  im=Image.new('RGB',(1440,1170),'white');txt(im,(20,16),'Ảnh gốc có mask không chứa nhãn 1–12',30,bold=True);txt(im,(20,60),'9/2.129 ảnh test; 48/13.752 ảnh · Không đồng nghĩa mask chỉ có nền',24)
  for i,r in enumerate(empty):
   x=(i%4)*360+15;y=115+(i//4)*345;image_at(im,load(r['image_path']),(x,y,325,265));txt(im,(x+5,y+274),f"SP {r['item_id']} · {r['split']}\n{Path(r['image_path']).name}",22)
  save(im,'hinh_05_anh_den_con_lai.png')
 im=Image.new('RGB',(1560,690),'white');txt(im,(30,25),'Hai baseline trên split tuần 2 · 586 ảnh hỏi',33,bold=True)
 d=ImageDraw.Draw(im)
 def box(x,y,w,h,text):
  d.rounded_rectangle((x,y,x+w,y+h),radius=12,fill='#eef4f8',outline='#416680',width=3);txt(im,(x+16,y+24),text,23,bold=True)
 def arrow(x1,y,x2):
  d.line((x1,y,x2,y),fill='#416680',width=4);d.polygon([(x2,y),(x2-12,y-8),(x2-12,y+8)],fill='#416680')
 for x,w,t in [(25,170,'Ảnh + mask'),(240,280,'BBox crop + pad\n224 × 224'),(565,270,'ResNet-50\n2.048 chiều · L2'),(880,365,'Cosine với 1.543\nảnh ứng viên'),(1290,240,'Top-10\nvisual-only')]:box(x,160,w,120,t)
 for x,y in [(195,240),(520,565),(835,880),(1245,1290)]:arrow(x,220,y)
 box(80,390,450,120,'Độ phổ biến giả lập\nSeed 42 · theo sản phẩm');arrow(530,450,600);box(600,390,480,120,'Xếp giảm dần · đồng hạng:\nSHA-256(42:đường dẫn ảnh)');arrow(1080,450,1170);box(1170,390,330,120,'Top-10\npopularity-only')
 txt(im,(35,575),'Cùng gallery · Loại chính ảnh hỏi · Không gộp ảnh thành sản phẩm',25)
 txt(im,(35,625),'Chỉ kiểm tra pipeline tuần 4; không huấn luyện hoặc kết hợp hai nhánh.',23)
 save(im,'hinh_03_so_do_pipeline.png')
 os.environ['MPLCONFIGDIR']='/private/tmp/week4_addendum_mpl'
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 fp=FontProperties(fname=FONT);plt.rcParams['font.family']=fp.get_name()
 keys=['week3_saved_top5','visual_before','visual_only','popularity_only']; labels=['Visual cũ\nCSV tuần 3','Visual cũ\nchạy lại tuần 4','Visual sửa crop','Popularity-only']
 fig,ax=plt.subplots(figsize=(8.8,4.8),dpi=180);import numpy as np
 x=np.arange(4);w=.34
 for delta,key,label,col in [(-w/2,'top1','Top-1','#227c9d'),(w/2,'recall_at_5','Recall@5','#e89b38')]:
  bars=ax.bar(x+delta,[100*san[k][key] for k in keys],w,label=label,color=col)
  ax.bar_label(bars,labels=[f'{100*san[k][key]:.4f}%' for k in keys],padding=4,fontsize=8)
 ax.set_xticks(x,labels);ax.set_ylim(0,40);ax.set_ylabel('Tỉ lệ (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right',fontsize=9)
 ax.set_title('Kiểm tra nhanh pipeline, chưa phải bảng đánh giá chính thức',fontsize=12,pad=20);fig.tight_layout();fig.savefig(FIG/'hinh_06_so_sanh_sanity.png',dpi=180,facecolor='white');plt.close(fig)
 (OUT/'figure_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
 captions=[('hinh_01_crop_truoc_sau.png','archive.zip: hai ảnh 1687_011, 3445_011 + mask; run_week4.crops; crop_checks.json','Đối chiếu ảnh gốc, CenterCrop và bbox crop của hai sản phẩm từng trùng vector; cosine sau sửa 0,4951. Hai ảnh chỉ định, không ngẫu nhiên.'),('hinh_02_top5_visual_only.png','visual_only_top10.csv và ảnh gốc archive.zip','Ba ảnh hỏi đầu theo thứ tự CSV có cả đúng/sai trong Top-5 visual-only. Mẫu được chọn có điều kiện, không ngẫu nhiên; ảnh hiển thị gốc, điểm từ vector masked đã sửa.'),('hinh_03_so_do_pipeline.png','run_week4.py; manifest.json; test.csv','Hai nhánh baseline độc lập sử dụng cùng protocol tuần 2. Sơ đồ mô tả mã đã chạy, không biểu diễn phép thử mới.'),('hinh_04_vi_du_popularity_only.png','popularity_only_top10.csv; archive.zip','Ảnh hỏi đầu của Hình 2 và Top-5 popularity-only. Viền theo ground truth; popularity là giả lập, không phải cosine.'),('hinh_05_anh_den_con_lai.png','mask_rong_chi_tiet.csv; archive.zip','Mười hai ảnh đầu theo thứ tự index trong nhóm không có nhãn 1–12. Không phải mask toàn nền và không phải mẫu ngẫu nhiên; ảnh hiển thị là ảnh gốc.'),('hinh_06_so_sanh_sanity.png','sanity_checks.json, bốn cấu hình','Top-1 và Recall@5 phục vụ kiểm tra pipeline, chưa phải đánh giá chính thức. Hai cột visual cũ là hai nguồn của cùng đối chứng, không phải hai baseline độc lập.')]
 text='# Mô tả hình\n\nPNG nền trắng, 180 dpi, rộng tối đa 1.600 px. Chữ tiếng Việt được raster hóa bằng Arial hệ thống vào PNG (không phụ thuộc font máy người xem).\n\n'
 for i,(n,src,cap) in enumerate(captions,1):text+=f'## {n}\n\nNguồn: {src}.\n\nHình {i}. {cap}\n\n'
 (FIG/'mo_ta_hinh.md').write_text(text)
 print('Created six figures')
if __name__=='__main__':main()
