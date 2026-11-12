import os,random,zipfile
from collections import defaultdict
from common import *
def main():
 out=BASE/'figures';out.mkdir(exist_ok=False);art=BASE/'artifacts_v2';keep=json.load(open(BASE/'mapping/decision.json'))['keep_labels'];rows=read(art/'fixed_all_v2_index.csv');idx={r['image_path']:r for r in rows};mapping=read(BASE/'mapping/mapping_de_xuat.csv');samples=read(BASE/'mapping/atlas_samples.csv');chosen={int(r['label']):r['image_path'] for r in reversed(samples) if r['kind']=='random24'}
 with zipfile.ZipFile(ARCHIVE) as z:
  canvas=Image.new('RGB',(1600,1680),'white');text(canvas,(20,10),'Mapping v2: đề xuất từ atlas thực nghiệm',30,bold=True)
  for r in mapping:
   v=int(r['nhan']);im,mask=load(z,idx[chosen[v]]);sel=mask==v;a=np.asarray(im).copy();a[sel]=(.45*a[sel]+.55*np.array([255,20,30])).astype('uint8');x=(v%4)*400;y=60+(v//4)*400;put(canvas,Image.fromarray(a),(x+10,y,380,300));text(canvas,(x+12,y+308),f"{v}: {r['ten_vung_de_xuat']}\nTin cậy {r['do_tin_cay']} · {'GIỮ' if v in keep else 'BỎ'}",20)
  save(canvas,out/'hinh_01_atlas_nhan.png')
  rng=random.Random(42);special=['images/1687_011.jpg','images/1000_031.jpg'];ps=special+rng.sample([r['image_path'] for r in rows if r['image_path'] not in special],2)
  panel=Image.new('RGB',(1380,1590),'white');text(panel,(25,10),'Ảnh gốc | crop v1 (1–12) | crop v2',30,bold=True)
  for j,p in enumerate(ps):
   im,mask=load(z,idx[p]);_,v1,_,_=crop(im,mask,list(range(1,13)));_,v2,_,_=crop(im,mask,keep)
   for col,img in enumerate([im,v1,v2]):put(panel,img,(col*460+20,60+j*375,420,315))
   text(panel,(25,380+j*375),f'{p} · {idx[p]["split"]}',22)
  save(panel,out/'hinh_02_crop_v1_v2.png')
  rr=[]
  for p in [OLD/'visual_only_top10.csv',art/'visual_only_v2_top10.csv']:
   byq=defaultdict(list)
   for r in read(p):
    if int(r['rank'])<=5:byq[r['query_id']].append(r)
   rr.append(byq)
  q=next(q for q in rr[0] if [r['candidate_id'] for r in rr[0][q]]!=[r['candidate_id'] for r in rr[1][q]])
  im=Image.new('RGB',(1560,920),'white');text(im,(20,10),'Top-5 visual: v1 và v2 trên cùng ảnh hỏi',30,bold=True);text(im,(20,55),'Xanh = ĐÚNG · Đỏ = SAI · Query đầu có Top-5 khác nhau theo thứ tự CSV',22)
  for j,label in enumerate(['v1: nhãn 1–12','v2: nhãn đã kiểm tra']):
   y=105+j*400
   for col,r in enumerate([None]+rr[j][q]):
    p=q if r is None else r['candidate_id'];photo,_=load(z,idx[p]);x=col*260+8;put(im,photo,(x,y,240,275),None if r is None else ('#00834b' if int(r['is_correct']) else '#cc3344'));caption=f'{label}\nẢnh hỏi {Path(q).name}' if r is None else f"#{r['rank']} · {'ĐÚNG' if int(r['is_correct']) else 'SAI'} · SP {r['product_id']}\nCosine {float(r['score']):.4f}";text(im,(x,y+285),caption,20)
  save(im,out/'hinh_03_top5_v1_v2.png')
  black=[r for r in read(art/'crop_audit_v2.csv') if r['all_black']=='1'][:12];height=120+360*((len(black)+3)//4);im=Image.new('RGB',(1440,max(height,300)),'white');text(im,(20,10),'Ảnh còn đen ở v2: không có pixel nhãn giữ',29,bold=True)
  for j,r in enumerate(black):
   photo,_=load(z,idx[r['image_path']]);x=(j%4)*360+10;y=75+(j//4)*360;put(im,photo,(x,y,330,270));text(im,(x+5,y+280),f"SP {r['product_id']} · {r['split']}\nNhãn {r['labels']}",20)
  if not black:text(im,(20,100),'Không có ảnh đen ở cấu hình này.',28)
  save(im,out/'hinh_06_anh_den_v2.png')
 im=Image.new('RGB',(1560,700),'white');text(im,(25,15),'Pipeline v2 · 586 ảnh hỏi · cùng split tuần 2',32,bold=True);d=ImageDraw.Draw(im)
 blocks=[(15,190,'Ảnh + mask'),(240,230,'Lọc nhãn\ntrang phục v2'),(505,260,'BBox crop + pad\n224 × 224'),(800,250,'ResNet-50\n2.048 chiều, L2'),(1085,250,'Cosine với\n1.543 ứng viên'),(1370,175,'Top-10')]
 for x,w,t in blocks:d.rounded_rectangle((x,150,x+w,280),radius=12,fill='#edf4f8',outline='#416680',width=3);text(im,(x+12,174),t,23,bold=True)
 for (x,w,_),(xx,_,_) in zip(blocks,blocks[1:]):d.line((x+w,215,xx,215),fill='#416680',width=4);d.polygon([(xx,215),(xx-10,208),(xx-10,222)],fill='#416680')
 for x,w,t in [(80,480,'Popularity giả lập · seed 42\nGiữ nguyên dữ liệu v1'),(640,440,'Xếp giảm dần\nTie-break hash cố định'),(1170,300,'Top-10\nGiống byte với v1')]:d.rounded_rectangle((x,395,x+w,525),radius=12,fill='#edf4f8',outline='#416680',width=3);text(im,(x+15,420),t,23,bold=True)
 for x,xx in [(560,640),(1080,1170)]:d.line((x,460,xx,460),fill='#416680',width=4);d.polygon([(xx,460),(xx-10,453),(xx-10,467)],fill='#416680')
 text(im,(25,590),'Giữ nhãn: '+', '.join(map(str,keep)),24);text(im,(25,635),'Không huấn luyện · Không gộp ảnh sản phẩm · Không kết hợp hai baseline',23)
 save(im,out/'hinh_04_so_do_pipeline.png')
 os.environ['MPLCONFIGDIR']='/private/tmp/week4_v2_mpl';import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;from matplotlib.font_manager import FontProperties
 plt.rcParams['font.family']=FontProperties(fname=FONT).get_name();san=json.load(open(art/'sanity_checks_v2.json'))['results'];keys=['v1_visual','v2_visual','popularity_only'];x=np.arange(3);w=.34;fig,ax=plt.subplots(figsize=(8.8,4.8),dpi=180)
 for delta,key,label,col in [(-w/2,'top1','Top-1','#227c9d'),(w/2,'recall_at_5','Recall@5','#e89b38')]:
  bars=ax.bar(x+delta,[100*san[k][key] for k in keys],w,label=label,color=col);ax.bar_label(bars,labels=[f'{100*san[k][key]:.4f}%' for k in keys],padding=4,fontsize=9)
 ax.set_xticks(x,['Visual v1\nnhãn 1–12','Visual v2','Popularity-only']);ax.set_ylim(0,max(40,max(v for r in san.values() for v in r.values())*115));ax.set_ylabel('Tỉ lệ (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right');ax.set_title('Kiểm tra nhanh pipeline, chưa phải bảng đánh giá chính thức',fontsize=12,pad=18);fig.tight_layout();fig.savefig(out/'hinh_05_so_sanh_sanity.png',dpi=180,facecolor='white');plt.close(fig)
 dump(out/'selections.json',{'crop_paths':ps,'crop_random_seed':42,'changed_query':q,'black_paths':[r['image_path'] for r in black]})
 caps=[('hinh_01_atlas_nhan.png','mapping/mapping_de_xuat.csv, atlas_samples.csv; archive.zip','Đề xuất mapping thực nghiệm; mỗi nhãn minh họa bằng mẫu đầu trong atlas ngẫu nhiên, không phải xác nhận chính thức từ nhà cung cấp.'),('hinh_02_crop_v1_v2.png','archive.zip, common.crop; decision.json','Ảnh gốc và crop hai cấu hình. Hai ảnh chỉ định cùng hai ảnh chọn bằng Random(42), không lựa chọn theo mức đúng/sai.'),('hinh_03_top5_v1_v2.png','hai CSV visual Top-10 v1/v2; ground truth','Query đầu theo CSV có danh sách Top-5 thay đổi. Mẫu chọn có điều kiện, không ngẫu nhiên; ảnh hiển thị gốc, điểm từ masked vector.'),('hinh_04_so_do_pipeline.png','run_v2.py, decision.json','Hai baseline độc lập; chỉ sửa tập nhãn giữ của visual. Popularity được sao chép nguyên byte.'),('hinh_05_so_sanh_sanity.png','sanity_checks_v2.json','Chỉ Top-1 và Recall@5 để kiểm tra pipeline, không kết luận ý nghĩa thống kê.'),('hinh_06_anh_den_v2.png','crop_audit_v2.csv, archive.zip','Tối đa mười hai ảnh đầu theo index còn đen; hiển thị ảnh gốc và nhãn mask, không phải mẫu ngẫu nhiên.')]
 s='# Mô tả hình v2\n\nPNG nền trắng, 180 dpi, rộng không quá 1.600 px; chữ Arial raster hóa vào PNG.\n\n'
 for i,(name,src,cap) in enumerate(caps,1):s+=f'## {name}\n\nNguồn: {src}.\n\nHình {i}. {cap}\n\n'
 (out/'mo_ta_hinh.md').write_text(s);print('Six v2 figures created')
if __name__=='__main__':main()
