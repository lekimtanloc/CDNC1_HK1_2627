import os
os.environ['MPLCONFIGDIR']='/private/tmp/week5b_mpl';os.environ['XDG_CACHE_HOME']='/private/tmp/week5b_cache'
from common_5c import *
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Arial','font.size':9})
metrics={s:read(O/f'metrics_{s}_5c.csv') for s in ('valid','test')};pairs={s:json.loads((O/f'paired_{s}_5c.json').read_text())['pairs'] for s in metrics};effects={s:json.loads((O/f'effects_{s}_5c.json').read_text()) for s in metrics};stats={s:json.loads((O/f'summary_{s}_5c.json').read_text()) for s in metrics};checks=[r for s in metrics for r in read(O/f'independent_{s}_checks_5c.csv')];assert all(r['match']=='True' for r in checks)
lookup={(r['split'],r['variant'],r['scope'],r['metric']):r for r in checks if r['scope']!='query'};ledger=[]
def record(s,c,scope,m,value):
 r=lookup[(s,c,scope,m)];assert abs(float(value)-float(r['independent']))<1e-12;ledger.append({'source':f'{scope}_{s}','location':f'{c}/{m}','reported':value,'independent':r['independent'],'match':True,'note':NOTE});return float(value)
def metric(s,c,m):return next(r for r in metrics[s] if r['variant']==c and r['metric']==m)
def cell(s,c,m):
 r=metric(s,c,m);v=[record(s,c,'aggregate',m+' '+k,r[k])*100 for k in ('value','ci_low','ci_high')];return f'{v[0]:.2f} [{v[1]:.2f}; {v[2]:.2f}]'
def table(s):
 md='\n'+NOTE+'\n\n| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |\n|---|---|---|---|---|---|---|---|\n'
 for c in VARIANTS+['B','F']:md+='| '+c+(' (đối chiếu 5b)' if c in ('B','F') else '')+' | '+' | '.join(cell(s,c,m) for m in ('Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','NDCG@10','MRR@10'))+' |\n'
 return md
# Recount bbox fallback versus black rendered image, do not conflate them.
counts=[]
for c in VARIANTS+['B','F']:
 p=BASE/'crop_audit_C_5b.csv' if c=='G0' else (W/'v2/artifacts_v2/crop_audit_v2.csv' if c=='B' else (BASE/'crop_audit_F_5b.csv' if c=='F' else O/f'crop_audit_{c}_5c.csv'))
 rr=read(p)
 for s in ('valid','test'):
  selected=[r for r in rr if r['split']==s];fallback=sum(int(r['empty_keep' if c=='B' else 'fallback']) for r in selected);black=sum(int(r['all_black']) for r in selected);assert black>=fallback
  counts.append({'variant':c,'split':s,'bbox_reference':'C' if c in VARIANTS else c,'images':len(selected),'fallback':fallback,'all_black':black,'black_without_bbox_fallback':black-fallback,'note':NOTE})
write(O/'fallback_counts_5c.csv',counts)
# Figure1: all new variants plus the two needed frame references.
fig,ax=plt.subplots(figsize=(8.8,4.5));cs=VARIANTS+['B','F']
for j,(s,label) in enumerate([('valid','Validation'),('test','Test hậu nghiệm')]):
 rr=[metric(s,c,'Recall@5') for c in cs];v=np.array([float(r['value']) for r in rr])*100;lo=np.array([float(r['ci_low']) for r in rr])*100;hi=np.array([float(r['ci_high']) for r in rr])*100;ax.bar(np.arange(len(cs))+(j-.5)*.36,v,.36,yerr=[v-lo,hi-v],capsize=3,label=label)
ax.set_xticks(range(len(cs)),cs);ax.set_ylim(bottom=0);ax.set_ylabel('Recall@5 (%) và CI 95%');ax.set_title('Khung bbox C cố định cho G0–G4; B/F là đối chiếu');ax.legend();fig.text(.5,.018,'HẬU NGHIỆM — không đổi baseline B2-v1; không kết luận nguyên nhân hoặc nhận diện người.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.055,1,1));fig.savefig(O/'hinh_01_recall_5c.png',dpi=180,facecolor='white');plt.close(fig)
labels=['G1−B: khung, trang phục','G0−G1: mặt/tóc, có mask','G2−G4: mặt/tóc, ảnh gốc','G2−G0: ngoài C, có mặt/tóc','G4−G1: ngoài C, bỏ mặt/tóc','G3−F: khung, chỉ mặt/tóc','G3−G0: đối chiếu nội dung'];keys=[a+'-'+b for a,b in PAIRS]
fig,axs=plt.subplots(1,2,figsize=(8.8,5.7),sharex=True)
for ax,s,title in zip(axs,('valid','test'),('Validation','Test hậu nghiệm')):
 for j,k in enumerate(keys):
  r=pairs[s][k]['Recall@5'];v=r['difference']*100;lo=r['ci_low']*100;hi=r['ci_high']*100;ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color='#32689c' if j in (1,2) else ('#c47c29' if j in (3,4) else '#777777'))
 ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Hiệu Recall@5 (điểm phần trăm)')
fig.text(.5,.02,'Hậu nghiệm; CI 95%; không đổi baseline. “Có” theo quy tắc thống kê, không kết luận nguyên nhân.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(O/'hinh_02_forest_5c.png',dpi=180,facecolor='white');plt.close(fig)
md='''# Kết quả tuần 05c — kiểm soát khung hình

## 1. Quyết định thiết kế và phạm vi
**BỔ SUNG HẬU NGHIỆM**, đặt sau khi đã thấy kết quả tuần 5 và 5b. **Không đổi baseline chính B2-v1**, dù bộ lọc baseline sai ngữ nghĩa. Không sửa protocol_v1.md, baseline_chinh.json, freeze.json hoặc bất kỳ hồ sơ cũ nào. Test đã được quan sát trước đây nên 5c không phải bằng chứng độc lập trên test chưa thấy.

Người dùng đã thay hoàn toàn thiết kế G0–G3 ban đầu bằng G0–G4 bên dưới. Màu tô và padding đều ĐEN (RGB 0,0,0), không dùng trung bình ImageNet. Chuẩn hóa đầu vào vẫn là Normalize ImageNet như pipeline cũ; pixel đen sau Normalize không đồng nghĩa tensor bằng không.

BBox C cố định cho mọi G: hợp nhãn [1,2,3,4,5,6,7,8,9,10,11,12,13,15]. Cùng bbox, cùng contain/BILINEAR vào 224×224, cùng căn giữa và padding. Nếu bbox C rỗng, mọi G trả ảnh đen như fallback v2, kể cả G2/G4. Nếu bbox C có nhưng nội dung giữ không còn pixel, ảnh có thể đen nhưng không được tính là bbox fallback.

'''+NOTE+'''\n
| Biến thể | Pixel trước crop | Bbox |
|---|---|---|
| G0 = C | Giữ nhãn C, ngoài C tô đen | C |
| G1 | Giữ nhãn B=[1,2,3,5,6,7,8,10,11,12,13,15], còn lại đen | C |
| G2 | Giữ ảnh gốc, không tô đen | C |
| G3 | Chỉ giữ nhãn 4,9, còn lại đen | C |
| G4 | Ảnh gốc, chỉ tô đen nhãn 4,9 | C |
| B (5b) | Cùng pixel như G1 | B |
| F (5b) | Cùng pixel như G3 | F={4,9} |

## 2. Tái lập và quy trình
design_5c.json ghi thiết kế, quy tắc đọc và trường hợp rỗng trước khi tính metric mới. G0 dùng đúng crop C tuần 4/5b, kiểm tra lại 200 ảnh trong 25 batch đầy đủ chọn seed42, giữ thứ tự/batch geometry của cache; tất cả khớp từng byte. Vì G0=C nên tái sử dụng toàn bộ cache C sau kiểm tra, không trích lại toàn bộ G0. G1–G4 chỉ trích validation/test, không trích train, không thay backbone hoặc weights. CPU, 4 luồng, batch8, deterministic; phiên bản môi trường và hash weights lưu trong manifest.

Ranking G0–G4 dựng từ vector chuẩn hóa bằng cosine (dot product), argsort stable theo gallery CSV, loại chính query, không gộp ảnh về sản phẩm. B/F dùng đúng Top-10 đã có ở 5b. Metric macro theo query; Top-1 là Hit@1; Recall@1 không đồng nhất Hit@1 khi nhiều đáp án. Binary NDCG, MRR@10 như protocol cũ. Bootstrap 10.000 lần, seed42, percentile95%, theo query. Validation trước; freeze_5c.json trước ranking/metric test; test5c chính thức một lần. Mã độc lập tính lại toàn bộ query, metric, aggregate, CI và paired; việc kiểm chứng không dùng để chọn lại cấu hình.

'''
for s in stats:md+=f"- {s}: {stats[s]['queries']} query; {stats[s]['gallery']} gallery; {stats[s]['truth_pairs']} cặp đúng.\n"
for c in VARIANTS[1:]:
 e=json.loads((O/f'extraction_{c}_5c.json').read_text());md+=f"- Trích {c}: {e['images']} ảnh, {e['seconds']:.3f} giây (tổng thời gian các checkpoint, không gồm nạp model).\n"
md+='\n'+NOTE+'\n\n| Biến thể | Tập | Bbox | Số ảnh | Fallback bbox rỗng | Tổng ảnh đen | Đen nhưng bbox không rỗng |\n|---|---|---|---:|---:|---:|---:|\n'
for r in counts:md+=f"| {r['variant']} | {r['split']} | {r['bbox_reference']} | {r['images']} | {r['fallback']} | {r['all_black']} | {r['black_without_bbox_fallback']} |\n"
md+='''
## 3. Metric validation
Đơn vị **%**, mỗi ô là giá trị [CI 95%]. CSV lưu tỉ lệ 0–1. B/F là đối chiếu cũ, không phải ứng viên thay baseline.
'''+table('valid')
md+='\n## 4. Metric test hậu nghiệm\nCùng cách đọc và đơn vị; không diễn giải đây là một test độc lập mới.\n'+table('test')
md+='''
## 5. Hiệu ghép cặp và quy tắc đọc
Nhóm mặt/tóc: G0−G1 và G2−G4. Nhóm nền/vùng ngoài C: G2−G0 và G4−G1. Chỉ ghi “có” nếu **cả hai CI Recall@5 loại 0 cùng chiều**, nếu không thì “không kết luận”. Quy tắc được ghi trước khi chấm validation. “Có” chỉ là kết quả của quy tắc đọc, không là kết luận cơ chế hay nhân quả. G1−B và G3−F chỉ báo số/CI; G3−G0 là đối chiếu mô tả không đặt ngưỡng chọn.

'''+NOTE+'\n\n| Tập | Hiệu | Metric | Hiệu (điểm %) | CI 95% (điểm %) |\n|---|---|---|---:|---|\n'
for s in pairs:
 for label,d in pairs[s].items():
  for m in ('Recall@5','NDCG@5','MRR@10'):
   v=[record(s,label,'paired',m+' '+k,d[m][k])*100 for k in ('difference','ci_low','ci_high')];md+=f'| {s} | {label} | {m} | {v[0]:.3f} | [{v[1]:.3f}; {v[2]:.3f}] |\n'
md+='\n'+NOTE+' McNemar exact hai phía: hàng là cấu hình bị trừ, cột là cấu hình đứng trước; thứ tự sai/đúng.\n\n| Tập | Hiệu | Bảng 2×2 | p exact |\n|---|---|---|---:|\n'
for s in pairs:
 for label,d in pairs[s].items():
  p=record(s,label,'paired','McNemar p',d['McNemar']['p_exact_two_sided']);t=d['McNemar']['table_rows_subtrahend_cols_minuend'];md+=f'| {s} | {label} | {t} | {p:.6g} |\n'
  for i in (0,1):
   for j in (0,1):record(s,label,'paired',f'McNemar cell {i},{j}',t[i][j])
md+='\n'+NOTE+'\n\n| Tập | Nhóm | Kết luận quy tắc | Chiều |\n|---|---|---|---|\n'
for s in effects:
 for key,title in [('face_hair','Mặt/tóc'),('background','Nền/vùng ngoài C')]:
  e=effects[s][key];md+=f"| {s} | {title} | {e['conclusion']} | {e['direction'] or 'không xác định'} |\n"
for key,title in [('face_hair','Mặt/tóc'),('background','Nền/vùng ngoài C')]:
 md+=f"\n{title}: validation **{effects['valid'][key]['conclusion']}**, test **{effects['test'][key]['conclusion']}**, theo đúng quy tắc hai CI đã ghi trước. Không suy nguyên nhân từ phân loại này.\n"
md+='''
Không gọi cấu hình tốt hơn ở metric đang xét nếu CI hiệu chứa 0. Không dùng kết quả test để chọn baseline, biến thể, nhóm hiệu ứng hoặc cách đọc khác. Các CI/p-value chưa điều chỉnh đa kiểm định; mọi so sánh đều hậu nghiệm so với toàn bộ chuỗi tuần trước.

## 6. G3, F và mức ngẫu nhiên
'''+NOTE+'\n\n| Tập | G3 Top-1 (%) | F Top-1 (%) | G0 Top-1 (%) | E[Top-1] (%) | G3 Recall@5 (%) | F Recall@5 (%) | G0 Recall@5 (%) | E[Recall@5] (%) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for s in stats:
 vals=[float(metric(s,c,'Top-1')['value'])*100 for c in ('G3','F','G0')]+[stats[s]['random_Top-1']*100]+[float(metric(s,c,'Recall@5')['value'])*100 for c in ('G3','F','G0')]+[stats[s]['random_Recall@5']*100];md+='| '+s+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |\n'
md+='''
Mức ngẫu nhiên E[Top-1]=tổng Rq/(Q·G), E[Recall@5]=5/G; so với ngẫu nhiên chỉ mô tả, chưa kiểm định null. G3−F giữ cùng pixel được chọn nhưng đổi bbox/scale/padding, nên là đối chiếu khung; không được đọc như việc thêm thông tin mặt/tóc. G3 so G0 chỉ mô tả khả năng truy hồi của phần pixel nhãn 4,9 trong khung C. Dù G3 cao hơn ngẫu nhiên hoặc thấp hơn G0, điều đó không chứng minh mô hình nhận diện người mẫu hoặc không dùng tín hiệu người, vì không có nhãn định danh người.

## 7. Giới hạn, điều chưa kiểm tra và diễn giải yêu cầu
Khung cố định kiểm soát bbox/scale/padding giữa các G nhưng thao tác tô đen vẫn thay tương phản, đường biên và phân bố pixel. G2 vẫn dùng mask C để xác định bbox, nên không phải đối chứng hoàn toàn không dùng mask. Tên vùng dựa trên mapping thực nghiệm chưa có tài liệu chính thức, mask có thể sai. “Nền/vùng ngoài C” gồm cả nhãn0 và nhãn14 (nền và da), không phải nền thuần túy. Không quy chênh lệch cho một cơ chế cụ thể. Ground truth là cùng sản phẩm khác ảnh, không là sở thích hay giao dịch. CI theo query có điều kiện trên gallery cố định, không bao phủ biến thiên split, seed hay đổi gallery. Không có nhãn người để kiểm chứng nhận diện người mẫu.

Chưa kiểm tra được: nguyên nhân/cơ chế, hiệu quả người dùng hay lượt mua, nhãn người, độ bền nhiều split/seed và peak RAM toàn đợt. Test đã được thấy trước đây nên không xem CI là bằng chứng độc lập ngoài chuỗi hậu nghiệm.

Diễn giải thực thi: G0 tái dùng cache C sau kiểm tra200 ảnh thay vì trích lại toàn bộ; G1–G4 trích mới. B/F giữ nguyên ranking5b làm đối chiếu. Fallback theo bbox C rỗng được áp dụng cả ảnh gốc G2/G4, đúng fallback đen pipeline cũ. Tính thêm CI và McNemar cho G3−G0 để minh bạch đối chiếu mô tả, không đặt ngưỡng. NDCG@10 được giữ để đủ metric retrieval của protocol. Checkpoint mỗi256 ảnh, batch vẫn8. Báo cáo dùng tên ket_qua_tuan_05c.md, còn các artifact khác hậu tố _5c; ZIP hồ sơ không phải gói phần mềm/demo. Thiết kế G0–G3 cũ và màu trung bình ImageNet đã được hủy theo quyết định người dùng.

## 8. Hình, minh chứng và tái chạy
Hình 1. Recall@5 G0–G4 cùng B/F tham chiếu trên validation/test, CI95%, trục y từ0; nguồn metrics_valid_5c.csv và metrics_test_5c.csv. Hình 2. Forest plot hiệu ghép cặp Recall@5 theo các tương phản; nguồn paired_valid_5c.json và paired_test_5c.json. Cả hai hình nền trắng, tiếng Việt, 180dpi, rộng không quá1600px; ghi rõ hậu nghiệm và baseline không đổi.

Xem manifest_5c.json, doi_chieu_so_lieu_5c.csv, verification_5c.txt, hash_cu_truoc_5c.json và hash_cu_sau_5c.json. Mã nguồn nằm trong frame_control_5c, không sửa mã cũ. commands_5c.txt ghi thứ tự đã chạy. Checkpoint có hash vector/audit/source/design, có thể tiếp tục trong trường hợp chỉ checkpoint đã commit được tái dùng. Không chạy lại vào thư mục giao nộp; dùng thư mục sibling mới cùng độ sâu. ZIP chứa toàn bộ thư mục5c, gồm vector/checkpoint, không chứa chính ZIP và không thay gói5b cũ.
'''
for s in stats:
 gt=read(P/f'ground_truth_{s}.csv');sr=read(P/f'{s}.csv');q=len({r['query_image_path'] for r in gt});g=sum(r['role']=='gallery' for r in sr);ind={'queries':q,'gallery':g,'truth_pairs':len(gt),'random_Top-1':len(gt)/q/g,'random_Recall@5':5/g}
 for key,v in ind.items():assert abs(stats[s][key]-v)<1e-12;ledger.append({'source':f'summary_{s}_5c.json','location':key,'reported':stats[s][key],'independent':v,'match':True,'note':NOTE})
for c in VARIANTS[1:]:
 e=json.loads((O/f'extraction_{c}_5c.json').read_text());secs=sum(json.loads(p.read_text())['seconds'] for p in (O/'checkpoints_5c').glob(c+'_*_5c.json'));assert abs(secs-e['seconds'])<1e-9;ledger.append({'source':f'extraction_{c}_5c.json','location':'seconds','reported':e['seconds'],'independent':secs,'match':True,'note':'sum immutable checkpoint timings; not a repeat benchmark. '+NOTE})
for r in counts:
 for key in ('images','fallback','all_black','black_without_bbox_fallback'):ledger.append({'source':'fallback_counts_5c.csv','location':f"{r['split']}/{r['variant']}/{key}",'reported':r[key],'independent':r[key],'match':'recounted source audit; final verification checks again','note':NOTE})
write(O/'doi_chieu_so_lieu_5c.csv',ledger)
with (O/'ket_qua_tuan_05c.md').open('x') as f:f.write(md)
print('Report, figures, and ledger written',flush=True)
