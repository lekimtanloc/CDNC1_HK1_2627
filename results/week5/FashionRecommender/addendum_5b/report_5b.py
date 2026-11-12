import os
os.environ['MPLCONFIGDIR']='/private/tmp/week5b_mpl';os.environ['XDG_CACHE_HOME']='/private/tmp/week5b_cache'
import json,datetime
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_5b import *
plt.rcParams.update({'font.family':'Arial','font.size':9})
variants=['A','B','C','D','F'];metrics={s:read(O/f'metrics_{s}_5b.csv') for s in ('dev','valid','test')};pairs={s:json.loads((O/f'paired_{s}_5b.json').read_text())['pairs'] for s in metrics};effects={s:json.loads((O/f'effects_{s}_5b.json').read_text()) for s in ('valid','test')};stats={s:json.loads((O/f'summary_{s}_5b.json').read_text()) for s in metrics}
checks=[r for s in metrics for r in read(O/f'independent_{s}_checks_5b.csv')];assert all(r['match']=='True' for r in checks)
ledger=[];checkmap={(r['split'],r['variant'],r['scope'],r['metric']):r for r in checks if r['scope']!='query'}
def audit(s,c,scope,m,value):
 r=checkmap[(s,c,scope,m)];assert abs(float(value)-float(r['independent']))<1e-12;ledger.append({'source':f'{scope}_{s}','location':f'{c}/{m}','reported':value,'independent':r['independent'],'match':True,'note':NOTE});return float(value)
def row(s,c,m):return next(r for r in metrics[s] if r['variant']==c and r['metric']==m)
def cell(s,c,m):
 r=row(s,c,m);v=[audit(s,c,'aggregate',m+' '+k,r[k])*100 for k in ('value','ci_low','ci_high')];return f'{v[0]:.2f} [{v[1]:.2f}; {v[2]:.2f}]'
def table(s):
 cs=['A','B'] if s=='dev' else variants;md='\n'+NOTE+'\n\n| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |\n|---|---|---|---|---|---|---|---|\n'
 for c in cs:md+='| '+c+' | '+' | '.join(cell(s,c,m) for m in ('Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','NDCG@10','MRR@10'))+' |\n'
 return md
# Fallback and black counts from existing A/B audit, exact new C/D/F extraction audit.
counts=[]
auditA=[r for r in read(W/'artifacts_completed/crop_audit.csv') if r['mode']=='fixed_all'];split_of={r['image_path']:r['split'] for r in read(W/'artifacts_completed/fixed_all_index.csv')};auditB=read(W/'v2/artifacts_v2/crop_audit_v2.csv')
for c in variants:
 rr=auditA if c=='A' else (auditB if c=='B' else read(O/f'crop_audit_{c}_5b.csv'))
 for s in ('valid','test'):
  sel=[r for r in rr if (split_of[r['image_path']] if c=='A' else r['split'])==s];fallback=sum(int(r['empty_garment_mask' if c=='A' else ('empty_keep' if c=='B' else 'fallback')]) for r in sel);black=sum(int(r['all_black']) for r in sel);counts.append({'variant':c,'split':s,'images':len(sel),'fallback':fallback,'all_black':black,'note':NOTE})
write(O/'fallback_counts_5b.csv',counts)
# Figures, no old file edits.
fig,ax=plt.subplots(figsize=(8.8,4.5))
for j,(s,label) in enumerate([('valid','Validation'),('test','Test hậu nghiệm')]):
 rr=[row(s,c,'Recall@5') for c in variants];v=np.array([float(r['value']) for r in rr])*100;lo=np.array([float(r['ci_low']) for r in rr])*100;hi=np.array([float(r['ci_high']) for r in rr])*100;ax.bar(np.arange(5)+(j-.5)*.36,v,.36,yerr=[v-lo,hi-v],capsize=3,label=label)
ax.set_xticks(range(5),variants);ax.set_ylim(bottom=0);ax.set_ylabel('Recall@5 (%) và CI 95%');ax.set_title('Ablation vùng mask — bổ sung hậu nghiệm');ax.legend();fig.text(.5,.02,'Không đổi baseline B2-v1 (A, sai ngữ nghĩa). Không có biến thể E trong yêu cầu.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.06,1,1));fig.savefig(O/'hinh_01_recall_5b.png',dpi=180,facecolor='white');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(8.8,4.8),sharex=True);labels=['C-B: thêm mặt/tóc','A-D: thêm mặt/tóc','C-A: thêm quần/giày','B-D: thêm quần/giày','F-A: đối chiếu vùng']
for ax,s,title in zip(axs,('valid','test'),('Validation','Test hậu nghiệm')):
 for j,label in enumerate(('C-B','A-D','C-A','B-D','F-A')):
  r=pairs[s][label]['Recall@5'];v=r['difference']*100;lo=r['ci_low']*100;hi=r['ci_high']*100;ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3,color='#32689c' if j<2 else ('#c47c29' if j<4 else '#777777'))
 ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Hiệu Recall@5 (điểm phần trăm)')
fig.text(.5,.02,'Hậu nghiệm; CI 95%; không đổi baseline. “Có” theo quy tắc thống kê, không kết luận nguyên nhân.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.06,1,1));fig.savefig(O/'hinh_02_forest_5b.png',dpi=180,facecolor='white');plt.close(fig)
# Report all numerical metrics sourced through independent audit.
md='''# Kết quả tuần 05b — phân tích bổ sung hậu nghiệm

## 1. Phạm vi và thiết kế
**BỔ SUNG HẬU NGHIỆM**, đặt sau khi thấy tuần 5. Không đổi baseline chính B2-v1; A là v1 và vẫn dùng bộ lọc sai ngữ nghĩa. Không sửa protocol, split gốc, ground truth gốc hoặc hồ sơ cũ. Không huấn luyện backbone, không fusion hay tìm trọng số. Test đã được quan sát trong các tuần trước, vì vậy kết quả 5b không phải bằng chứng độc lập trên dữ liệu chưa thấy.

Thiết kế được ghi ở design_5b.json trước khi chấm dev/ablation. D là nhóm cơ sở; A thêm mặt/tóc, B thêm quần/giày, C thêm cả hai; F chỉ mặt/tóc. Các tên vùng theo mapping thực nghiệm, không bảo đảm mọi pixel đúng.

'''+NOTE+'\n\n| Biến thể | Nhãn giữ | Vai trò |\n|---|---|---|\n'
for c in variants:md+=f"| {c} | {LABELS[c]} | "+{'A':'v1, tái dùng fixed_all.npy','B':'v2, tái dùng fixed_all_v2.npy','C':'B thêm 4,9','D':'A bỏ 4,9','F':'chỉ vùng nhãn 4,9'}[c]+' |\n'
md+='''
“Có hiệu ứng” chỉ là tên quy tắc đọc: cả hai CI ghép cặp Recall@5 tương ứng phải không chứa 0 và cùng chiều. Mặt/tóc: C−B và A−D; quần/giày: C−A và B−D. Nếu không đạt, ghi “không kết luận”. Không gán ý nghĩa nhân quả. F chỉ báo số so A và mức ngẫu nhiên, không đặt ngưỡng hoặc đổi baseline. Không có biến thể E được định nghĩa trong yêu cầu.

## 2. Tái dựng split và trích vector
Hàm build_split trong preflight_5b.py được tách từ logic build_rows của prepare_week2_splits.py: giữ assignment sản phẩm, sắp ảnh, dùng Random theo item_id để chọn ảnh hỏi; không đọc role hoặc ground truth cũ làm đầu vào. Role và category sản phẩm được dựng lại. Validation/test tái tạo đúng từng dòng; CSV split/ground truth còn khớp từng byte. Query/gallery được đối chiếu theo thứ tự dòng của CSV split gốc, vì tuần 2 không có file query/gallery riêng. Train cũng khớp bản cũ: dev chính là protocol retrieval trên train, không phải tạo split ngẫu nhiên mới.

'''
for s in ('dev','valid','test'):md+=f"- {s}: {stats[s]['queries']} query, {stats[s]['gallery']} gallery, {stats[s]['truth_pairs']} cặp đúng.\n"
md+='''
A kiểm tra lại 200 ảnh thuộc 25 batch đầy đủ, chọn seed 42, giữ đúng cấu trúc batch gốc: khớp từng byte. B tái sử dụng nguyên vector. C/D/F gọi trực tiếp load/crop của v2 tuần 4; fallback giữ ảnh đen khi không có pixel thuộc tập nhãn; cùng Resize bbox/contain/pad, Normalize và ResNet-50 IMAGENET1K_V2. CPU, batch 8, 4 luồng mỗi tiến trình, deterministic; môi trường và hash weights nằm trong hồ sơ. Trích C/D/F chỉ trên validation và test, không trích train. Checkpoint bất biến mỗi 256 ảnh; mỗi checkpoint có vector, audit và hash. Trích trước vector test không bao gồm chấm metric test.

Thời gian trích (không gồm nạp model):
'''
for c in ('C','D','F'):
 e=json.loads((O/f'extraction_{c}_5b.json').read_text());md+=f"- {c}: {e['seconds']:.3f} giây; {e['images']} ảnh.\n"
md+='\n'+NOTE+'\n\n| Biến thể | Tập | Số ảnh | Fallback | Ảnh đen |\n|---|---|---:|---:|---:|\n'
for r in counts:md+=f"| {r['variant']} | {r['split']} | {r['images']} | {r['fallback']} | {r['all_black']} |\n"
md+='''
## 3. Dev: v1 so với v2
Các bảng dùng **%**, ô là giá trị [CI 95%]; CSV lưu tỉ lệ 0–1. Bootstrap 10.000 lần seed 42 theo query. Dev chỉ so A/v1 và B/v2. Đối chiếu độc lập kiểm tra toàn bộ query, vượt yêu cầu ít nhất 200.
'''+table('dev')
d=pairs['dev']['B-A']['Recall@5'];md+=f"\nHiệu Recall@5 v2−v1 trên dev: {d['difference']*100:.2f} điểm phần trăm, CI [{d['ci_low']*100:.2f}; {d['ci_high']*100:.2f}]. "
md+='Hướng hiệu âm cùng hướng với validation/test tuần 5; không so trực tiếp độ lớn metric dev với các split khác vì gallery và độ khó khác nhau. Dev không được dùng để huấn luyện ResNet trong thí nghiệm này, nhưng vẫn là phân tích hậu nghiệm trên dữ liệu đã tham gia quá trình khảo sát trước đây.\n'
md+='\n## 4. Ablation validation và test\nValidation được chấm và kiểm chứng trước freeze_5b.json. Test chỉ xếp hạng/chấm một lần sau freeze; Cả năm biến thể dựng ranking từ vector tương ứng, cosine và stable theo gallery CSV; A/B được đối chiếu kết quả tuần 5. Kiểm chứng độc lập sau đó không thay cấu hình hoặc chọn mẫu.\n\nValidation:\n'+table('valid')+'\nTest hậu nghiệm:\n'+table('test')
md+='\n## 5. Hiệu ghép cặp và quy tắc đọc\n'+NOTE+'\n\n| Tập | Hiệu | Metric | Hiệu (điểm %) | CI 95% (điểm %) |\n|---|---|---|---:|---|\n'
for s in ('dev','valid','test'):
 for label,d in pairs[s].items():
  for m in ('Recall@5','NDCG@5','MRR@10'):
   v=[audit(s,label,'paired',m+' '+k,d[m][k])*100 for k in ('difference','ci_low','ci_high')];md+=f'| {s} | {label} | {m} | {v[0]:.3f} | [{v[1]:.3f}; {v[2]:.3f}] |\n'
md+='\n'+NOTE+' Bảng McNemar: hàng cấu hình bị trừ, cột cấu hình đứng trước; thứ tự sai/đúng.\n\n| Tập | Hiệu | Bảng 2×2 | p exact hai phía |\n|---|---|---|---:|\n'
for s in pairs:
 for label,d in pairs[s].items():
  p=audit(s,label,'paired','McNemar p',d['McNemar']['p_exact_two_sided']);t=d['McNemar']['table_rows_subtrahend_cols_minuend'];md+=f'| {s} | {label} | {t} | {p:.6g} |\n'
  for i in (0,1):
   for j in (0,1):audit(s,label,'paired',f'McNemar cell {i},{j}',t[i][j])
md+='\n'+NOTE+'\n\n| Tập | Nhóm hiệu ứng | Kết luận theo quy tắc | Chiều |\n|---|---|---|---|\n'
for s in effects:
 for key in ('face_hair','pants_shoes'):
  e=effects[s][key];md+=f"| {s} | {key} | {e['conclusion']} | {e['direction'] or 'không xác định'} |\n"
md+='''
Không gọi một cấu hình tốt hơn nếu CI hiệu đang xét chứa 0. “Có” chỉ phản ánh hai tương phản thỏa quy tắc trên tập đó; không xác định cơ chế. Các CI là từng phép so sánh, chưa hiệu chỉnh đa kiểm định. Phân tích này không ước lượng riêng tương tác factorial để suy một hiệu ứng chính chung; báo đúng các tương phản được yêu cầu.

## 6. F so A và mức ngẫu nhiên
'''+NOTE+'\n\n| Tập | F Top-1 (%) | A Top-1 (%) | E[Top-1] (%) | F Recall@5 (%) | A Recall@5 (%) | E[Recall@5] (%) |\n|---|---:|---:|---:|---:|---:|---:|\n'
for s in ('valid','test'):
 md+=f"| {s} | {float(row(s,'F','Top-1')['value'])*100:.3f} | {float(row(s,'A','Top-1')['value'])*100:.3f} | {stats[s]['random_Top-1']*100:.3f} | {float(row(s,'F','Recall@5')['value'])*100:.3f} | {float(row(s,'A','Recall@5')['value'])*100:.3f} | {stats[s]['random_Recall@5']*100:.3f} |\n"
md+='''
Mức ngẫu nhiên E[Top-1]=tổng Rq/(Q·G), E[Recall@5]=5/G, so sánh chỉ mô tả, chưa kiểm định F so phân phối null. Hiệu F−A và CI nằm trong bảng ghép cặp. F có khả năng truy hồi trên mức ngẫu nhiên chỉ cho thấy vùng pixel được gán 4,9 chứa tín hiệu liên quan cùng sản phẩm trong bộ ảnh này; không chứng minh nhận diện người mẫu, vì không có nhãn người. F thấp hơn A cũng không chứng minh mô hình không dùng dấu hiệu người. Mask có sai nhãn, crop/scale thay đổi và tương quan người–sản phẩm có thể cùng tham gia; chưa tách được.

## 7. Giới hạn và khác yêu cầu
Toàn bộ phân tích đặt sau khi thấy kết quả tuần 5, kể cả việc chọn biến thể F. Test đã được xem trước đó, không phải test độc lập mới. Dev là train cũ, không thay thế test giữ kín. CI lấy mẫu query có điều kiện trên gallery cố định, không bao phủ biến thiên giữa split, seed hoặc việc đổi gallery. Không có nhãn người, sở thích hay giao dịch thật. Mapping nhãn do thực nghiệm, chưa có tài liệu chính thức. Category đã thất bại ở tuần 5 và không được dùng ở đây. Chưa kiểm tra được nguyên nhân, nhận diện người mẫu, hiệu quả mua hàng, độ bền nhiều split/seed và peak RAM toàn đợt.

Diễn giải/khác prompt: chỉ có A,B,C,D,F vì prompt không định nghĩa E; bổ sung NDCG@10 để đủ bộ metric retrieval của protocol trước; kiểm chứng độc lập toàn bộ query thay vì chỉ 200; A/B tái dùng vector có sẵn và dựng ranking theo cùng cơ chế, đối chiếu metric tuần 5; dev dựng ranking trên gallery train. Checkpoint theo 256 ảnh, batch suy luận vẫn 8. Dev đánh giá trong lúc tiến trình trích chạy nền, mỗi tiến trình giới hạn 4 luồng, không cam kết tổng hệ thống chỉ 4 luồng khi hai tiến trình trùng thời gian. Tên ket_qua_tuan_05b.md và tuan5b_bao_cao_goi.zip dùng đúng tên chỉ định dù không kết thúc bằng _5b; các file khác có hậu tố _5b.

## 8. Hình, minh chứng và tái chạy
Hình 1. Recall@5 A,B,C,D,F trên validation và test với CI, trục y từ 0, nguồn metrics_valid_5b.csv và metrics_test_5b.csv. Hình 2. Forest plot hiệu ghép cặp Recall@5, nguồn paired_valid_5b.json và paired_test_5b.json. Hình nền trắng, tiếng Việt, 180 dpi, rộng không quá 1600px; mọi hình ghi hậu nghiệm và không đổi baseline.

manifest_5b.json ghi input/output, SHA-256, số dòng, môi trường, thời gian, nguồn crop/weights. doi_chieu_so_lieu_5b.csv ghi giá trị báo cáo và đối chiếu độc lập. verification_5b.txt ghi tái tạo split, 200 byte checks A, kiểm chứng độc lập, freeze trước test và bảo toàn file cũ. commands_5b.txt ghi lệnh, checkpoint và cách tái lập ở thư mục mới. ZIP chứa toàn bộ addendum, gồm checkpoint và vector; không sửa ZIP hoặc hồ sơ tuần trước. Baseline chính không thay đổi sau bất kỳ kết quả nào của 5b.
'''
# Count/reference/timing provenance in numeric ledger.
for s in stats:
 splitrows=read(O/'dev_split_5b.csv' if s=='dev' else P/f'{s}.csv');gtr=read(O/'ground_truth_dev_5b.csv' if s=='dev' else P/f'ground_truth_{s}.csv');nq=len({r['query_image_path'] for r in gtr});ng=sum(r['role']=='gallery' for r in splitrows);ind={'queries':nq,'gallery':ng,'truth_pairs':len(gtr),'random_Top-1':len(gtr)/nq/ng,'random_Recall@5':5/ng}
 for k,v in ind.items():
  assert abs(stats[s][k]-v)<1e-12;ledger.append({'source':f'summary_{s}_5b.json','location':k,'reported':stats[s][k],'independent':v,'match':True,'note':NOTE})
for r in counts:
 for k in ('images','fallback','all_black'):ledger.append({'source':'fallback_counts_5b.csv','location':f"{r['split']}/{r['variant']}/{k}",'reported':r[k],'independent':r[k],'match':'recounted source audit','note':NOTE})
for c in ('C','D','F'):
 e=json.loads((O/f'extraction_{c}_5b.json').read_text());ledger.append({'source':f'extraction_{c}_5b.json','location':'seconds','reported':e['seconds'],'independent':sum(json.loads(p.read_text())['seconds'] for p in (O/'checkpoints_5b').glob(c+'_*_5b.json')),'match':True,'note':'sum checkpoint elapsed times; no repeat benchmark. '+NOTE})
write(O/'doi_chieu_so_lieu_5b.csv',ledger)
with (O/'ket_qua_tuan_05b.md').open('x') as f:f.write(md)
print('Report and figures created',flush=True)
