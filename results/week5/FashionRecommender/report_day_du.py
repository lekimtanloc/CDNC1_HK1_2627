import json,csv,datetime,math
from pathlib import Path
from collections import Counter
import openpyxl
from run_test_day_du import O,R,read,write,dump,configs,WARN,sha
metrics={'valid':read(O/'metrics_valid.csv'),'test':read(O/'metrics_test_day_du.csv')};paired=json.loads((O/'paired_day_du.json').read_text());groups=read(O/'group_metrics_day_du.csv');refs=read(O/'reference_levels_day_du.csv');checks=read(O/'independent_test_checks.csv');gchecks=read(O/'independent_groups_day_du.csv');ledger=[]
lookup={(r['split'],r['config'],r['scope'],r['metric']):r for r in checks if r['scope']!='per_query'}
def audited(split,c,scope,m,value,source,scale=1):
 r=lookup[(split,c,scope,m)];assert abs(float(value)-float(r['independent']))<1e-12
 ledger.append({'source':source,'location':f'{split}/{c}/{scope}/{m}','reported_numeric':float(value)*scale,'independent_numeric':float(r['independent'])*scale,'match':True,'method':'independent CSV-rank implementation; see independent_test_checks.csv'})
 return float(value)*scale
def metric_cell(s,c,m,scale):
 r=next(r for r in metrics[s] if r['config']==c and r['metric']==m);vals=[audited(s,c,'aggregate',m+' '+k,r[k],'metrics_valid.csv' if s=='valid' else 'metrics_test_day_du.csv',scale) for k in ('value','ci_low','ci_high')]
 return f'{vals[0]:.6f} [{vals[1]:.6f}; {vals[2]:.6f}]' if scale==1 else f'{vals[0]:.2f} [{vals[1]:.2f}; {vals[2]:.2f}]'
def table(s,scale):
 out='\n'+WARN+' B2p và B2-v2 được giữ làm đối chiếu.\n\n| Metric | B0 | B2p | B2-v1 (sai ngữ nghĩa) | B2-v2 |\n|---|---|---|---|---|\n'
 for m in ('Top-1','Recall@1','Recall@5','Recall@10','NDCG@5','NDCG@10','MRR@10','Popularity MAE'):
  out+='| '+m+' | '+' | '.join(metric_cell(s,c,m,scale) for c in configs)+' |\n'
 return out
# Verify preservation before reporting completion.
old=json.loads((O/'hash_cu_truoc_day_du.json').read_text());old5=json.loads((O/'hash_week5_truoc_day_du.json').read_text())
assert all(sha(R/p)==h for p,h in old.items());assert all(sha(O/p)==h for p,h in old5.items())
# Copy Excel; requested historical wording is supplied by user, not an API re-test.
w=openpyxl.load_workbook(O/'ke_hoach_do_an_cap_nhat.xlsx');s=w['file kế hoạch thực hiện'];styles={cell.coordinate:cell.style_id for row in s for cell in row}
for row in (14,15,16,17,18):s[f'F{row}']='100%'
s['E16']='Kiểm tra 13.752 cặp ảnh–mask; xác nhận ground truth cùng sản phẩm; bổ sung kiểm thử hàm Recall/NDCG/MRR. Trích vector ResNet-50 cho 2.129 ảnh test; chấm Top-1 30,20%, Recall@5 32,20%. Triển khai Recommendation API và Event API, lưu gợi ý và tương tác vào PostgreSQL. Phát hiện lỗi CenterCrop khiến 100/2.129 ảnh test bị đen.'
s['D17']='Xây baseline popularity-only và visual-only trên split cố định theo ground truth cùng sản phẩm (không dùng ground truth theo category).'
s['E17']='Sửa crop bằng bbox; kiểm tra lại mapping nhãn mask (v2). Trích lại vector cho 13.752 ảnh; xuất Top-10 cho visual-only (v1, v2) và popularity-only. Kiểm tra nhanh: Top-1 / Recall@5 của v2 là 31,23% / 29,61%; ảnh đen toàn bộ giảm từ 48 xuống 21.'
s['D18']='Đóng băng protocol; tính lại category bằng mapping v2; tính Recall@K, NDCG@K, MRR, Popularity MAE (và Category Match nếu category qua kiểm tra) cho B0, B2p, B2-v1, B2-v2 trên validation và test; khoảng tin cậy; chọn baseline chính bằng validation.'
s['E18']='Hoàn tất đánh giá tuần 5: baseline chính B2-v1 theo luật validation, đã được người dùng chấp nhận dù bộ lọc KHÔNG đúng ngữ nghĩa (giữ 4,9; bỏ 13,15). Category 88/100 không đạt nên bỏ Category Match, w_c=0. Test chính thức đã chạy một lần, đối chiếu độc lập toàn bộ query và CI khớp; có bootstrap, McNemar, mức nền, phân nhóm và montage validation. Luật đặt sau khi thấy sanity test tuần 4 nên test chỉ là xác nhận, không độc lập.'
s['D19']='Cấu hình P (visual + popularity; category chỉ nếu qua kiểm tra, hiện KHÔNG) trên baseline chính B2-v1 theo công thức §3.5; lưới trọng số trên validation, đường đánh đổi giữa độ đúng và Popularity MAE; test một lần.'
s['D21']='So sánh có mask và không mask với cách crop đối chứng (B1a, B1b, B1c, B2p, B2-v1, B2-v2) và một đối chứng chỉ vùng nhãn 4, 9 (mặt, tóc); quyết định pipeline cho demo.'
assert all(cell.style_id==styles[cell.coordinate] for row in s for cell in row)
path=O/'ke_hoach_do_an_cap_nhat_day_du.xlsx';assert not path.exists();w.save(path)
w2=openpyxl.load_workbook(path);assert all(w2[s.title][f'F{row}'].value=='100%' for row in range(14,19));assert all(w2[s.title][coord].style_id==style for coord,style in styles.items())
dump(O/'excel_check_day_du.json',{'source':'ke_hoach_do_an_cap_nhat.xlsx','all_cell_styles_preserved':True,'updated_cells':['F14','F15','F16','F17','F18','E16','D17','E17','D18','E18','D19','D21'],'weeks_1_to_5_percent':'100%','historical_wording':'user-supplied; API/PostgreSQL not retested during this task'})
res=json.loads((O/'independent_result_day_du.json').read_text());cat=read(O/'category_check_100.csv');errors=read(O/'category_error_groups_day_du.csv')
counts=Counter(tag for r in errors for tag in r['groups'].split('; '));correct=sum(int(r['correct']) for r in cat);assert correct==88 and len(cat)==100 and len(errors)==12
md='''# Kết quả tuần 05 — bản đầy đủ

## 1. Đã làm gì và quyết định tiếp tục
Hoàn tất phần còn lại của tuần 5: đóng băng trước test, tính metric test một lần, đối chiếu độc lập toàn bộ query và CI, so sánh ghép cặp với CenterCrop, mức ngẫu nhiên và mức nền popularity, mô tả phân nhóm, montage validation, hình và hồ sơ. Không triển khai fusion, trọng số, đối chứng ảnh gốc hay demo.

**Baseline chính: B2-v1 theo luật validation đã đóng băng và được người dùng chấp nhận trong user_decision.json.** B2-v1 dùng bộ lọc nhãn 1–12, giữ nhãn 4 và 9 (mặt, tóc) và bỏ 13,15 (quần, giày) mà v2 giữ, nên **KHÔNG đúng ngữ nghĩa**. Kết quả chỉ là retrieval proxy trên validation, chưa có giải thích nguyên nhân. B2p và B2-v2 luôn là cấu hình đối chiếu; B0 là tham chiếu popularity giả lập.

**Lưu ý trung thực:** luật chọn được đặt sau khi đã thấy sanity tuần 4 trên test, trong đó v1 hơn v2 về Recall@5. Do đó test ở đây chỉ là xác nhận, **không phải bằng chứng độc lập**. Thứ tự user_decision → freeze → tính metric test chứng minh đợt chạy tiếp không dùng metric test mới để thay quyết định; không xóa được việc đã thấy test trước đó.

## 2. Protocol, cấu hình và kiểm chứng
Không sửa protocol_v1.md hoặc category_rule_v2.json. SHA-256 được đối chiếu trong freeze.json, manifest_day_du.json. Split, ground truth cùng item_id khác ảnh, gallery theo thứ tự CSV, Top-10 stable và loại chính ảnh hỏi giữ nguyên. Recall macro cấp ảnh; Top-1 là Hit@1, không đồng nhất Recall@1. NDCG gain nhị phân/log2; MRR giới hạn Top-10. CI 95% percentile, bootstrap 10.000 lần seed 42 theo query. Môi trường Python 3.11.9, torch 2.14.1, torchvision 0.29.1, NumPy 2.2.1, Pillow 12.2.0; CPU, 4 luồng tính số, không trích vector mới trong đợt tiếp tục (batch 8 giữ ở lượt trích trước).

NormPop từ tuần 4 và t_q theo hash cố định của query. Popularity MAE: lỗi tuyệt đối trên các sản phẩm duy nhất xuất hiện trong Top-10 rồi trung bình query; xếp hạng vẫn ở cấp ảnh. B0 giảm dần NormPop và SHA-256 khi hòa, không tối ưu độ gần t_q. Vì vậy B0 không mặc nhiên có MAE nhỏ.

'''
md+=f"Kiểm tra độc lập: {res['checks']} phép đối chiếu, trong đó {res['test_checks']} thuộc test, {res['mismatches']} không khớp. Mã độc lập không import module metric chính; tái tính query, aggregate, CI, bảng ghép cặp và mức nền. Tái tính kiểm chứng không phải một lần chọn/chạy test chính thức mới.\n"
md+='''
## 3. Kết quả validation và lựa chọn baseline
Bảng chính dùng tỉ lệ 0–1; mỗi ô là ước lượng [CI thấp; CI cao]. Popularity MAE cũng ở thang 0–1.
'''+table('valid',1)
md+='\nBảng quy đổi sang % (MAE là lỗi tuyệt đối nhân 100, không phải accuracy); CI cũng nhân 100.\n'+table('valid',100)
d=paired['valid']['B2-v2 minus B2-v1']['Recall@5']
md+=f"\nCI hiệu Recall@5 v2−v1 trên validation [{d['ci_low']:.6f}; {d['ci_high']:.6f}] không chứa 0, hiệu {d['difference']:.6f}. Theo luật, chọn B2-v1 có Recall@5 cao hơn. Người dùng đã chấp nhận; không chọn lại dựa trên test và không coi mapping v1 đúng.\n"
md+='''
## 4. Kết quả test xác nhận, ghép cặp và các mức nền
Bảng dùng tỉ lệ 0–1 và CI 95%. Test chính thức chạy một lần sau freeze, từ CSV Top-10 cũ đã xác nhận hash và số dòng.
'''+table('test',1)
md+='\nQuy đổi sang % như bảng validation (MAE nhân 100 chỉ là đổi thang).\n'+table('test',100)
md+='\n'+WARN+' Các so sánh dưới đây là mô tả, không đổi baseline.\n\n| Tập | Hiệu cấu hình | Metric | Hiệu (tỉ lệ) | CI 95% |\n|---|---|---|---:|---|\n'
for split in ('valid','test'):
 for label,d in paired[split].items():
  if label=='semantic_note':continue
  for m in ('Recall@5','NDCG@5','MRR@10'):
   vals=[audited(split,label,'paired',m+' '+k,d[m][k],'paired_day_du.json') for k in ('difference','ci_low','ci_high')];md+=f'| {split} | {label} | {m} | {vals[0]:.6f} | [{vals[1]:.6f}; {vals[2]:.6f}] |\n'
md+='\n'+WARN+' McNemar exact hai phía; bảng hàng = cấu hình bị trừ, cột = cấu hình đứng trước, thứ tự sai/đúng. Không điều chỉnh đa kiểm định.\n\n| Tập | Hiệu cấu hình | Bảng 2×2 | p exact |\n|---|---|---|---:|\n'
for split in ('valid','test'):
 for label,d in paired[split].items():
  if label=='semantic_note':continue
  p=audited(split,label,'paired','McNemar p',d['McNemar']['p_exact_two_sided'],'paired_day_du.json');t=d['McNemar']['table_rows_subtrahend_cols_minuend_0_1']
  for i in (0,1):
   for j in (0,1):audited(split,label,'paired',f'McNemar cell {i},{j}',t[i][j],'paired_day_du.json')
  md+=f'| {split} | {label} | {t} | {p:.6f} |\n'
md+='''
Trên test, CI hiệu Recall@5 v2−v1 hoàn toàn âm; CI NDCG@5 và MRR@10 chứa 0 nên không kết luận cấu hình nào tốt hơn ở hai metric đó. McNemar cũng không cho bằng chứng khác biệt Top-1. So với B2p, các CI hiệu trên test đều chứa 0. Trên validation, v1−B2p có CI dương ở các metric ghép cặp, còn v2−B2p có CI chứa 0. Không suy kết quả test độc lập hoặc suy nguyên nhân từ các phép so sánh này.

Mức ngẫu nhiên: E[Top-1] = tổng Rq/(Q·G), E[Recall@5] = 5/G. Mức nền MAE lấy trung bình trên **mọi sản phẩm gallery khác sản phẩm của query**, rồi macro theo query; không trọng số theo số ảnh. Đây là đường tham chiếu, không phải một cấu hình xếp hạng được đánh giá. B0 so với mức ngẫu nhiên/mức nền chỉ mô tả, **chưa kiểm định**.

'''+WARN+' B2p/v2 giữ trong các bảng metric phía trên; bảng sau chỉ đối chiếu riêng B0 theo yêu cầu.\n\n| Tập | Q | G | Cặp đúng | E[Top-1] | B0 Top-1 | E[Recall@5] | B0 Recall@5 | MAE nền | B0 MAE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for r in refs:
 for m in ('random_Top-1','random_Recall@5','popularity_mae_reference'):audited(r['split'],'reference','reference',m,r[m],'reference_levels_day_du.csv')
 md+='| '+r['split']+' | '+' | '.join(r[k] for k in ('queries','gallery_images','ground_truth_pairs'))+' | '+' | '.join(f'{float(r[k]):.6f}' for k in ('random_Top-1','B0_Top-1','random_Recall@5','B0_Recall@5','popularity_mae_reference','B0_Popularity_MAE'))+' |\n'
md+='''
## 5. Mô tả v1/v2, hình và category
Các nhóm dùng cùng danh sách query cho hai cấu hình. Trục nhóm chồng lấp, không cộng số query giữa các trục. Bbox nhỏ/lớn lấy đúng phần đầu/cuối của bbox_extremes.csv đã lưu; “không trong danh sách” không có nghĩa bbox bình thường. Không đặt lại ngưỡng sau khi xem metric. Query đen và gallery đen được xác định riêng theo audit bbox v1/v2.

'''+WARN+' Bảng nhóm chỉ so v1/v2 theo yêu cầu; B2p là đối chiếu ở bảng tổng thể.\n\n| Tập | Trục / nhóm | n | Top-1 v1 | Top-1 v2 | Recall@5 v1 | Recall@5 v2 | Lưu ý |\n|---|---|---:|---:|---:|---:|---:|---|\n'
for i in range(0,len(groups),2):
 a,b=groups[i:i+2];assert a['config']=='B2-v1' and b['config']=='B2-v2';n=int(a['n']);values=[]
 for m in ('Top-1','Recall@5'):
  for r in (a,b):
   if not n:values.append('—');continue
   ck=next(v for v in gchecks if v['split']==r['split'] and v['config']==r['config'] and v['axis']==r['axis'] and v['group']==r['group'] and v['metric']==m)
   ledger.append({'source':'group_metrics_day_du.csv','location':f"{r['split']}/{r['axis']}/{r['group']}/{r['config']}/{m}",'reported_numeric':r[m],'independent_numeric':ck['independent'],'match':ck['match'],'method':'independent scores aggregated on explicit group membership'})
   values.append(f"{float(r[m]):.6f}")
 md+=f"| {a['split']} | {a['axis']} / {a['group']} | {n} | "+' | '.join(values)+f" | {'n nhỏ, không diễn giải' if n<20 else 'Chỉ mô tả, chưa kiểm định nhóm'} |\n"
md+='''
Mẫu hình đáng chú ý về mặt mô tả: ở nhóm một đáp án, Recall@5 v2 thấp hơn v1 trên cả validation và test. Ở nhóm nhiều đáp án trên test, Top-1 v2 cao hơn trong khi Recall@5 gần nhau; không gọi đây là khác biệt có ý nghĩa thống kê vì chưa kiểm định nhóm. Các nhóm ảnh đen/bbox cực trị quá ít query để diễn giải. Độ lớn chênh lệch và n nằm trong bảng; không chọn các nhóm này để thay baseline.

Giả thuyết chưa kiểm tra: hai mask giữ các vùng khác nhau, cụ thể v1 giữ mặt/tóc còn v2 giữ quần/giày; điều này có thể liên quan biểu diễn. Chưa có đối chứng để xác định ảnh hưởng hay quan hệ nhân quả. Không chạy đối chứng chỉ vùng mặt/tóc hoặc ảnh không mask ở tuần này.

Hình 1: Recall@5/NDCG@5 và CI validation/test. Hình 2: chuyển trạng thái Top-1. Hình 3 category cũ giữ nguyên. Hình 4: montage validation lấy query_id tăng dần trong mỗi chiều chuyển trạng thái, không chọn tay, không ngẫu nhiên. Hình 5: MAE cùng đường mức nền. Nguồn/caption và quy tắc chọn nằm ở figures/mo_ta_hinh_day_du.md.

'''
md+=f"Category: **{correct}/{len(cat)} KHÔNG ĐẠT**, bỏ Category Match; **w_c = 0** là trạng thái loại thành phần category, chưa triển khai/chỉnh trọng số mô hình fusion. Có {len(errors)} sản phẩm đánh dấu sai hoặc chưa xác nhận được. Đây là đánh giá agent, vẫn cần người xem lại montage.\n\n"
md+='Phân loại ca sai từ ghi chú đã có (không chấm lại): '+', '.join(f'{n} ca {tag}' for tag,n in counts.items())+'. Ca item 3304 nằm trong nhóm váy/váy liền/playsuit và nhóm váy liền/áo dài phối quần; đây là **phân nhóm chồng lấp**, không cộng thành số sản phẩm. Tổng sản phẩm duy nhất vẫn là 12. Xem category_error_groups_day_du.csv.\n'
md+='''
## 6. Giới hạn và các số cần đính chính
Popularity và t đều giả lập; MAE thấp chỉ là gần t giả lập hơn, không phải lượt mua hay sở thích thật. Ground truth cùng sản phẩm khác ảnh đánh giá nhận diện/truy hồi sản phẩm. Mapping v2 do thực nghiệm tự xác định, chưa có tài liệu nhãn chính thức. Category một nhãn không phù hợp nhiều ảnh cả bộ đồ; việc nhìn một ảnh đại diện có thể không khớp majority sản phẩm, nên ca chưa xác nhận được đã tính sai bảo thủ. Giữ nguyên kết quả kiểm tra cũ, không dùng metric category.

Các số Category Match 55,72% / 80,50% / 79,75% trong báo cáo tuần 1 và Excel E15; 78% / 54% trong báo cáo tuần 3 và doi_chieu_cau_hinh_P.json cần đính chính vì dựa trên mapping cũ sai. Đường dẫn đầy đủ: can_dinh_chinh.md. Không sửa file nguồn, không thay bằng category metric mới.

README.md của data_protocol khác hash lịch sử trong validation_report.json nhưng không đổi giữa đầu/cuối lượt chạy trước hoặc lượt này. CSV split/inventory/ground truth vẫn khớp hash lịch sử. Không dùng README làm dữ liệu tính metric.

Chưa kiểm tra được: peak RAM toàn đợt; nguyên nhân chênh lệch v1/v2; ý nghĩa với người mua; độ đúng category do con người; khả năng tổng quát nhiều split/seed; định nghĩa trong paper trực tuyến. Nội dung API/PostgreSQL lịch sử trong Excel chép đúng yêu cầu người dùng, chưa kiểm thử API lại ở lượt này. Không kết luận khác biệt khi CI chứa 0; các nhóm chỉ mô tả, nhiều phép so sánh chưa hiệu chỉnh đa kiểm định.

Khác/diễn giải prompt: (i) phân loại ca category có một trường hợp giao nhau để giữ đúng tổng sản phẩm đã kiểm tra, không bịa ca thứ 13; (ii) “không đóng gói/demo” được hiểu là không xây gói phần mềm hoặc demo, vẫn tạo ZIP hồ sơ theo mục E.4; (iii) tên freeze.json, user_decision.json, t_q_test.csv, independent_test_checks.csv theo tên chỉ định, các đầu ra còn lại hậu tố _day_du; (iv) các bảng phân nhóm chỉ v1/v2 và bảng mức nền riêng B0 theo yêu cầu cụ thể, mọi bảng tổng thể đều giữ B2p/v2; (v) không tự tạo một lần test mới khi kiểm chứng, chỉ tính độc lập từ cùng CSV đóng băng. Protocol, luật chọn và mọi artifact cũ không bị sửa.

## 7. Minh chứng, toàn vẹn và tái chạy
Các file mới gồm freeze.json, user_decision.json, metrics_test_day_du.csv, paired_day_du.json, reference_levels_day_du.csv, independent_test_checks.csv, group_metrics_day_du.csv, montage_selection_day_du.csv, ảnh _day_du, bản sao Excel và manifest_day_du.json. Các báo cáo “đã dừng” cũ là lịch sử của lần bàn giao trước; bản đầy đủ này là kết quả chạy tiếp.

verification_day_du.txt ghi thứ tự thời gian user_decision → freeze → test, kèm timestamp nội dung và timestamp filesystem. hash_cu_truoc_day_du.json / hash_cu_sau_day_du.json chứng minh nguồn tuần 1–4 không đổi; hash_week5_truoc_day_du.json / hash_week5_sau_day_du.json bảo vệ toàn bộ artifact cũ tuần 5. doi_chieu_so_lieu_day_du.csv đối chiếu số báo cáo với triển khai độc lập; các giá trị lịch sử cần đính chính được đánh dấu nguồn lịch sử, không trình bày như metric mới được tính lại.

commands_day_du.txt ghi lệnh và thứ tự đã chạy. Các script dùng tạo file độc quyền để tránh ghi đè; không chạy lại vào thư mục giao nộp. Khi cần tái lập, dùng một thư mục sibling mới với bản sao đầu vào cần thiết. Không trích lại vector hoặc đổi ranking trong lượt này. Manifest liệt kê input/output, SHA-256, số dòng, môi trường, thời gian; RAM ghi chưa kiểm tra được. ZIP hồ sơ chứa cả file cũ và mới, kể cả ZIP cũ; không thể chứa chính ZIP đang được tạo.

## 8. Dự kiến tuần 6
Chỉ cập nhật kế hoạch: cấu hình P visual + popularity trên baseline chính B2-v1 đã được chấp nhận, category hiện bị loại; lựa chọn trọng số trên validation, xem đánh đổi retrieval và MAE giả lập, test một lần theo protocol riêng phù hợp. Chưa triển khai các bước đó. Kế hoạch tuần 8 thêm đối chứng vùng mặt/tóc và crop/mask theo văn bản người dùng; chưa chạy ở tuần này.
'''
# Additional numeric facts/counts appearing outside tables.
def fact(key,value,independent,method):ledger.append({'source':'artifact facts','location':key,'reported_numeric':value,'independent_numeric':independent,'match':value==independent,'method':method})
fact('independent checks',res['checks'],len(checks),'count independent_test_checks.csv');fact('test checks',res['test_checks'],sum(r['split']=='test' for r in checks),'count split=test');fact('mismatches',res['mismatches'],sum(r['match']!='True' for r in checks),'count mismatch flags');fact('category correct',correct,sum(int(r['correct']) for r in cat),'sum original review CSV');fact('category total',100,len(cat),'count original review CSV');fact('category failed',12,len(errors),'unique reviewed failed products');fact('category disabled w_c',0,0,'protocol excludes failed category; no weight optimization')
for tag,n in counts.items():fact('category group '+tag,n,sum(tag in r['groups'].split('; ') for r in errors),'count group flags; overlapping case documented')
for r in refs:
 rr=read(R/f"results/week2/FashionRecommender/data_protocol/{r['split']}.csv");gt=read(R/f"results/week2/FashionRecommender/data_protocol/ground_truth_{r['split']}.csv")
 for key,val in [('queries',len({v['query_image_path'] for v in gt})),('gallery_images',sum(v['role']=='gallery' for v in rr)),('ground_truth_pairs',len(gt))]:fact(r['split']+' '+key,int(r[key]),val,'independent CSV count')
for i in range(0,len(groups),2):
 r=groups[i];mm=read(O/'group_membership_day_du.csv');fact(f"group n {r['split']}/{r['axis']}/{r['group']}",int(r['n']),sum(v['split']==r['split'] and v[r['axis']]==r['group'] for v in mm),'count membership CSV')
for value in ('55,72%','80,50%','79,75%','78%','54%'):
 assert value in (O/'can_dinh_chinh.md').read_text();ledger.append({'source':'can_dinh_chinh.md','location':'historical invalid category value','reported_numeric':value,'independent_numeric':'không áp dụng: số lịch sử cần đính chính, chưa tái chạy category cũ','match':'source_verified','method':'literal verified in prior correction document; not a valid new metric'})
write(O/'doi_chieu_so_lieu_day_du.csv',ledger)
with (O/'ket_qua_tuan_05_day_du.md').open('x') as f:f.write(md)
print('Report, numeric audit, and styled Excel copy created')
