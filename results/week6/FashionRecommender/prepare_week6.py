import json,hashlib,csv,shutil
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=O.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,s):
 with (O/n).open('x') as f:f.write(s)
t=datetime.now(timezone.utc).isoformat()
write('user_decision_week6.json',json.dumps({'recorded_at_utc':t,'timestamp_kind':'time decision recorded, not message server timestamp','rule':'(wv,wp,wc,gamma) -> (wv+wc,wp,0,gamma); preserve wp/gamma; historical configurations not evaluated; deduplicate grid aliases','source':'User continuation explicitly accepts and generalizes category removal rule','baseline':'A=B2-v1'},ensure_ascii=False,indent=2))
shutil.copyfile(O/'tong_hop_dau_vao_tuan06.md',O/'tong_hop_dau_vao_tuan06_truoc_quyet_dinh.md')
p=O/'tong_hop_dau_vao_tuan06.md'
s=p.read_text().replace('Trạng thái: DỪNG ở bước 1; chưa chạy lưới validation, oracle hoặc test tuần 6.','Trạng thái: mâu thuẫn đã được người dùng chốt; tiếp tục bước 2. Bản dừng lịch sử lưu riêng.')
s+='\n## Quyết định tiếp tục\n\nGhi nhận UTC '+t+' (thời điểm ghi nhận, không phải timestamp máy chủ của tin nhắn). Người dùng chốt chuyển wc sang wv, giữ wp và gamma cho mọi cấu hình có tên. Đặt hậu tố “đã bỏ category”, không đánh giá cấu hình gốc, trùng lưới chỉ tính một lần. Không dùng chuẩn hóa tỉ lệ. Mâu thuẫn bước 1 đã giải quyết. Xem user_decision_week6.json và protocol_week6.md.\n\nĐã đối chiếu arXiv HTML v1: §3.5 công thức; §4.2 wp=.3/gamma2; §4.5 wp=.2 nhưng không ghi wc hay gamma riêng. Không bịa wc=.1 là con số paper công bố; điểm §4.5 được hoàn thiện theo quy ước tuần6 wv=1-wp, gamma mặc định §4.2. Xem paper_source_week6.md.\n'
p.write_text(s)
write('paper_source_week6.md','''# Đối chiếu paper
Nguồn: https://arxiv.org/html/2506.07773v1 (đọc trước protocol).
§3.5, dòng HTML 135: thứ tự thành phần visual, popularity, category.
§4.2, dòng 168: wv=.7, wp=.3, gamma=2; không nêu wc số.
§4.5, dòng 205: wv=.7, wp=.2; wc và gamma riêng không nêu.
§4.3, dòng 184: MAE giữa popularity chuẩn hóa và trendiness; không quy định gộp ảnh nhiều góc như protocol_v1.
Quy ước thực thi: P giữ cách đối chiếu repo (.7,.3,0), gamma2. §4.5 sau bỏ category dùng (.8,.2,0), gamma2 mặc định §4.2, theo tổng trọng số tuần6; không khẳng định paper ghi wc=.1. Chỉ là điểm tham chiếu triển khai, không tái lập đầy đủ paper.
''')
write('protocol_week6.md','''# Protocol bổ sung tuần 6 — bất biến
Được ghi trước mọi ranking/metric tuần6. SHA-256 lưu protocol_week6.sha256. Quyết định người dùng trong user_decision_week6.json. CPU, 4 threads, seed42, deterministic; tái dùng vector L2 float32 cũ, không trích mới. Ghi phiên bản thực tế. Validation G=1597,Q=568; test G=1543,Q=586.

## Công thức và thứ tự
Relevance(q,i)=(1-wp)*cos(q,i)+wp*(1-|NormPop_i-t_q|^gamma); wc=0. Cos dot vector L2 giữ nguyên tuần4, KHÔNG min-max/rescale. Tính cos float32 đúng phép nhân ma trận cũ; fusion float64. wp=0 lấy nguyên cos để tái tạo score cũ. Stable argsort giảm dần theo thứ tự gallery CSV; loại chính query, không loại sản phẩm đáp án. Top10 schema query_id,rank,candidate_id,product_id,score,is_correct. Ground truth chỉ dùng chấm, không dùng score chính thức. NormPop sản phẩm từ artifacts_completed/popularity.csv theo popularity_lineage.json. t lấy nguyên t_q_valid.csv/t_q_test.csv tuần5 (hash 42:query_id). A=B2-v1 chính; B=B2-v2 đối chiếu; G2 ảnh gốc trong bbox C cần mask, chỉ tham chiếu. Không category vì 88/100 không đạt.

## Lưới và cấu hình có tên
wp={0,.1,.2,.3,.4,.5}, gamma={2,4}, từng A,B,G2. Hai gamma tại wp=0 là cùng score/ranking: tính một lần gamma2, ghi alias gamma4. Không chạy lại điểm có tên trùng lưới.
Quy tắc chung do người dùng chốt: chuyển toàn bộ wc sang wv, giữ wp và gamma. Tên “<tên gốc>, đã bỏ category”; chỉ lưu trọng số gốc trong bảng lịch sử, không đánh giá category. Giữ wp vì đang khảo sát tác động wp; chuẩn hóa tỉ lệ .6:.3 thành 2/3:1/3 đã cân nhắc nhưng KHÔNG dùng (wp=.333... ngoài lưới).
Thứ tự dưới đây LUÔN (wv,wp,wc), khác thứ tự cột cũ (visual,category,popularity).

| Tên | Gốc/lịch sử, không đánh giá | Sau bỏ category | gamma | Nguồn và xử lý |
|---|---|---|---|---|
| Tuần 3, đã bỏ category | (.6,.3,.1) | (.7,.3,0) | 4 | compare_week3_p.py:72,114; doi_chieu_cau_hinh_P.json:6–11; trùng điểm lưới |
| P của paper, đã bỏ category | (.7,.3,wc không nêu); repo dùng wc=0 | (.7,.3,0) | 2 | paper §4.2 HTML:168; compare_week3_p.py:73–75,115; trùng điểm lưới |
| §4.5 wp=.2, đã bỏ category | (.7,.2,wc không nêu) | (.8,.2,0) | 2 mặc định §4.2 | paper HTML:205; hoàn thiện theo tổng trọng số tuần6, không khẳng định wc=.1 được paper công bố; trùng điểm lưới |

Với phần trọng số chưa được paper công bố, không suy diễn số lịch sử; sau bỏ category dùng ràng buộc wv=1-wp đã được yêu cầu. Gamma §4.5 kế thừa mặc định §4.2, không tối ưu theo kết quả. Nếu named P có wp=0 thì ghi trùng visual-only và không coi là P thật; P hiện tại wp=.3 nên không rơi trường hợp này.

## Metric và thống kê
Top1=Hit1. Recall@1,5,10 là số ảnh đúng / toàn bộ ảnh đáp án query. NDCG@5,10 binary gain/log2 rank với IDCG min(K,Rq). MRR10 rank đúng đầu tiên hoặc0. Popularity MAE@5,10: lấy sản phẩm duy nhất TRONG Top-K ảnh (không lấy thêm ảnh để đủ K sản phẩm), trung bình abs(NormPop-t), macro query. CI95% percentile bootstrap 10000 seed42 theo query (thứ tự ground_truth CSV); dùng cùng resample cho hiệu mỗi điểm so wp0 cùng base và cùng t. Báo CI mọi metric; McNemar exact hai phía Top1. Không diễn giải CI marginal như kiểm định hiệu.
Đa kiểm định: 30 tương phản grid khác wp0 (10/base), 20 oracle, tối đa6 test. Họ McNemar chính gồm30 validation + số test khác wp0 thực chạy, Bonferroni alpha=.05/m; oracle tách riêng. Họ CI hiệu 5 metric chính (Recall5,NDCG5,MRR10,MAE5,MAE10) nhân số tương phản chính. Báo thêm percentile Bonferroni từ cùng bootstrap, cảnh báo đuôi Monte Carlo hạn chế. Không dùng hiệu chỉnh để chọn cấu hình.

## Mức tham chiếu và thang điểm
Báo mean abs(NormPop-t) trên mọi ẢNH gallery và trên sản phẩm duy nhất; bản theo sản phẩm là tham chiếu chính cùng đơn vị MAE. Báo thêm nền loại sản phẩm query để đối chiếu tuần5.
“Chặn dưới 10 sản phẩm” chính: mean 10 sản phẩm khác nhau gần t nhất trong gallery, có tính sản phẩm query vì ranking hợp lệ chứa ảnh đáp án. Báo thêm bản loại sản phẩm query theo yêu cầu tham chiếu. Bản loại không là chặn dưới của không gian ứng viên chính. Cả hai chỉ là tối ưu có ràng buộc đủ10 sản phẩm, KHÔNG là chặn dưới tuyệt đối của MAE trên Top10 ảnh (có thể dưới10 sản phẩm). Chặn dưới hợp lệ cho MAE thực là min lỗi trên một sản phẩm khả dụng; báo riêng. Không dùng tham chiếu chọn cấu hình.
Validation từng base: min/median/max cos ở Top1 và toàn bộ Top10 visual wp0; SD số hạng chưa nhân wp (1-|pop-t|^gamma) trên toàn cặp query-gallery, thêm SD thành phần nhân wp cho từng điểm. Không đổi lưới theo thang.

## Luật vận hành và điều kiện dừng
Mỗi base chọn wp lớn nhất có Recall5 >= baseline-0.02; hòa chọn MAE10 thấp hơn; hòa nữa gamma2. Nếu không wp>0 đạt thì “không có điểm vận hành”. Độ nhạy ngưỡng .01/.02/.03 mô tả בלבד, không thay luật .02; ngưỡng chủ quan định trước.
(a) wp0 phải tái tạo từng dòng sáu cột Top10 và metric cũ A/B tuần5, G2 tuần5c (bỏ cột analysis_note). MAE G2 không có trong5c: đối chiếu độc lập, không giả vờ có số cũ.
(b) Kỳ vọng Recall giảm gần đơn điệu do t/pop độc lập ground truth. “Tăng rõ rệt” định trước là CI95% hiệu Recall5 dương hoàn toàn so wp0 hoặc bước wp liền kề cùng gamma. Nếu xảy ra DỪNG tìm rò rỉ, không freeze/test; các dao động nhỏ báo đầy đủ, không ép đơn điệu. Không khẳng định độc lập bảo đảm giảm trong mọi mẫu hữu hạn. Kiểm tra tương quan t và pop sản phẩm query, lineage, tie, số sản phẩm trùng NormPop.
(c) Code độc lập không import module chính: toàn bộ query cho wp0/gamma2, wp.3/gamma2 và wp.5/gamma4 mỗi base; kiểm cả ranking, metric, CI. Test kiểm độc lập mọi cấu hình A và tái tạo wp0 mọi base. Sai khác -> dừng.
Hash cũ đổi, freeze/test sai thứ tự hoặc kiểm chứng không khớp -> dừng; sửa lỗi code phải ghi phiên bản và kiểm lại, không âm thầm đổi protocol.

## Oracle chỉ validation
Chỉ A. Với sigma=.1,.3: seed riêng int SHA256('42:'+query_id+':'+format(sigma,'.1f')) tám byte đầu big endian; default_rng(seed).normal(0,sigma); clip(pop sản phẩm đáp án+noise,0,1). Ground truth cùng item nên pop đáp án=pop item query; kiểm item khớp. Đây là ORACLE có chủ đích, giả định trendiness khớp món cần tìm, không bằng chứng người thật. Cùng lưới; so wp0 cùng t bằng CI ghép cặp Recall5 và MAE5/10. Lưu riêng oracle/, không dùng để chọn/freeze, không test oracle. Sigma=∞ trên hình là nhãn t chính thức độc lập, không phải lấy giới hạn phép clip Gaussian. CI chứa0 nghĩa chưa phát hiện giảm, không chứng minh không giảm.

## Freeze và test
Sau validation, kiểm (a)-(c), oracle mô tả: freeze_week6.json ghi wp0, P paper và điểm vận hành nếu có cho từng base; loại trùng; hash protocol, input nguồn, mã, Top10 validation, timestamp. Test một lần tối đa9 cấu hình, marker tạo độc quyền trước chấm; từ chối chạy lại. Kiểm chứng từ kết quả đóng băng không phải lần chọn/test mới. Test đã thấy từ các tuần trước, chỉ xác nhận có điều kiện.

## Kỳ vọng trước khi chạy
Tăng wp thường kéo MAE xuống, Recall giảm; gamma4 giảm phạt khoảng cách trung bình nên có thể tác động nhẹ hơn gamma2. Thang cos khác base có thể tạo độ dốc khác; không suy ưu thế mask hoặc trọng số tối ưu chung. Oracle có thể làm đường giảm phẳng hơn; không hứa Recall tăng. Kết luận chỉ proxy, B0/ngẫu nhiên chỉ mô tả. Không thêm seed, ablation, B1, chấm tay, demo, tách nền/da hoặc đổi baseline.
'''.replace(' בלבד',''))
write('protocol_week6.sha256',sha(O/'protocol_week6.md')+'  protocol_week6.md\n')
print(t,sha(O/'protocol_week6.md'))
