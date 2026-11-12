from pathlib import Path
import hashlib,json,datetime
O=Path(__file__).resolve().parent; R=O.parents[2]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 old=[p for w in range(1,5) for p in (R/'results'/f'week{w}').rglob('*') if p.is_file()]+list((R/'FashionRecommender').glob('*.py'))
 dump(O/'hash_cu_truoc.json',{str(p.relative_to(R)):sha(p) for p in sorted(old)})
 p=O/'protocol_v1.md'
 p.write_text('''# Protocol tuần 5 — v1
Đóng băng trước mọi phép tính metric tuần 5. Không sửa; đính chính ở protocol_v1_changes.md có thời điểm và lý do.

R1. Giữ nguyên CSV split và ground truth tuần 2. Cấp ảnh, cùng item khác ảnh; không đổi split, backbone hay ground truth.
R2. Recall@K (1,5,10) = số đáp án trong Top-K / tổng đáp án, trung bình đều ảnh hỏi. Top-1 = Hit@1. Đối chiếu Recall và Top-1 với week3_retrieval_metrics.py (reciprocal_rank tại K=1). NDCG@5,10 dùng gain nhị phân, discount log2(rank+1), IDCG cho min(K,Rq). MRR@10 là nghịch đảo rank đúng đầu tiên hoặc 0. Không có đáp án: module mới trả 0 cho unit test; dữ liệu chính thức phải có đáp án.
R3. Chọn chỉ bằng validation; test chính thức một lần sau freeze.json. Sanity tuần 4 trước đó chỉ tái tạo Top-1/Recall@5; không dùng từng query test để chọn cấu hình hay ví dụ.
R4. NormPop từ popularity.csv tuần 4 (giả lập seed 42, min–max [0,1], chung sản phẩm). t_q = int.from_bytes(sha256(f"42:{query_id}".encode()).digest()[:8], "big") / 2**64. Lưu CSV. Popularity MAE: trung bình |NormPop_i-t_q| trên sản phẩm khác nhau có trong Top-10 từng query, rồi macro trên query. Lặp sản phẩm chỉ tính một lần vì định nghĩa ở cấp sản phẩm. Không dùng popularity làm điểm visual.
R5. Category theo quy tắc v2 đóng băng riêng. Chỉ tính Category Match@5 khi agent xem đủ 100 sản phẩm và đúng ít nhất 90; cần người xem lại. Bỏ query unknown và candidate unknown; mẫu số là candidate hợp lệ trong Top-5, macro trên query còn ít nhất một candidate; báo số bỏ. Similarity 1 cùng category, 0 khác category, không dùng nhóm semantic paper.
R6. Chỉ kết luận proxy: popularity và t giả lập, cùng sản phẩm không phải sở thích hay lượt mua; mapping thực nghiệm chưa có tài liệu chính thức.
R7. Seed 42, CPU, 4 luồng, batch 8, deterministic. Môi trường mục tiêu Python 3.11.9, torch 2.14.1, torchvision 0.29.1, NumPy 2.2.1, Pillow 12.2.0; ghi thực tế.

Cấu hình: B0 popularity giảm dần, hòa SHA-256("42:{candidate_image_path}"); B2p nhãn 1–12 CenterCrop đúng crops() cũ; B2-v1 nhãn 1–12 bbox fixed_all.npy; B2-v2 nhãn [1,2,3,5,6,7,8,10,11,12,13,15] bbox fixed_all_v2.npy. Visual cosine, argsort stable theo gallery CSV, bỏ chính query, không gộp sản phẩm khi xếp hạng, Top-10. Test dùng CSV cũ, 5860 dòng/cấu hình.

Bootstrap 10000 lần seed 42, percentile 2.5/97.5, theo query, cho Top-1, Recall@5, NDCG@5, MRR@10; dùng cùng chỉ số resample cho paired v2-v1. McNemar exact hai phía bằng nhị thức trên discordant Top-1.
Luật chọn: chỉ B2-v1/v2; nếu CI hiệu Recall@5 v2-v1 không chứa 0 chọn Recall@5 cao hơn; ngược lại chọn B2-v2 vì đúng ngữ nghĩa hơn. Nếu chọn B2-v1 dừng và báo người dùng, chưa dùng tuần 6. B0/B2p chỉ tham chiếu.
Ngẫu nhiên: E[Top-1]=sum(Rq)/(Q*G), E[Recall@5]=5/G; chỉ so mô tả, chưa kiểm định B0 so ngẫu nhiên.
Dừng nếu sanity không khớp, hash cũ đổi hoặc thiếu validation. Category không đạt thì tiếp tục bỏ metric category. Không thực hiện việc tuần 6 trở đi.
Prompt liệt kê 9 nội dung nhưng gọi 8 mục báo cáo: gộp minh chứng/tái chạy và dự kiến tuần 6 vào mục cuối. Không thấy file kế hoạch tuần 5–15 khi kiểm tra repository.
''')
 dump(O/'manifest.json',{'status':'protocol_frozen','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(p)})
 print('Frozen',sha(p),'old files',len(old))
