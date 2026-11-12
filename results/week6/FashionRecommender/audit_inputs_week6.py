"""Read-only source audit. Writes exclusively new Week 6 artifacts; no evaluation."""
import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(data)
def js(name,obj):write(name,json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def snapshot():
    paths=sorted(p for w in range(1,6) for p in (ROOT/f'results/week{w}').rglob('*') if p.is_file())
    paths+=sorted((ROOT/'FashionRecommender').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
start=datetime.now(timezone.utc).isoformat()
before=snapshot(); js('hash_cu_truoc_tuan06.json',before)
notes={
'bao_cao_tuan_03.md':'Bảng trọng số: thử tuần 3 (visual,popularity,category)=(0.6,0.3,0.1), gamma=4; P=(0.7,0.3,0), gamma=2. Ghi chú §4.5 wp=0.2.',
'protocol_v1.md':'R4: MAE macro trên sản phẩm duy nhất trong Top-10; t_q hash seed42. R7 CPU deterministic. Cos không rescale; stable gallery.',
'baseline_chinh.json':'Chọn B2-v1 theo validation; cần đọc cùng user_decision.json.',
'freeze.json':'Freeze tuần 5 tồn tại; không phải freeze tuần 6.',
'user_decision.json':'Đã chấp nhận B2-v1 sai ngữ nghĩa để tiếp tục tuần 5; yêu cầu hiện tại cho phép dùng A tuần 6.',
'can_dinh_chinh.md':'Category cũ không được dùng làm bằng chứng ngữ nghĩa.',
'ket_qua_tuan_05.md':'Bản lịch sử dừng; được bổ sung bởi ket_qua_tuan_05_day_du.md, không coi là thiếu test tuần 5.',
'ket_qua_tuan_05_day_du.md':'Baseline B2-v1; category 88/100 không đạt, wc=0; popularity/t giả lập; test đã được thấy từ tuần trước.',
'ket_qua_tuan_05b.md':'A=v1, B=v2; không đổi baseline chính; phân tích hậu nghiệm.',
'ket_qua_tuan_05c.md':'G2 ảnh gốc trong bbox C cần mask; chỉ mốc tham chiếu, không phải pipeline không mask.',
'doi_chieu_cau_hinh_P.json':'Dòng 6–17: trọng số có tên khóa; tuần3 visual=.6/category=.1/popularity=.3/gamma4; P visual=.7/category=0/popularity=.3/gamma2.',
'compare_week3_p.py':'Dòng 72: .6*visual+.1*category+.3*(1-distance**4); dòng 75: .7*visual+.3*(1-distance**2). Không có quy tắc chuyển tuần3 về wc=0.',
'direction_a_repro.py':'Mô phỏng user/purchases và gamma4 lịch sử khác t_q hash chính thức. Không tái sinh popularity bằng pipeline này.',
'final_recommandation.py':'Mã gốc dùng min-max visual, mặc định gamma4; không thay thế quy ước cosine tuần4.',
'popularity_lineage.json':'Nguồn giả lập default_rng(42), integer 0..100 theo metadata; min-max; không từ giao dịch.',
}
selected=[]
for p in sorted((ROOT/'results').rglob('*')):
    if not p.is_file() or OUT in p.parents:continue
    if p.name in notes or (p.name.startswith('ket_qua_tuan_04') and p.suffix=='.md') or p.name in ('t_q_valid.csv','t_q_test.csv'):
        selected.append(p)
selected += [ROOT/'FashionRecommender'/n for n in ('compare_week3_p.py','direction_a_repro.py','final_recommandation.py')]
checks={}
for split,expected in [('valid',568),('test',586)]:
    p=ROOT/f'results/week5/FashionRecommender/t_q_{split}.csv'
    rows=list(csv.DictReader(p.open()))
    bad=[r['query_id'] for r in rows if float(r['t_q'])!=int.from_bytes(hashlib.sha256(('42:'+r['query_id']).encode()).digest()[:8],'big')/2**64]
    assert len(rows)==expected and not bad
    checks[split]={'rows':len(rows),'hash_formula_matches':True,'gallery_expected_from_reports':1597 if split=='valid' else 1543}
lines=['# Tổng hợp đầu vào tuần 06','', 'Trạng thái: DỪNG ở bước 1; chưa chạy lưới validation, oracle hoặc test tuần 6.','',
'Phạm vi dự kiến: validation 1.597 gallery / 568 ảnh hỏi; test 1.543 gallery / 586 ảnh hỏi. Các kích thước gallery lấy từ hồ sơ cũ, chưa dựng lại ranking.', '',
'## Mâu thuẫn chưa giải được — điều kiện dừng', '',
'Yêu cầu vừa buộc wc=0 và wv=1-wp, vừa yêu cầu thêm điểm chạy thử tuần 3 với đúng trọng số nguồn. Nguồn xác nhận (wv,wp,wc)=(0.6,0.3,0.1), gamma=4. Không có ánh xạ duy nhất sang hai thành phần trong file nguồn. Giữ wp rồi chuyển phần category sang visual cho (0.7,0.3,0), còn bỏ category rồi chuẩn hóa cho (2/3,1/3,0); hai cách khác nhau và đều không phải cấu hình gốc. Không tự chọn hoặc ghi nhãn sai là tái tạo tuần 3.', '',
'Đề xuất cần làm rõ: điểm mang tên “tuần 3 đã bỏ category” dùng (0.7,0.3,0), gamma=4, đồng thời lưu cấu hình lịch sử trong bảng đối chiếu. Đây chỉ là đề xuất, chưa được chạy hoặc đóng băng.', '',
'## Đối chiếu và khoảng trống đã giải thích', '',
'- A=B2-v1 là baseline chính; B=B2-v2 đối chiếu; G2 chỉ mốc tham chiếu và vẫn cần mask C lấy bbox.',
'- Category 88/100 không đạt: wc=0. Không dùng các Category Match lịch sử đã đính chính.',
'- Báo cáo tuần5 bản dừng và freeze tồn tại không mâu thuẫn thời gian: bản day_du và user_decision ghi lần tiếp tục.',
'- MAE tuần3 khác protocol tuần5 về đơn vị và t. Tuần6 phải dùng macro query, sản phẩm duy nhất trong Top-K; Top-5 là mở rộng cùng định nghĩa.',
'- Cos min-max trong mã gốc khác cosine tuần4. Chỉ giữ cosine tuần4 theo yêu cầu, không rescale.',
'- Popularity chính thức dùng artifacts_completed, không dùng artifacts dở. Không dùng popularity tăng theo purchases của direction_a_repro.',
'- t_q_valid/test đã kiểm lại toàn bộ công thức hash seed42; không tính metric test. Kết quả chi tiết trong input_checks_week6.json.',
'- P nguồn repo quy về (wv,wp,wc)=(0.7,0.3,0), gamma2; §4.5 wp=.2 được ghi trong báo cáo tuần3. Chưa kiểm tra paper trực tuyến, không tuyên bố đã xác minh PDF gốc.',
'- Test đã được quan sát trong các tuần trước; chạy một lần tuần6 không làm nó thành holdout chưa từng thấy.', '',
'## File, SHA-256 và kết luận dựa vào', '', '| File (tương đối gốc dự án) | SHA-256 | Kết luận |','|---|---|---|']
for p in selected:
    p.read_text(encoding='utf-8')
    lines.append(f'| `{p.relative_to(ROOT)}` | `{sha(p)}` | '+notes.get(p.name,'t_q chính thức đã kiểm hash.' if p.suffix=='.csv' else 'Cosine L2, stable gallery; bộ lọc v1/v2 và đính chính mask lịch sử.')+' |')
write('tong_hop_dau_vao_tuan06.md','\n'.join(lines)+'\n')
js('input_checks_week6.json',checks)
after=snapshot();js('hash_cu_sau_tuan06.json',after)
assert before==after,'STOP: old files changed'
js('status_week6.json',{'status':'stopped_input_contradiction','started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'old_files_hashed':len(before),'old_files_unchanged':True,'validation_run':False,'test_run':False,'freeze_written':False})
write('verification_tuan06.txt',f'PASS: {len(before)} old files unchanged (weeks1–5 recursively incl.5b/5c, plus FashionRecommender root Python source).\nPASS: t_q hash formula for all valid/test rows.\nSTOP: historical Week3 configuration incompatible with wc=0 and wv=1-wp.\nNOT RUN: ranking, metric, CI, oracle, freeze, test.\n')
print(json.dumps({'old_files':len(before),'unchanged':before==after,'status':'stopped_input_contradiction'}))
