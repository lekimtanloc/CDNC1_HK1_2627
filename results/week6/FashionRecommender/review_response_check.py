import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent
versions=O/'review_versions';versions.mkdir(exist_ok=True)
old='4. **Không áp dụng cho cáo buộc chọn trên test tuần6 — có bằng chứng thứ tự.**'
new='4. **Đã sửa — giới hạn chứng thực lịch sử test.** Hồ sơ hiện có nhất quán với việc freeze trước lượt test được ghi nhận; không thấy dấu hiệu chọn cấu hình bằng kết quả test tuần6. Timestamp/marker nội bộ không chứng thực độc lập mọi lần thực thi ngoài hồ sơ.'
for name in ['phan_bien_tuan_06.md','ket_qua_tuan_06.md']:
 p=O/name;backup=versions/(p.stem+'_v1.md')
 if not backup.exists():shutil.copyfile(p,backup)
 if name.startswith('phan_bien'):
  text=p.read_text().replace(old,new)
  if '## Xử lý sau phản biện độc lập' not in text:text+='\n## Xử lý sau phản biện độc lập\n\nBản diễn đạt v2 làm mềm mục4; không đổi số liệu v1. Đã kiểm lại timeline/hash/marker sau sửa; xem review_response_checks.json. Phản biện độc lập chỉ đọc tuần6, không xác minh trực tiếp nguồn tuần2/4/5 hay paper; phân biệt với mã kiểm chứng độc lập của tác giả có đọc nguồn.\n'
 else:
  text=p.read_text()
  if 'Timestamp/marker nội bộ không chứng thực' not in text:text+='\nGiới hạn được bổ sung sau phản biện độc lập: Timestamp/marker nội bộ không chứng thực độc lập mọi lần thực thi ngoài hồ sơ; hồ sơ hiện có nhất quán với freeze trước lượt test được ghi nhận. Agent phản biện chỉ đọc tuần6 nên xác nhận tính nhất quán nội bộ, không xác minh trực tiếp nguồn dataset/vector/paper. Mã kiểm chứng độc lập của tác giả có đọc các nguồn cũ là một lớp kiểm tra khác. Chặn dưới lỗi nhỏ nhất một sản phẩm có thể không đạt được khi yêu cầu đủ10 ảnh, không phải giá trị tối ưu khả thi đã tìm. Không thay số liệu hoặc chạy lại test.\n'
 p.write_text(text)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((O/'freeze_week6.json').read_text());marker=json.loads((O/'test_started_week6.json').read_text());done=json.loads((O/'completed_test.json').read_text());decision=json.loads((O/'user_decision_week6.json').read_text())
assert marker['freeze_sha256']==sha(O/'freeze_week6.json')
assert decision['recorded_at_utc']<freeze['timestamp']<marker['timestamp']<done['timestamp']
assert new in (O/'phan_bien_tuan_06.md').read_text()
assert 'Timestamp/marker nội bộ không chứng thực' in (O/'ket_qua_tuan_06.md').read_text()
records=[{'file':n,'before_sha256':sha(versions/(Path(n).stem+'_v1.md')),'after_sha256':sha(O/n)} for n in ['phan_bien_tuan_06.md','ket_qua_tuan_06.md']]
(O/'review_response_checks.json').write_text(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(),'before_version':'v1 wording, unchanged numerical results','after_version':'v2 wording, unchanged numerical results','changes':records,'timeline_rechecked':True,'test_marker_freeze_hash_rechecked':True,'replacement_language_verified':True,'test_rerun':False},ensure_ascii=False,indent=2))
print('review response checks PASS')
