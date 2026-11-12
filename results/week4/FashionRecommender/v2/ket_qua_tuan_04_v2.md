# Kết quả tuần 04 v2 — sửa bộ lọc mask

## Công việc đã thực hiện
Đã hoàn tất xác minh thực nghiệm mapping, qua điểm dừng rồi trích lại 13,752 vector ảnh, mỗi vector 2,048 chiều. Giữ split sản phẩm 3.219/691/690, cùng 586 query, 1,543 gallery, 1,439 cặp đúng. Không huấn luyện, đổi backbone, đổi split hay chạy nhiều seed.

Đã xuất Top-10 visual v2 (5,860 dòng) và bản tham chiếu popularity giống byte v1. Trích hai lần cùng tập con 512 ảnh gồm các ảnh đặc biệt; 512 vector bằng byte, max |Δ|=0.0. Hash trước/sau xác nhận 279 file cũ không đổi, gồm tuần 1–3, mã/cache cũ, artifacts_completed và addendum.

## Mapping nhãn và cách sửa crop
Mapping là đề xuất thực nghiệm, chưa có tài liệu chính thức nối bảng mã với archive adjusted. Đã xem 384 ô mẫu chính và tổng 516 ô kể cả bổ sung. Mẫu chính chọn seed 42, trải đều các split; có thống kê tâm/diện tích/co-xuất hiện toàn bộ mask. Nguồn và giới hạn: [nguon_mapping.md](nguon_mapping.md); atlas, từng mẫu và ngoại lệ: [mapping/nhan_atlas.md](mapping/nhan_atlas.md).

| Nhãn | Tên vùng đề xuất | Tin cậy | Giữ/bỏ |
| --- | --- | --- | --- |
| 0 | Nền | cao | bỏ |
| 1 | Áo thân trên | cao | giữ |
| 2 | Chân váy / váy ngắn | cao | giữ |
| 3 | Tất / quần tất | trung_binh | giữ |
| 4 | Mặt | cao | bỏ |
| 5 | Váy liền | cao | giữ |
| 6 | Túi / dây túi | cao | giữ |
| 7 | Trang sức / phụ kiện nhỏ | cao | giữ |
| 8 | Áo khoác / lớp ngoài | cao | giữ |
| 9 | Tóc | cao | bỏ |
| 10 | Mũ / đồ đội đầu | cao | giữ |
| 11 | Kính | cao | giữ |
| 12 | Thắt lưng / áo buộc eo | cao | giữ |
| 13 | Quần dài / quần ngắn | cao | giữ |
| 14 | Da / tay / chân / cổ | cao | bỏ |
| 15 | Giày / dép / bốt | cao | giữ |

Nhãn có độ tin cậy trung_binh được hiểu là trung bình-cao cho quyết định giữ tất/quần tất, có ngoại lệ nên không ghi cao. Các nhóm còn lại rõ trên nhiều mẫu đã xem. Qua điểm dừng; không có nhóm thấp/chưa rõ đáng kể cần chạy v2b. Mức tin cậy không phải độ chính xác segmentation đo được.

Tập giữ: `[1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15]`; bỏ nền, mặt, tóc, da (`[0,4,9,14]`). So v1, thêm nhãn 13/15 và bỏ 4/9; không đổi cách resize, pad hay chuẩn hóa:
```python
selected = np.isin(mask, keep_labels)

masked = np.where(selected[..., None], image_rgb, 0)

y, x = np.where(selected)

fixed = Image.new("RGB", (224, 224))

if len(x):

    bbox = (x.min(), y.min(), x.max()+1, y.max()+1)
    fitted = ImageOps.contain(Image.fromarray(masked).crop(bbox),
                              (224, 224), Image.Resampling.BILINEAR)
    fixed.paste(fitted, ((224-fitted.width)//2, (224-fitted.height)//2))
```
Ảnh không có nhãn giữ được để đen, không thay bằng ảnh gốc. Sau đó normalize ImageNet, ResNet-50 IMAGENET1K_V2, L2-normalize. Audit từng ảnh có nhãn, bbox, diện tích và trạng thái đen.

Ảnh đen theo train/valid/test: 14/3/4; tổng 21. Không có pixel nhãn giữ: 21, gồm 13 toàn nền và 8 có nhãn ngoài tập giữ. Bbox/diện tích ảnh (tính cả bbox rỗng=0): min=0.000000, p01=0.238566, p05=0.282482, median=0.438573, p95=0.693310, p99=0.813333, max=1.000000. Có 2 bbox dương dưới 1% ảnh và 15 bbox trên 95%; danh sách cực trị ở bbox_extremes.csv, đây là cờ kiểm tra chứ không tự kết luận lỗi.

Cosine hai ảnh 1687_011 và 3445_011: v1=0.49510920, v2=0.63727754. Ảnh gốc, vùng giữ và crop v2 của cả hai cùng 1000_031 ở [special_crop_v2.png](artifacts_v2/special_crop_v2.png).

## Hai baseline
Visual-only xếp ảnh gallery theo cosine giảm dần; `argsort(kind="stable")` giữ thứ tự gallery test.csv khi hòa điểm. Loại chính query, không gộp ảnh về sản phẩm, không lọc category. CSV có `query_id,rank,candidate_id,product_id,score,is_correct`, K=10.

Popularity-only độc lập vector: sao chép nguyên byte CSV xếp hạng, popularity.csv và lineage v1; cả ba hash khớp trong popularity_reference.json. Nguồn là `week3_verify_and_smoke.py::smoke()`: `default_rng(42).integers(0,101,len(metadata))`, chuẩn hóa min–max, giữ thứ tự metadata cũ; inventory.csv/tracking.sqlite3 không chứa số mua hàng. Giá trị vẫn là giả lập; xếp giảm dần, hòa điểm theo SHA-256 của `42:candidate_path`. Không sinh lại giá trị hoặc thêm baseline.

Ảnh đen test v2: 1 query và 3 gallery (black_test.csv). Các vector đen bằng byte: True; cosine đen×đen gần 1 (chi tiết số float32 ở tie_diagnostics.json). Thứ hạng các gallery đen theo từng query có trong black_gallery_ranks.csv. Khi query đen, điểm hòa trong nhóm đen không phản ánh độ giống trang phục; stable gallery order quyết định thứ hạng giữa chúng. Không đổi tie-break để làm đẹp kết quả.

## Sanity check và so sánh v1–v2
Chỉ kiểm tra pipeline, chưa phải bảng đánh giá chính thức. Dùng nguyên week3_retrieval_metrics.evaluate: Top-1 lấy reciprocal_rank ở K=1; Recall@5 là số ảnh đáp án đúng trong Top-5 / tổng số ảnh đáp án của query, trung bình đều trên query. Cấp ảnh, không phải hit-rate “ít nhất một đúng”. Không báo NDCG/MRR hay metric khác.

| Cấu hình | Top-1 | Recall@5 |
| --- | --- | --- |
| v1_visual | 30.8874% | 33.5115% |
| v2_visual | 31.2287% | 29.6054% |
| popularity_only | 0.1706% | 0.6257% |

### So sánh v1 và v2
| Đại lượng / cùng phạm vi | v1 | v2 |
| --- | --- | --- |
| Ảnh đen toàn bộ | 48 | 21 |
| Ảnh đen test | 9 | 4 |
| Vector trùng toàn bộ | 48 | 21 |
| Sản phẩm trong nhóm trùng khác sản phẩm | 39 | 21 |
| Vector trùng trên ảnh đại diện | 11 | 2 |
| Top-1 (sanity) | 30.8874% | 31.2287% |
| Recall@5 (sanity) | 33.5115% | 29.6054% |

Vector trùng là bằng byte float32; ảnh đại diện là đúng tập metadata cũ, không đổi phạm vi so sánh. Mọi con số bảng được tính lại trong doi_chieu_so_lieu.csv.

### Chuyển trạng thái Top-1
| Nhóm query | n | Đúng→sai | Sai→đúng | Đúng cả hai | Sai cả hai |
| --- | --- | --- | --- | --- | --- |
| all | 586 | 61 | 63 | 120 | 342 |
| black_v1 | 3 | 0 | 1 | 0 | 2 |
| black_v2 | 1 | 0 | 0 | 0 | 1 |
| neither_black | 583 | 61 | 62 | 120 | 340 |
| black_both | 1 | 0 | 0 | 0 | 1 |
| black_v1_only | 2 | 0 | 1 | 0 | 1 |
| black_v2_only | 0 | 0 | 0 | 0 | 0 |

Hai hàng black_v1/black_v2 có thể giao nhau; bốn nhóm black_both/black_v1_only/black_v2_only/neither_black là phân hoạch. Chỉ đếm, không kiểm định và không kết luận “cải thiện” có ý nghĩa. Dữ liệu từng query: top1_chuyen_trang_thai_v1_v2.csv.

## Hình minh họa
Sáu hình tiếng Việt ở figures/: hinh_01_atlas_nhan.png, hinh_02_crop_v1_v2.png, hinh_03_top5_v1_v2.png, hinh_04_so_do_pipeline.png, hinh_05_so_sanh_sanity.png, hinh_06_anh_den_v2.png. Nguồn, cách chọn mẫu và caption nằm trong [figures/mo_ta_hinh.md](figures/mo_ta_hinh.md). Atlas đầy đủ nằm trong mapping/; không chỉ dựa vào hình tóm tắt.

## Vấn đề / giới hạn
Chưa xác minh được tài liệu đánh số chính thức của archive. Mapping dựa trên atlas ngẫu nhiên và ảnh phóng vùng; không coi là nhãn do nhà cung cấp xác nhận. Có lỗi mask riêng lẻ, ví dụ vùng mặt bị gán nhãn tất, váy bị gán nhãn da. Phân biệt subtype của chân váy/quần ngắn, trang sức, áo buộc eo không tuyệt đối nhưng quyết định giữ cùng nhóm.

Category tuần 2 suy từ mapping cũ và được dùng stratify sản phẩm; vì thế phân tầng không phải category đã xác minh. Ground truth chỉ dùng cùng item_id, không lọc theo category; vẫn giữ split để so công bằng.

Popularity giả lập; bài toán cùng sản phẩm khác góc, không phải hành vi người dùng/phối đồ. Nhiều ảnh cùng sản phẩm có thể cùng chiếm Top-K.

Lần trích đầu bị ngắt bởi lỗi I/O khi đọc archive, chưa lưu vector; đã đọc lại và kiểm tra SHA-256 khớp trước khi chạy lại thành công. Nhật ký: execution_attempts.json. Thời gian dưới đây chỉ tính lần hoàn tất.

Không bảo đảm mask đúng tại mọi pixel, vector bằng byte trên mọi máy/phiên bản; kiểm tra tái lập chỉ trong môi trường hiện tại. Thời gian/RSS là số đo của tiến trình lần chạy, không đo toàn máy.

## Minh chứng và tái chạy
mapping/: mapping_de_xuat.csv, decision.json, nhan_atlas.md, nhan_thong_ke.csv, thong_ke_tung_anh.csv, dong_xuat_hien.csv, on_dinh_giua_split.csv, atlas_samples.csv và PNG atlas.

artifacts_v2/: fixed_all_v2.npy + index; crop_audit_v2.csv, crop_checks_v2.json, bbox_extremes.csv; visual_only_v2_top10.csv, popularity tham chiếu; sanity_checks_v2.json; replay_1.npy/replay_2.npy + index + reproducibility.json; black_test.csv, black_gallery_ranks.csv, tie_diagnostics.json; manifest_v2.json.

hash_cu_truoc.json/hash_cu_sau.json: hash trước/sau file cũ; doi_chieu_so_lieu.csv và verified_values.json: tính lại số; verification_v2.txt: log kiểm tra. manifest_ban_giao_v2.json và muc_luc_goi.md kiểm kê bản bàn giao; file_npy_bo_qua.txt ghi NPY để riêng nếu gói quá lớn.

Môi trường giữ như v1: Python 3.11.9, torch 2.14.1, torchvision 0.29.1, NumPy 2.2.1, Pillow 12.2.0; CPU, batch 8, 4 threads, seed 42, deterministic algorithms. Weights SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. Pipeline 972.71 giây, peak RSS 791.22 MiB; gồm extraction chính và hai lần tập con, không gồm tạo atlas/đóng gói.

Lệnh từ .., theo thứ tự (xem atlas trước khi ghi quyết định):
```bash
python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/build_mapping_evidence.py

python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/record_mapping_review.py

python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/run_v2.py

python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/verify_and_report.py

python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/make_figures_v2.py

python3 -B CDNC1_HK1_2627/results/week4/FashionRecommender/v2/package_v2.py
```
Mã từ chối ghi đè đầu ra. Tái chạy extraction có thể dùng `--output-dir` thư mục mới, nhưng file chẩn đoán cấp v2 cũng phải nằm trong bản sao cây thư mục mới chưa có đầu ra. Không chạy nguyên chuỗi lên bản bàn giao. Việc gán mapping là bước xem ảnh có ghi nhận, không phải suy tự động từ thống kê.

## Dự kiến tuần 5
Chốt protocol và giới hạn mapping, sau đó mới lập bảng metric đánh giá chính thức hai baseline. Chưa thực hiện tuần 5 trong lần này.
