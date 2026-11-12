# Kết quả tuần 04 — bản sửa diễn giải sau kiểm tra

Em sửa cách diễn giải mask theo bằng chứng A1–A7 trong `xac_nhan.md` (hồ sơ đối chiếu lưu trên máy), giữ nguyên số liệu của pipeline đã chạy.

## Công việc đã thực hiện
Kiểm tra SHA-256 archive và CSV theo validation_report.json tuần 2; không tạo lại split. train: 3,219 sản phẩm / 9,458 ảnh; valid: 691 sản phẩm / 2,165 ảnh; test: 690 sản phẩm / 2,129 ảnh.

Trích ResNet-50 pretrained cho đủ 13,752 ảnh, mỗi ảnh một vector 2.048 chiều, L2-normalize, không huấn luyện. Chạy thêm crop cũ trên 2.129 ảnh test; giữ nguyên cache và kết quả tuần 1–3.

Xuất hai baseline và bản visual trước sửa, mỗi CSV 5.860 dòng (586 query × Top-10), cùng 1.543 ảnh ứng viên và 1.439 cặp đúng. Chạy 3 test metric có sẵn tuần 3; kiểm tra độc lập hash, vector, thứ hạng, nhãn đúng/sai bằng verify_week4.py.

## Sửa CenterCrop

Đính chính: sửa crop chỉ giữ đầy đủ vùng được bộ lọc 1–12 chọn, không bảo đảm giữ đầy đủ trang phục. Kiểm tra trực quan item 1687 thấy nhãn 13 là quần, 15 là giày; cả hai bị loại. Mẫu 1000_031 còn cho thấy nhãn 4 là mặt và 9 là tóc, vẫn bị giữ lại. 48 mask không có nhãn 1–12 gồm 13 toàn nền và 35 vẫn có nhãn khác. Mapping adjusted chưa xác minh đầy đủ; không tự sửa bộ lọc hoặc trích lại vector.

Giữ pixel có mask nhãn 1–12 (quy ước cũ, chưa đúng đầy đủ về ngữ nghĩa trang phục), đặt nền đen; lấy bounding box bao toàn bộ pixel này, resize bilinear giữ tỉ lệ rồi pad giữa khung 224×224. Chuẩn hóa ImageNet như tuần 3. Không CenterCrop nữa. Mask không có nhãn 1–12 được giữ đen và đánh dấu riêng, không dùng ảnh gốc thay thế.

```python
y, x = np.where((mask >= 1) & (mask <= 12))

if not len(x):

    fixed = Image.new("RGB", (224, 224))  # audit: empty mask
else:

    bbox = (x.min(), y.min(), x.max() + 1, y.max() + 1)
    fitted = ImageOps.contain(masked.crop(bbox), (224, 224),
                              Image.Resampling.BILINEAR)
    fixed = Image.new("RGB", (224, 224))
    fixed.paste(fitted, ((224-fitted.width)//2, (224-fitted.height)//2))
```

| Phép kiểm tra (cùng phạm vi mỗi hàng) | Trước | Sau |
|---|---|---|
| Ảnh test hoàn toàn đen / 2.129 | 100 | 9 |
| Vector trùng hệt trên 2.129 ảnh test | 100 vector trong 1 nhóm; 42 sản phẩm trùng khác sản phẩm | 9 vector trong 1 nhóm; 7 sản phẩm trùng khác sản phẩm |
| 4.600 ảnh đại diện đúng theo metadata cache cũ | 162 vector trong 1 nhóm; 162 sản phẩm trùng khác sản phẩm | 11 vector trong 1 nhóm; 11 sản phẩm trùng khác sản phẩm |
| Hai ảnh đại diện item 1687 và 3445 | Trùng hệt: True | Trùng hệt: False; cosine 0.49510920 |

Trên toàn bộ 13,752 ảnh sau sửa: 48 ảnh đen; 48 mask không có nhãn 1–12 vùng nhãn 1–12; 48 vector trong 1 nhóm; 39 sản phẩm trùng khác sản phẩm. Tập ảnh còn đen trùng chính xác tập mask không có nhãn 1–12; vector trùng sau sửa cũng chính là nhóm này. Không có phép trích crop cũ toàn bộ ảnh, nên không suy diễn số trước sửa cho phạm vi này. “Trùng hệt” là bằng toàn bộ byte float32 của vector đã chuẩn hóa, không phải gần nhau theo cosine. Mốc 162 thuộc 4.600 ảnh đại diện/sản phẩm, khác phạm vi 2.129 ảnh test.

Ảnh đối chiếu riêng: `images/1687_011.jpg` và `images/3445_011.jpg`. crop_checks.json còn lưu trạng thái đen của tất cả góc chụp hai sản phẩm.

## Hai baseline
Visual-only: cosine giữa vector ảnh hỏi và từng ảnh gallery test, giảm dần; đồng điểm giữ thứ tự gallery trong test.csv (stable sort). Không gộp ảnh thành sản phẩm; nhiều ảnh của cùng sản phẩm có thể cùng xuất hiện. Loại chính query bằng image_path; query và gallery vốn rời nhau.

Popularity-only: cùng gallery, xếp giảm dần popularity sản phẩm; mọi ảnh của sản phẩm nhận cùng điểm. Đồng hạng sắp theo SHA-256 của chuỗi `42:{candidate_image_path}`, rồi image_path, cố định seed 42. Không dùng vector, độ giống ảnh, category hay trendiness trong điểm xếp hạng.

Nguồn popularity: inventory.csv và schema tracking.sqlite3 không có trường popularity. Khôi phục đúng phép sinh trong `FashionRecommender/week3_verify_and_smoke.py::smoke`: `np.random.default_rng(42).integers(0,101,len(metadata))`, min-max về [0,1], theo đúng thứ tự `repro_outputs_resnet50_full/metadata.csv`. Đây là giá trị giả lập, không phải số lượt mua; không dùng điểm tương thích popularity–trendiness của tuần 3. Lưu 4.600 giá trị cùng số nguyên gốc trong popularity.csv; nguồn/schema ở popularity_lineage.json.

K=10. Ba CSV cùng cột `query_id,rank,candidate_id,product_id,score,is_correct`; ID ảnh là đường dẫn trong archive, product_id là item_id; is_correct tra ground_truth_test.csv. Gallery có cả ảnh khác góc và sản phẩm một ảnh làm negative; không lọc category.

## Sanity check — chỉ để kiểm tra pipeline
| Lần kiểm tra | Top-1 | Recall@5 |
|---|---:|---:|
| Visual cũ — CSV tuần 3 | 30.2048% | 32.1968% |
| Visual crop cũ — chạy lại tuần 4 | 30.2048% | 32.1968% |
| Visual sửa crop | 30.8874% | 33.5115% |
| Popularity-only | 0.1706% | 0.6257% |

Dùng nguyên `week3_retrieval_metrics.evaluate`; Top-1 lấy reciprocal_rank ở K=1, Recall@5 lấy recall ở K=5. Chỉ lưu hai số này, không lập bảng metric chính thức; đánh giá chính thức để tuần 5. Hàng CSV cũ được tính lại từ file tuần 3, không chép số làm tròn từ báo cáo. Nếu có sai khác chạy lại crop cũ, không quy thành tác dụng của sửa crop; hàng chạy lại là đối chứng cùng môi trường tuần 4.

## Hình minh họa

Bộ sáu hình mới có tiếng Việt, nguồn và caption: [figures/mo_ta_hinh.md](figures/mo_ta_hinh.md). Bằng chứng vùng nhãn: [label_regions_1687.png](label_regions_1687.png), [label_regions_all.png](label_regions_all.png).

[../artifacts_completed/crop_before_after.png](../artifacts_completed/crop_before_after.png): ảnh gốc, CenterCrop đen, bbox đã sửa.

[../artifacts_completed/visual_top5_1.png](../artifacts_completed/visual_top5_1.png), [../artifacts_completed/visual_top5_2.png](../artifacts_completed/visual_top5_2.png), [../artifacts_completed/visual_top5_3.png](../artifacts_completed/visual_top5_3.png): ba query đầu theo thứ tự test có cả đúng và sai trong Top-5, không phải mẫu ngẫu nhiên. DUNG/SAI dựa trên ground truth; hiển thị ảnh gốc để dễ xem, xếp hạng vẫn dùng ảnh masked đã sửa.

## Vấn đề / giới hạn
Từ addendum: không được xem 48 mask không có nhãn 1–12 là 48 mask toàn nền. Bộ lọc bỏ vùng quần/giày trong mẫu kiểm tra; đầy đủ mapping archive chưa xác minh. Hai baseline vẫn là kết quả thực chạy với bộ lọc cũ.

3 query và 6 gallery test đen đồng vector; thứ tự stable gallery quyết định rank trong nhóm hòa điểm.

Chance kỳ vọng Top-1 0,159147%, Recall@5 0,324044%; chưa kiểm định khác biệt của popularity. Đếm Top-1 visual: 59 sai→đúng, 55 đúng→sai, 122 đúng cả hai, 350 sai cả hai. Không suy ra ý nghĩa thống kê.

Popularity giả lập độc lập nhãn, không chứng minh khả năng dự đoán nhu cầu thật. Dùng lại cách sinh của smoke tuần 3 thay vì số mua từ mô phỏng khác; giả định này và thứ tự ánh xạ được lưu rõ.

Ground truth cùng sản phẩm, khác góc chụp: đây là image retrieval, không phải sở thích người dùng hay phối đồ. Không gộp ảnh sản phẩm nên số góc chụp ảnh hưởng số vị trí xếp hạng.

Có mask không có nhãn 1–12 vùng nhãn 1–12. Lần chạy đầu dừng tại images/1012_021.jpg; thư mục `artifacts/` chỉ chứa đầu ra dở, không dùng làm kết quả. Lần hoàn tất ở `../artifacts_completed/`; vẫn giữ mọi ảnh theo split, không loại ảnh khó.

Không đảm bảo bitwise giữa mọi phiên bản thư viện/phần cứng. Cố định seed 42, CPU, 4 threads, deterministic algorithms, batch 8; không thử seed khác hay masking so với ảnh gốc.

## File minh chứng và tái chạy

Bổ sung: xac_nhan.md, mask_rong_chi_tiet.csv, mask_label_frequency.csv, doi_chieu_so_lieu.csv, audit_results.json và muc_luc_goi.md. File cũ được giữ nguyên, kiểm tra bằng original_files_sha256.json.

Tất cả đường dẫn dưới `results/week4/FashionRecommender/`:
`run_week4.py`: pipeline và hàm crop; `verify_week4.py`: kiểm tra đầu ra; `make_report.py`: dựng báo cáo từ JSON/CSV thật.
`../artifacts_completed/fixed_all.npy` + `fixed_all_index.csv`: vector mỗi ảnh và ánh xạ dòng/split.
`../artifacts_completed/old_test.npy` + `old_test_index.csv`: vector crop cũ trích lại, không ghi đè cache cũ.
`../artifacts_completed/visual_only_top10.csv`, `popularity_only_top10.csv`, `visual_before_top10.csv`: xếp hạng ảnh cùng schema.
`../artifacts_completed/popularity.csv`, `popularity_lineage.json`: giá trị giả lập và nguồn.
`../artifacts_completed/crop_audit.csv`, `crop_checks.json`: ảnh đen, mask không có nhãn 1–12, nhóm vector trùng, item 1687/3445.
`../artifacts_completed/sanity_checks.json`: hai số kiểm tra pipeline cho mỗi cấu hình.
`../artifacts_completed/manifest.json`: input/output, số dòng CSV/vector, SHA-256, môi trường và thời gian; manifest không tự hash chính nó.
`verification.txt`: kết quả kiểm tra độc lập; `delivery_manifest.json`: hash báo cáo, mã và log kiểm tra.

Weights `ResNet50_Weights.IMAGENET1K_V2`, SHA-256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. Môi trường: Python 3.11.9, torch 2.14.1, torchvision 0.29.1, NumPy 2.2.1, Pillow 12.2.0. Thời gian pipeline 915.47 giây; peak RSS 815.61 MiB (ru_maxrss của tiến trình, không phải toàn máy).

Lệnh chạy từ thư mục cha của repo:
```bash
python3 CDNC1_HK1_2627/results/week4/FashionRecommender/run_week4.py --batch-size 8 --output-dir CDNC1_HK1_2627/results/week4/FashionRecommender/artifacts_completed

python3 -m unittest discover -s CDNC1_HK1_2627/FashionRecommender -p test_week3_retrieval_metrics.py

python3 CDNC1_HK1_2627/results/week4/FashionRecommender/verify_week4.py CDNC1_HK1_2627/results/week4/FashionRecommender/artifacts_completed

python3 CDNC1_HK1_2627/results/week4/FashionRecommender/make_report.py CDNC1_HK1_2627/results/week4/FashionRecommender/artifacts_completed
```
Tái chạy pipeline: dùng nguyên lệnh trên nhưng đổi `--output-dir` sang thư mục mới chưa tồn tại (ví dụ `.../artifacts_rerun`); mã từ chối ghi đè. Không chạy lại split tuần 2.

## Dự kiến tuần 5
Đánh giá chính thức hai baseline trên protocol đã cố định, kiểm tra định nghĩa metric và lập bảng báo cáo. Chưa thực hiện trong tuần 4.
