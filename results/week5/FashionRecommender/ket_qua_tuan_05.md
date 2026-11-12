# Kết quả tuần 05 — dừng tại lựa chọn baseline

## Công việc đã thực hiện
Em đã đóng băng thiết kế thí nghiệm, kiểm thử metric, tái tạo sanity tuần trước, tính category v2, xem montage, trích vector CenterCrop validation và đánh giá các cấu hình trên validation. Chưa hoàn thành toàn bộ tuần 5: dừng tại bước chọn baseline khi luật chọn B2-v1. Không chạy test chính thức hoặc việc tuần sau.

## Protocol đóng băng
Nguồn: protocol_v1.md; SHA-256 `420b4a8205c81a6f629f3e5a3cc0ba04cee8ce57ae6137c74feb65b1d3c2bf25`. Split và ground truth giữ nguyên; mọi file cũ thuộc phạm vi hash trước/sau không đổi. Recall cấp ảnh, Top-1 là Hit@1; NDCG gain nhị phân; MRR ở kết quả đúng đầu tiên. CI percentile bootstrap theo query, seed cố định. Chi tiết định nghĩa trong protocol.

NormPop là giá trị giả lập tuần trước. t_q lấy từ hash query. Popularity MAE gộp các sản phẩm khác nhau có trong Top-10 trước khi macro theo query. Mã direction_a_repro.py trước đây tính lỗi trên danh sách gợi ý sản phẩm cho user giả lập và dùng trendiness RNG; tuần này dùng query và hash đã định trước. Chưa kiểm tra định nghĩa trong bản paper trực tuyến; không viện dẫn paper làm nguồn metric.

## Cấu hình và kiểm tra
B0: popularity giảm dần và SHA-256 khi hòa. B2p: crop cũ CenterCrop. B2-v1: bbox mask nhãn cũ. B2-v2: bbox mapping thực nghiệm mới. Xếp hạng visual stable theo gallery CSV, không gộp ảnh thành sản phẩm. Tất cả số sanity khớp JSON tuần trước và module tuần 3. Log: sanity_log.txt, unit_tests.txt.

Vector validation và kiểm tra từng byte: extraction.json, reproducibility_200.csv. Môi trường thực tế khớp môi trường yêu cầu. Category được suy từ thống kê pixel lưu trong tuần 4, chưa quét độc lập toàn bộ mask lần nữa; montage đọc trực tiếp archive.

## Kết quả validation và quyết định
Đơn vị trong bảng là tỉ lệ, kèm CI bootstrap (không phải phần trăm).

| Cấu hình | Recall@5 [CI] | NDCG@5 [CI] | MRR@10 [CI] | Popularity MAE [CI] |
|---|---|---|---|---|
| B0 | 0.004464 [0.000440; 0.010249] | 0.003460 [0.000266; 0.008149] | 0.004025 [0.000787; 0.008715] | 0.490073 [0.466753; 0.514187] |
| B2p | 0.318726 [0.285167; 0.352923] | 0.302086 [0.270849; 0.333423] | 0.376025 [0.339292; 0.412273] | 0.329618 [0.320423; 0.338793] |
| B2-v1 | 0.353902 [0.319082; 0.388688] | 0.334858 [0.303398; 0.366388] | 0.418478 [0.381302; 0.455242] | 0.332787 [0.323713; 0.341695] |
| B2-v2 | 0.313876 [0.281037; 0.347134] | 0.303799 [0.273039; 0.334425] | 0.385505 [0.348199; 0.421581] | 0.338056 [0.328526; 0.347580] |

Các metric Recall@1, Recall@10, NDCG@10 và Top-1 nằm đầy đủ trong metrics_valid.csv.

| Hiệu v2 − v1 | Giá trị | CI thấp | CI cao |
|---|---:|---:|---:|
| Recall@5 | -0.040026 | -0.073237 | -0.007275 |
| NDCG@5 | -0.031059 | -0.058727 | -0.003501 |
| MRR@10 | -0.032972 | -0.064379 | -0.001677 |

McNemar chính xác hai phía Top-1: p = 0.104204. Bảng hàng v1 (sai, đúng), cột v2 (sai, đúng): `[[314, 52], [71, 131]]`. Chưa có bằng chứng khác biệt Top-1 theo kiểm định này.

Luật chọn B2-v1 vì CI hiệu Recall@5 v2−v1 hoàn toàn âm. Không tự đổi sang v2. B2-v1 dùng bộ lọc sai ngữ nghĩa: cần xác nhận lựa chọn baseline trước khi tiếp tục. Đây là lựa chọn theo retrieval proxy trên validation, không phải xác nhận mapping v1 đúng. Các so sánh là phân tích định trước, không điều chỉnh đa kiểm định.

## Kết quả test
Chưa kiểm tra được chính thức: dừng tại lựa chọn baseline. Chưa tạo freeze.json và chưa tính metric/CI test. Các CSV test cũ đã được kiểm tra số dòng và hash, chỉ dùng Top-1/Recall@5 để tái tạo sanity theo bước đầu. Không trình bày sanity như kết quả test chính thức. Chưa tính mức ngẫu nhiên vì chưa tới bước test theo thứ tự đã định.

## Category và mô tả v1/v2
Kiểm tra category: 88/100 sản phẩm được đánh dấu đúng trong kiểm tra sơ bộ tự động, KHÔNG ĐẠT. Không tính Category Match. Mẫu có phân tầng, trường hợp chưa xác nhận được tính sai; không coi đây là độ chính xác nhãn chuẩn trên toàn tập. Cần người xem lại toàn bộ montage và category_check_100.csv.

Phân bố cũ được đối chiếu đúng với validation_report.json ở cấp ảnh theo category sản phẩm; phân bố mới lưu category_distribution_comparison.csv. Hệ nhãn mới tách quần, váy, áo khỏi các nhãn cũ không đúng ngữ nghĩa. Đây là thay cách gán nhãn, không thay dữ liệu.

Chỉ báo cáo bảng chuyển trạng thái validation ở trên; chưa thực hiện phân nhóm ảnh đen/bbox/số đáp án hoặc montage Top-5 vì đã chạm điểm dừng. Không kết luận nguyên nhân chênh lệch v1/v2.

## Vấn đề, giới hạn và đính chính
Category Match cũ 55,72% / 80,50% / 79,75% (báo cáo tuần 1 và Excel E15), 78% / 54% (báo cáo tuần 3 và doi_chieu_cau_hinh_P.json) không được dùng làm bằng chứng ngữ nghĩa. Đường dẫn trong can_dinh_chinh.md. Không sửa nguồn cũ.

README.md của data_protocol khác hash lịch sử trong validation_report.json nhưng khớp hash lúc bắt đầu tác vụ này; toàn bộ CSV split/ground truth/inventory vẫn khớp hash lịch sử. Chi tiết: split_source_checks.json.

Popularity và t đều giả lập; không có bằng chứng lượt mua. Mapping tự xác định, chưa có tài liệu nhãn chính thức; ground truth cùng sản phẩm không phải sở thích. Kiểm tra category qua một ảnh đại diện có thể không phản ánh majority toàn bộ ảnh sản phẩm; các trường hợp không chắc được chấm sai bảo thủ.

Phạm vi thực hiện: chỉ bàn giao phần trước điểm dừng; hình metric và chuyển trạng thái chỉ có validation; không tạo hình montage Top-5. Bổ sung CI Popularity MAE và các metric phụ. Lấy lại nguyên batch để kiểm tra byte của vector. Chọn tie category unknown sau nhãn số và tie diện tích theo nhãn nhỏ nhất, đã ghi trước khi tính. Popularity MAE tính trên sản phẩm duy nhất. Không tìm thấy kế hoạch tuần 5–15 nên không thể đối chiếu mục cập nhật Excel; cập nhật mô tả theo phần đã làm. Ô hoàn thành tuần 5 ghi trạng thái thay phần trăm ước lượng không có cơ sở. Chưa đo peak RAM toàn pipeline; chưa có số đo RAM.

Lần chạy sanity đầu có lỗi nhập tay hằng số tham chiếu v2 trong mã mới; đã sửa sang đọc trực tiếp JSON nguồn, không đổi protocol hoặc dữ liệu. Tái kiểm tra xác nhận khớp hoàn toàn. Không phải sai khác kết quả tuần trước.

## Minh chứng, tái chạy và việc tiếp theo
Xem manifest.json, hash_cu_truoc.json, hash_cu_sau.json, doi_chieu_so_lieu.csv, independent_validation_checks.csv và verification.txt. Kiểm tra độc lập mọi query validation cho mọi metric, đối chiếu aggregate và CI; không dùng module metric chính trong mã tính độc lập.

Lệnh đã chạy ở commands.txt. Không chạy lại initialize.py/prepare.py/evaluate.py vào thư mục này vì sẽ thay artifact; dùng thư mục sibling mới có cùng độ sâu, sao chép mã nguồn và chạy từ đó. Test chính thức chưa chạy. Mã evaluate.py nhánh test yêu cầu freeze.json, vốn chưa được tạo ở lần bàn giao này.

Tuần sau chỉ xem xét sau khi xác nhận lựa chọn baseline sai ngữ nghĩa; chưa triển khai fusion, trọng số hay các công việc ngoài tuần này.
