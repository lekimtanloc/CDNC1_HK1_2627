# Kết quả tuần 05b — phân tích bổ sung hậu nghiệm

## Phạm vi và thiết kế
BỔ SUNG HẬU NGHIỆM, đặt sau khi thấy tuần 5. Không đổi baseline chính B2-v1; A là v1 và vẫn dùng bộ lọc sai ngữ nghĩa. Không sửa protocol, split gốc, ground truth gốc hoặc hồ sơ cũ. Không huấn luyện backbone, không fusion hay tìm trọng số. Test đã được quan sát trong các tuần trước, vì vậy kết quả 5b không phải bằng chứng độc lập trên dữ liệu chưa thấy.

Thiết kế được ghi ở design_5b.json trước khi chấm dev/ablation. D là nhóm cơ sở; A thêm mặt/tóc, B thêm quần/giày, C thêm cả hai; F chỉ mặt/tóc. Các tên vùng theo mapping thực nghiệm, không bảo đảm mọi pixel đúng.

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Biến thể | Nhãn giữ | Vai trò |
|---|---|---|
| A | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | v1, tái dùng fixed_all.npy |
| B | [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15] | v2, tái dùng fixed_all_v2.npy |
| C | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15] | B thêm 4,9 |
| D | [1, 2, 3, 5, 6, 7, 8, 10, 11, 12] | A bỏ 4,9 |
| F | [4, 9] | chỉ vùng nhãn 4,9 |

“Có hiệu ứng” chỉ là tên quy tắc đọc: cả hai CI ghép cặp Recall@5 tương ứng phải không chứa 0 và cùng chiều. Mặt/tóc: C−B và A−D; quần/giày: C−A và B−D. Nếu không đạt, ghi “không kết luận”. Không gán ý nghĩa nhân quả. F chỉ báo số so A và mức ngẫu nhiên, không đặt ngưỡng hoặc đổi baseline. Không có biến thể E được định nghĩa trong yêu cầu.

## Tái dựng split và trích vector
Hàm build_split trong preflight_5b.py được tách từ logic build_rows của prepare_week2_splits.py: giữ assignment sản phẩm, sắp ảnh, dùng Random theo item_id để chọn ảnh hỏi; không đọc role hoặc ground truth cũ làm đầu vào. Role và category sản phẩm được dựng lại. Validation/test tái tạo đúng từng dòng; CSV split/ground truth còn khớp từng byte. Query/gallery được đối chiếu theo thứ tự dòng của CSV split gốc, vì tuần 2 không có file query/gallery riêng. Train cũng khớp bản cũ: dev chính là protocol retrieval trên train, không phải tạo split ngẫu nhiên mới.

dev: 2660 query, 6798 gallery, 6239 cặp đúng.

valid: 568 query, 1597 gallery, 1474 cặp đúng.

test: 586 query, 1543 gallery, 1439 cặp đúng.

A kiểm tra lại 200 ảnh thuộc 25 batch đầy đủ, chọn seed 42, giữ đúng cấu trúc batch gốc: khớp từng byte. B tái sử dụng nguyên vector. C/D/F gọi trực tiếp load/crop của v2 tuần 4; fallback giữ ảnh đen khi không có pixel thuộc tập nhãn; cùng Resize bbox/contain/pad, Normalize và ResNet-50 IMAGENET1K_V2. CPU, batch 8, 4 luồng mỗi tiến trình, deterministic; môi trường và hash weights nằm trong hồ sơ. Trích C/D/F chỉ trên validation và test, không trích train. Checkpoint bất biến mỗi 256 ảnh; mỗi checkpoint có vector, audit và hash. Trích trước vector test không bao gồm chấm metric test.

Thời gian trích (không gồm nạp model):

C: 205.014 giây; 4294 ảnh.

D: 207.146 giây; 4294 ảnh.

F: 219.054 giây; 4294 ảnh.

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Biến thể | Tập | Số ảnh | Fallback | Ảnh đen |
|---|---|---:|---:|---:|
| A | valid | 2165 | 8 | 8 |
| A | test | 2129 | 9 | 9 |
| B | valid | 2165 | 3 | 3 |
| B | test | 2129 | 4 | 4 |
| C | valid | 2165 | 2 | 2 |
| C | test | 2129 | 4 | 4 |
| D | valid | 2165 | 9 | 9 |
| D | test | 2129 | 9 | 9 |
| F | valid | 2165 | 121 | 121 |
| F | test | 2129 | 158 | 158 |

## Dev: v1 so với v2
Các bảng dùng %, ô là giá trị [CI 95%]; CSV lưu tỉ lệ 0–1. Bootstrap 10.000 lần seed 42 theo query. Dev chỉ so A/v1 và B/v2. Đối chiếu độc lập kiểm tra toàn bộ query, vượt yêu cầu ít nhất 200.

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| A | 21.24 [19.70; 22.78] | 11.97 [10.91; 13.07] | 22.36 [20.97; 23.76] | 27.41 [25.94; 28.90] | 20.88 [19.65; 22.13] | 22.44 [21.21; 23.68] | 27.15 [25.63; 28.66] |
| B | 21.62 [20.04; 23.20] | 11.53 [10.52; 12.61] | 20.59 [19.28; 21.91] | 26.21 [24.77; 27.64] | 19.92 [18.71; 21.14] | 21.59 [20.39; 22.79] | 26.93 [25.38; 28.47] |

Hiệu Recall@5 v2−v1 trên dev: -1.77 điểm phần trăm, CI [-3.12; -0.41]. Hướng hiệu âm cùng hướng với validation/test tuần 5; không so trực tiếp độ lớn metric dev với các split khác vì gallery và độ khó khác nhau. Dev không được dùng để huấn luyện ResNet trong thí nghiệm này, nhưng vẫn là phân tích hậu nghiệm trên dữ liệu đã tham gia quá trình khảo sát trước đây.

## Ablation validation và test
Validation được chấm và kiểm chứng trước freeze_5b.json. Test chỉ xếp hạng/chấm một lần sau freeze; Cả năm biến thể dựng ranking từ vector tương ứng, cosine và stable theo gallery CSV; A/B được đối chiếu kết quả tuần 5. Kiểm chứng độc lập sau đó không thay cấu hình hoặc chọn mẫu.

Validation:

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| A | 35.56 [31.51; 39.61] | 20.48 [17.56; 23.40] | 35.39 [31.91; 38.87] | 41.40 [37.84; 45.01] | 33.49 [30.34; 36.64] | 35.03 [31.95; 38.12] | 41.85 [38.13; 45.52] |
| B | 32.22 [28.35; 36.09] | 17.91 [15.20; 20.61] | 31.39 [28.10; 34.71] | 37.42 [34.02; 40.94] | 30.38 [27.30; 33.44] | 31.86 [28.86; 34.82] | 38.55 [34.82; 42.16] |
| C | 35.21 [31.34; 39.08] | 20.34 [17.49; 23.32] | 34.94 [31.56; 38.42] | 42.97 [39.41; 46.54] | 33.25 [30.14; 36.36] | 35.48 [32.48; 38.53] | 42.21 [38.57; 45.84] |
| D | 29.58 [25.88; 33.27] | 17.14 [14.38; 19.86] | 30.81 [27.53; 34.15] | 35.96 [32.53; 39.44] | 29.46 [26.43; 32.52] | 30.78 [27.81; 33.75] | 36.62 [33.01; 40.16] |
| F | 15.14 [12.32; 18.13] | 7.52 [5.71; 9.46] | 15.98 [13.40; 18.65] | 20.37 [17.47; 23.33] | 14.99 [12.68; 17.40] | 16.21 [13.88; 18.61] | 20.16 [17.18; 23.18] |

Test hậu nghiệm:

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| A | 30.89 [27.30; 34.47] | 17.52 [14.96; 20.15] | 33.51 [30.20; 36.82] | 39.64 [36.23; 43.13] | 31.27 [28.28; 34.33] | 33.11 [30.18; 36.09] | 37.62 [34.18; 41.10] |
| B | 31.23 [27.47; 34.98] | 16.04 [13.63; 18.58] | 29.61 [26.50; 32.76] | 37.69 [34.32; 41.11] | 29.05 [26.14; 32.04] | 31.36 [28.51; 34.30] | 37.91 [34.41; 41.46] |
| C | 32.42 [28.67; 36.18] | 16.93 [14.44; 19.53] | 32.87 [29.59; 36.17] | 38.88 [35.43; 42.36] | 31.35 [28.31; 34.44] | 33.18 [30.18; 36.20] | 39.11 [35.45; 42.66] |
| D | 30.89 [27.30; 34.64] | 17.40 [14.84; 20.08] | 31.73 [28.46; 34.99] | 39.96 [36.52; 43.42] | 29.95 [27.00; 32.93] | 32.66 [29.74; 35.60] | 38.02 [34.50; 41.50] |
| F | 14.68 [11.95; 17.58] | 8.00 [6.16; 9.98] | 17.98 [15.36; 20.71] | 24.91 [21.91; 28.07] | 16.08 [13.77; 18.48] | 18.30 [15.98; 20.69] | 21.14 [18.31; 24.03] |

## Hiệu ghép cặp và quy tắc đọc
BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Tập | Hiệu | Metric | Hiệu (điểm %) | CI 95% (điểm %) |
|---|---|---|---:|---|
| dev | B-A | Recall@5 | -1.773 | [-3.115; -0.409] |
| dev | B-A | NDCG@5 | -0.963 | [-2.113; 0.227] |
| dev | B-A | MRR@10 | -0.216 | [-1.606; 1.184] |
| valid | C-B | Recall@5 | 3.552 | [0.695; 6.504] |
| valid | C-B | NDCG@5 | 2.867 | [0.310; 5.438] |
| valid | C-B | MRR@10 | 3.663 | [0.772; 6.613] |
| valid | A-D | Recall@5 | 4.580 | [1.733; 7.468] |
| valid | A-D | NDCG@5 | 4.027 | [1.607; 6.463] |
| valid | A-D | MRR@10 | 5.230 | [2.520; 8.013] |
| valid | C-A | Recall@5 | -0.450 | [-3.331; 2.303] |
| valid | C-A | NDCG@5 | -0.239 | [-2.465; 1.946] |
| valid | C-A | MRR@10 | 0.365 | [-2.124; 2.860] |
| valid | B-D | Recall@5 | 0.578 | [-2.255; 3.373] |
| valid | B-D | NDCG@5 | 0.921 | [-1.294; 3.067] |
| valid | B-D | MRR@10 | 1.933 | [-0.705; 4.562] |
| valid | F-A | Recall@5 | -19.412 | [-23.459; -15.451] |
| valid | F-A | NDCG@5 | -18.494 | [-21.981; -15.098] |
| valid | F-A | MRR@10 | -21.689 | [-25.648; -17.800] |
| test | C-B | Recall@5 | 3.265 | [0.624; 5.922] |
| test | C-B | NDCG@5 | 2.296 | [0.120; 4.496] |
| test | C-B | MRR@10 | 1.207 | [-1.303; 3.725] |
| test | A-D | Recall@5 | 1.779 | [-1.132; 4.676] |
| test | A-D | NDCG@5 | 1.323 | [-1.003; 3.629] |
| test | A-D | MRR@10 | -0.398 | [-2.968; 2.221] |
| test | C-A | Recall@5 | -0.641 | [-3.140; 1.767] |
| test | C-A | NDCG@5 | 0.082 | [-2.044; 2.210] |
| test | C-A | MRR@10 | 1.493 | [-0.995; 4.030] |
| test | B-D | Recall@5 | -2.127 | [-4.832; 0.494] |
| test | B-D | NDCG@5 | -0.891 | [-3.125; 1.350] |
| test | B-D | MRR@10 | -0.112 | [-2.578; 2.390] |
| test | F-A | Recall@5 | -15.529 | [-19.377; -11.731] |
| test | F-A | NDCG@5 | -15.192 | [-18.510; -11.886] |
| test | F-A | MRR@10 | -16.482 | [-20.313; -12.675] |

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa). Bảng McNemar: hàng cấu hình bị trừ, cột cấu hình đứng trước; thứ tự sai/đúng.

| Tập | Hiệu | Bảng 2×2 | p exact hai phía |
|---|---|---|---:|
| dev | B-A | [[1854, 241], [231, 334]] | 0.678729 |
| valid | C-B | [[321, 64], [47, 136]] | 0.128475 |
| valid | A-D | [[332, 68], [34, 134]] | 0.000986537 |
| valid | C-A | [[327, 39], [41, 161]] | 0.911072 |
| valid | B-D | [[349, 51], [36, 132]] | 0.132902 |
| valid | F-A | [[338, 28], [144, 58]] | 5.2105e-20 |
| test | C-B | [[356, 47], [40, 143]] | 0.520292 |
| test | A-D | [[360, 45], [45, 136]] | 1 |
| test | C-A | [[359, 46], [37, 144]] | 0.379999 |
| test | B-D | [[361, 44], [42, 139]] | 0.914212 |
| test | F-A | [[367, 38], [133, 48]] | 1.48053e-13 |

BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Tập | Nhóm hiệu ứng | Kết luận theo quy tắc | Chiều |
|---|---|---|---|
| valid | face_hair | có | positive |
| valid | pants_shoes | không kết luận | không xác định |
| test | face_hair | không kết luận | không xác định |
| test | pants_shoes | không kết luận | không xác định |

Không gọi một cấu hình tốt hơn nếu CI hiệu đang xét chứa 0. “Có” chỉ phản ánh hai tương phản thỏa quy tắc trên tập đó; không xác định cơ chế. Các CI là từng phép so sánh, chưa hiệu chỉnh đa kiểm định. Phân tích này không ước lượng riêng tương tác factorial để suy một hiệu ứng chính chung; báo đúng các tương phản được yêu cầu.

## F so A và mức ngẫu nhiên
BỔ SUNG HẬU NGHIỆM sau tuần 5; không đổi baseline chính B2-v1 (bộ lọc sai ngữ nghĩa).

| Tập | F Top-1 (%) | A Top-1 (%) | E[Top-1] (%) | F Recall@5 (%) | A Recall@5 (%) | E[Recall@5] (%) |
|---|---:|---:|---:|---:|---:|---:|
| valid | 15.141 | 35.563 | 0.162 | 15.978 | 35.390 | 0.313 |
| test | 14.676 | 30.887 | 0.159 | 17.982 | 33.511 | 0.324 |

Mức ngẫu nhiên E[Top-1]=tổng Rq/(Q·G), E[Recall@5]=5/G, so sánh chỉ mô tả, chưa kiểm định F so phân phối null. Hiệu F−A và CI nằm trong bảng ghép cặp. F có khả năng truy hồi trên mức ngẫu nhiên chỉ cho thấy vùng pixel được gán 4,9 chứa tín hiệu liên quan cùng sản phẩm trong bộ ảnh này; không chứng minh nhận diện người mẫu, vì không có nhãn người. F thấp hơn A cũng không chứng minh mô hình không dùng dấu hiệu người. Mask có sai nhãn, crop/scale thay đổi và tương quan người–sản phẩm có thể cùng tham gia; chưa tách được.

## Giới hạn
Toàn bộ phân tích đặt sau khi thấy kết quả tuần 5, kể cả việc chọn biến thể F. Test đã được xem trước đó, không phải test độc lập mới. Dev là train cũ, không thay thế test giữ kín. CI lấy mẫu query có điều kiện trên gallery cố định, không bao phủ biến thiên giữa split, seed hoặc việc đổi gallery. Không có nhãn người, sở thích hay giao dịch thật. Mapping nhãn do thực nghiệm, chưa có tài liệu chính thức. Category đã thất bại ở tuần 5 và không được dùng ở đây. Chưa kiểm tra được nguyên nhân, nhận diện người mẫu, hiệu quả mua hàng, độ bền nhiều split/seed và peak RAM toàn đợt.

Em chỉ đánh giá A,B,C,D,F; E chưa được định nghĩa trong thiết kế; bổ sung NDCG@10 để đủ bộ metric retrieval của protocol trước; kiểm chứng độc lập toàn bộ query thay vì chỉ 200; A/B tái dùng vector có sẵn và dựng ranking theo cùng cơ chế, đối chiếu metric tuần 5; dev dựng ranking trên gallery train. Checkpoint theo 256 ảnh, batch suy luận vẫn 8. Dev đánh giá trong lúc tiến trình trích chạy nền, mỗi tiến trình giới hạn 4 luồng, không cam kết tổng hệ thống chỉ 4 luồng khi hai tiến trình trùng thời gian. Tên ket_qua_tuan_05b.md và tuan5b_bao_cao_goi.zip dùng đúng tên chỉ định dù không kết thúc bằng _5b; các file khác có hậu tố _5b.

## Hình, minh chứng và tái chạy
Hình 1. Recall@5 A,B,C,D,F trên validation và test với CI, trục y từ 0, nguồn metrics_valid_5b.csv và metrics_test_5b.csv. Hình 2. Forest plot hiệu ghép cặp Recall@5, nguồn paired_valid_5b.json và paired_test_5b.json. Hình nền trắng, tiếng Việt, 180 dpi, rộng không quá 1600px; mọi hình ghi hậu nghiệm và không đổi baseline.

manifest_5b.json ghi input/output, SHA-256, số dòng, môi trường, thời gian, nguồn crop/weights. doi_chieu_so_lieu_5b.csv ghi giá trị báo cáo và đối chiếu độc lập. verification_5b.txt ghi tái tạo split, 200 byte checks A, kiểm chứng độc lập, freeze trước test và bảo toàn file cũ. commands_5b.txt ghi lệnh, checkpoint và cách tái lập ở thư mục mới. ZIP chứa toàn bộ addendum, gồm checkpoint và vector; không sửa ZIP hoặc hồ sơ tuần trước. Baseline chính không thay đổi sau bất kỳ kết quả nào của 5b.
