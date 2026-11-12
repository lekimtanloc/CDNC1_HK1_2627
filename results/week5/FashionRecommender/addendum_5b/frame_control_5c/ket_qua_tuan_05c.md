# Kết quả tuần 05c — kiểm soát khung hình

## Quyết định thiết kế và phạm vi
BỔ SUNG HẬU NGHIỆM, đặt sau khi đã thấy kết quả tuần 5 và 5b. Không đổi baseline chính B2-v1, dù bộ lọc baseline sai ngữ nghĩa. Không sửa protocol_v1.md, baseline_chinh.json, freeze.json hoặc bất kỳ hồ sơ cũ nào. Test đã được quan sát trước đây nên 5c không phải bằng chứng độc lập trên test chưa thấy.

Em dùng thiết kế G0–G4 bên dưới thay cho G0–G3 ban đầu. Màu tô và padding đều ĐEN (RGB 0,0,0), không dùng trung bình ImageNet. Chuẩn hóa đầu vào vẫn là Normalize ImageNet như pipeline cũ; pixel đen sau Normalize không đồng nghĩa tensor bằng không.

BBox C cố định cho mọi G: hợp nhãn [1,2,3,4,5,6,7,8,9,10,11,12,13,15]. Cùng bbox, cùng contain/BILINEAR vào 224×224, cùng căn giữa và padding. Nếu bbox C rỗng, mọi G trả ảnh đen như fallback v2, kể cả G2/G4. Nếu bbox C có nhưng nội dung giữ không còn pixel, ảnh có thể đen nhưng không được tính là bbox fallback.

BỔ SUNG HẬU NGHIỆM 5c; không đổi baseline B2-v1 (bộ lọc sai ngữ nghĩa); không kết luận nguyên nhân/nhận diện người.

| Biến thể | Pixel trước crop | Bbox |
|---|---|---|
| G0 = C | Giữ nhãn C, ngoài C tô đen | C |
| G1 | Giữ nhãn B=[1,2,3,5,6,7,8,10,11,12,13,15], còn lại đen | C |
| G2 | Giữ ảnh gốc, không tô đen | C |
| G3 | Chỉ giữ nhãn 4,9, còn lại đen | C |
| G4 | Ảnh gốc, chỉ tô đen nhãn 4,9 | C |
| B (5b) | Cùng pixel như G1 | B |
| F (5b) | Cùng pixel như G3 | F={4,9} |

## Tái lập và quy trình
design_5c.json ghi thiết kế, quy tắc đọc và trường hợp rỗng trước khi tính metric mới. G0 dùng đúng crop C tuần 4/5b, kiểm tra lại 200 ảnh trong 25 batch đầy đủ chọn seed42, giữ thứ tự/batch geometry của cache; tất cả khớp từng byte. Vì G0=C nên tái sử dụng toàn bộ cache C sau kiểm tra, không trích lại toàn bộ G0. G1–G4 chỉ trích validation/test, không trích train, không thay backbone hoặc weights. CPU, 4 luồng, batch8, deterministic; phiên bản môi trường và hash weights lưu trong manifest.

Ranking G0–G4 dựng từ vector chuẩn hóa bằng cosine (dot product), argsort stable theo gallery CSV, loại chính query, không gộp ảnh về sản phẩm. B/F dùng đúng Top-10 đã có ở 5b. Metric macro theo query; Top-1 là Hit@1; Recall@1 không đồng nhất Hit@1 khi nhiều đáp án. Binary NDCG, MRR@10 như protocol cũ. Bootstrap 10.000 lần, seed42, percentile95%, theo query. Validation trước; freeze_5c.json trước ranking/metric test; test5c chính thức một lần. Mã độc lập tính lại toàn bộ query, metric, aggregate, CI và paired; việc kiểm chứng không dùng để chọn lại cấu hình.

valid: 568 query; 1597 gallery; 1474 cặp đúng.

test: 586 query; 1543 gallery; 1439 cặp đúng.

Trích G1: 4294 ảnh, 273.049 giây (tổng thời gian các checkpoint, không gồm nạp model).

Trích G2: 4294 ảnh, 277.367 giây (tổng thời gian các checkpoint, không gồm nạp model).

Trích G3: 4294 ảnh, 248.710 giây (tổng thời gian các checkpoint, không gồm nạp model).

Trích G4: 4294 ảnh, 241.248 giây (tổng thời gian các checkpoint, không gồm nạp model).

| Biến thể | Tập | Bbox | Số ảnh | Fallback bbox rỗng | Tổng ảnh đen | Đen nhưng bbox không rỗng |
|---|---|---|---:|---:|---:|---:|
| G0 | valid | C | 2165 | 2 | 2 | 0 |
| G0 | test | C | 2129 | 4 | 4 | 0 |
| G1 | valid | C | 2165 | 2 | 3 | 1 |
| G1 | test | C | 2129 | 4 | 4 | 0 |
| G2 | valid | C | 2165 | 2 | 2 | 0 |
| G2 | test | C | 2129 | 4 | 4 | 0 |
| G3 | valid | C | 2165 | 2 | 121 | 119 |
| G3 | test | C | 2129 | 4 | 158 | 154 |
| G4 | valid | C | 2165 | 2 | 2 | 0 |
| G4 | test | C | 2129 | 4 | 4 | 0 |
| B | valid | B | 2165 | 3 | 3 | 0 |
| B | test | B | 2129 | 4 | 4 | 0 |
| F | valid | F | 2165 | 121 | 121 | 0 |
| F | test | F | 2129 | 158 | 158 | 0 |

## Metric validation
Đơn vị %, mỗi ô là giá trị [CI 95%]. CSV lưu tỉ lệ 0–1. B/F là đối chiếu cũ, không phải ứng viên thay baseline.

| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| G0 | 35.21 [31.34; 39.08] | 20.34 [17.49; 23.32] | 34.94 [31.56; 38.42] | 42.97 [39.41; 46.54] | 33.25 [30.14; 36.36] | 35.48 [32.48; 38.53] | 42.21 [38.57; 45.84] |
| G1 | 26.94 [23.24; 30.63] | 15.17 [12.60; 17.82] | 27.56 [24.43; 30.71] | 34.12 [30.77; 37.53] | 26.48 [23.62; 29.35] | 28.34 [25.50; 31.15] | 34.20 [30.67; 37.66] |
| G2 | 38.73 [34.68; 42.78] | 23.41 [20.46; 26.50] | 42.19 [38.62; 45.89] | 48.86 [45.22; 52.57] | 39.12 [35.90; 42.45] | 40.93 [37.82; 44.17] | 46.51 [42.80; 50.18] |
| G3 | 9.33 [7.04; 11.80] | 4.46 [3.07; 6.00] | 10.78 [8.59; 13.05] | 15.76 [13.18; 18.53] | 9.93 [8.01; 11.91] | 11.53 [9.61; 13.53] | 14.38 [11.91; 16.93] |
| G4 | 32.57 [28.70; 36.44] | 18.59 [15.85; 21.41] | 35.39 [31.94; 38.86] | 42.64 [39.07; 46.20] | 32.58 [29.54; 35.68] | 34.73 [31.74; 37.79] | 40.36 [36.78; 43.97] |
| B (đối chiếu 5b) | 32.22 [28.35; 36.09] | 17.91 [15.20; 20.61] | 31.39 [28.10; 34.71] | 37.42 [34.02; 40.94] | 30.38 [27.30; 33.44] | 31.86 [28.86; 34.82] | 38.55 [34.82; 42.16] |
| F (đối chiếu 5b) | 15.14 [12.32; 18.13] | 7.52 [5.71; 9.46] | 15.98 [13.40; 18.65] | 20.37 [17.47; 23.33] | 14.99 [12.68; 17.40] | 16.21 [13.88; 18.61] | 20.16 [17.18; 23.18] |

## Metric test hậu nghiệm
Cùng cách đọc và đơn vị; không diễn giải đây là một test độc lập mới.

| Biến thể | Top-1 | Recall@1 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| G0 | 32.42 [28.67; 36.18] | 16.93 [14.44; 19.53] | 32.87 [29.59; 36.17] | 38.88 [35.43; 42.36] | 31.35 [28.31; 34.44] | 33.18 [30.18; 36.20] | 39.11 [35.45; 42.66] |
| G1 | 28.50 [24.91; 32.25] | 13.59 [11.38; 15.91] | 25.56 [22.61; 28.56] | 33.03 [29.83; 36.34] | 25.44 [22.61; 28.29] | 27.60 [24.86; 30.38] | 34.43 [30.91; 37.98] |
| G2 | 37.71 [33.79; 41.64] | 22.11 [19.28; 25.10] | 40.64 [37.13; 44.20] | 48.84 [45.27; 52.43] | 37.78 [34.68; 40.98] | 40.32 [37.30; 43.44] | 45.52 [41.96; 49.09] |
| G3 | 10.07 [7.68; 12.63] | 4.03 [2.83; 5.33] | 11.57 [9.45; 13.78] | 16.53 [13.95; 19.08] | 10.20 [8.45; 12.07] | 11.79 [9.97; 13.66] | 14.98 [12.49; 17.52] |
| G4 | 30.72 [26.96; 34.64] | 16.73 [14.13; 19.37] | 33.12 [29.85; 36.45] | 41.14 [37.70; 44.70] | 30.96 [27.95; 33.98] | 33.32 [30.37; 36.31] | 38.38 [34.91; 42.01] |
| B (đối chiếu 5b) | 31.23 [27.47; 34.98] | 16.04 [13.63; 18.58] | 29.61 [26.50; 32.76] | 37.69 [34.32; 41.11] | 29.05 [26.14; 32.04] | 31.36 [28.51; 34.30] | 37.91 [34.41; 41.46] |
| F (đối chiếu 5b) | 14.68 [11.95; 17.58] | 8.00 [6.16; 9.98] | 17.98 [15.36; 20.71] | 24.91 [21.91; 28.07] | 16.08 [13.77; 18.48] | 18.30 [15.98; 20.69] | 21.14 [18.31; 24.03] |

## Hiệu ghép cặp và quy tắc đọc
Nhóm mặt/tóc: G0−G1 và G2−G4. Nhóm nền/vùng ngoài C: G2−G0 và G4−G1. Chỉ ghi “có” nếu cả hai CI Recall@5 loại 0 cùng chiều, nếu không thì “không kết luận”. Quy tắc được ghi trước khi chấm validation. “Có” chỉ là kết quả của quy tắc đọc, không là kết luận cơ chế hay nhân quả. G1−B và G3−F chỉ báo số/CI; G3−G0 là đối chiếu mô tả không đặt ngưỡng chọn.

| Tập | Hiệu | Metric | Hiệu (điểm %) | CI 95% (điểm %) |
|---|---|---|---:|---|
| valid | G1-B | Recall@5 | -3.825 | [-6.204; -1.372] |
| valid | G1-B | NDCG@5 | -3.903 | [-5.776; -2.012] |
| valid | G1-B | MRR@10 | -4.354 | [-6.613; -2.092] |
| valid | G0-G1 | Recall@5 | 7.378 | [4.416; 10.282] |
| valid | G0-G1 | NDCG@5 | 6.770 | [4.406; 9.159] |
| valid | G0-G1 | MRR@10 | 8.016 | [5.384; 10.697] |
| valid | G2-G4 | Recall@5 | 6.799 | [4.276; 9.427] |
| valid | G2-G4 | NDCG@5 | 6.532 | [4.474; 8.652] |
| valid | G2-G4 | MRR@10 | 6.148 | [3.877; 8.454] |
| valid | G2-G0 | Recall@5 | 7.253 | [3.807; 10.752] |
| valid | G2-G0 | NDCG@5 | 5.870 | [2.841; 8.843] |
| valid | G2-G0 | MRR@10 | 4.299 | [0.988; 7.538] |
| valid | G4-G1 | Recall@5 | 7.831 | [4.478; 11.125] |
| valid | G4-G1 | NDCG@5 | 6.107 | [3.387; 8.779] |
| valid | G4-G1 | MRR@10 | 6.167 | [3.125; 9.212] |
| valid | G3-F | Recall@5 | -5.200 | [-7.843; -2.568] |
| valid | G3-F | NDCG@5 | -5.061 | [-7.291; -2.848] |
| valid | G3-F | MRR@10 | -5.781 | [-8.549; -2.977] |
| valid | G3-G0 | Recall@5 | -24.161 | [-27.974; -20.457] |
| valid | G3-G0 | NDCG@5 | -23.316 | [-26.733; -20.066] |
| valid | G3-G0 | MRR@10 | -27.835 | [-31.728; -24.044] |
| test | G1-B | Recall@5 | -4.044 | [-6.273; -1.862] |
| test | G1-B | NDCG@5 | -3.613 | [-5.417; -1.833] |
| test | G1-B | MRR@10 | -3.479 | [-5.571; -1.400] |
| test | G0-G1 | Recall@5 | 7.309 | [4.859; 9.796] |
| test | G0-G1 | NDCG@5 | 5.909 | [3.788; 8.063] |
| test | G0-G1 | MRR@10 | 4.686 | [2.225; 7.190] |
| test | G2-G4 | Recall@5 | 7.517 | [4.908; 10.166] |
| test | G2-G4 | NDCG@5 | 6.819 | [4.750; 8.884] |
| test | G2-G4 | MRR@10 | 7.144 | [4.895; 9.413] |
| test | G2-G0 | Recall@5 | 7.771 | [4.340; 11.123] |
| test | G2-G0 | NDCG@5 | 6.434 | [3.680; 9.169] |
| test | G2-G0 | MRR@10 | 6.407 | [3.382; 9.460] |
| test | G4-G1 | Recall@5 | 7.563 | [4.475; 10.675] |
| test | G4-G1 | NDCG@5 | 5.524 | [2.846; 8.236] |
| test | G4-G1 | MRR@10 | 3.949 | [0.869; 6.976] |
| test | G3-F | Recall@5 | -6.415 | [-9.122; -3.809] |
| test | G3-F | NDCG@5 | -5.873 | [-8.126; -3.659] |
| test | G3-F | MRR@10 | -6.163 | [-8.972; -3.346] |
| test | G3-G0 | Recall@5 | -21.303 | [-24.860; -17.695] |
| test | G3-G0 | NDCG@5 | -21.148 | [-24.220; -17.985] |
| test | G3-G0 | MRR@10 | -24.138 | [-27.743; -20.398] |

BỔ SUNG HẬU NGHIỆM 5c; không đổi baseline B2-v1 (bộ lọc sai ngữ nghĩa); không kết luận nguyên nhân/nhận diện người. McNemar exact hai phía: hàng là cấu hình bị trừ, cột là cấu hình đứng trước; thứ tự sai/đúng.

| Tập | Hiệu | Bảng 2×2 | p exact |
|---|---|---|---:|
| valid | G1-B | [[362, 23], [53, 130]] | 0.000764582 |
| valid | G0-G1 | [[345, 70], [23, 130]] | 1.10263e-06 |
| valid | G2-G4 | [[325, 58], [23, 162]] | 0.000126611 |
| valid | G2-G0 | [[295, 73], [53, 147]] | 0.0901229 |
| valid | G4-G1 | [[339, 76], [44, 109]] | 0.00445502 |
| valid | G3-F | [[459, 23], [56, 30]] | 0.000263637 |
| valid | G3-G0 | [[350, 18], [165, 35]] | 6.37558e-31 |
| test | G1-B | [[380, 23], [39, 144]] | 0.0558972 |
| test | G0-G1 | [[361, 58], [35, 132]] | 0.0220186 |
| test | G2-G4 | [[348, 58], [17, 163]] | 2.18233e-06 |
| test | G2-G0 | [[324, 72], [41, 149]] | 0.00455356 |
| test | G4-G1 | [[355, 64], [51, 116]] | 0.263054 |
| test | G3-F | [[472, 28], [55, 31]] | 0.00403921 |
| test | G3-G0 | [[377, 19], [150, 40]] | 1.87582e-26 |

| Tập | Nhóm | Kết luận quy tắc | Chiều |
|---|---|---|---|
| valid | Mặt/tóc | có | positive |
| valid | Nền/vùng ngoài C | có | positive |
| test | Mặt/tóc | có | positive |
| test | Nền/vùng ngoài C | có | positive |

Mặt/tóc: validation có, test có, theo đúng quy tắc hai CI đã ghi trước. Không suy nguyên nhân từ phân loại này.

Nền/vùng ngoài C: validation có, test có, theo đúng quy tắc hai CI đã ghi trước. Không suy nguyên nhân từ phân loại này.

Không gọi cấu hình tốt hơn ở metric đang xét nếu CI hiệu chứa 0. Không dùng kết quả test để chọn baseline, biến thể, nhóm hiệu ứng hoặc cách đọc khác. Các CI/p-value chưa điều chỉnh đa kiểm định; mọi so sánh đều hậu nghiệm so với toàn bộ chuỗi tuần trước.

## G3, F và mức ngẫu nhiên

| Tập | G3 Top-1 (%) | F Top-1 (%) | G0 Top-1 (%) | E[Top-1] (%) | G3 Recall@5 (%) | F Recall@5 (%) | G0 Recall@5 (%) | E[Recall@5] (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 9.331 | 15.141 | 35.211 | 0.162 | 10.779 | 15.978 | 34.940 | 0.313 |
| test | 10.068 | 14.676 | 32.423 | 0.159 | 11.567 | 17.982 | 32.870 | 0.324 |

Mức ngẫu nhiên E[Top-1]=tổng Rq/(Q·G), E[Recall@5]=5/G; so với ngẫu nhiên chỉ mô tả, chưa kiểm định null. G3−F giữ cùng pixel được chọn nhưng đổi bbox/scale/padding, nên là đối chiếu khung; không được đọc như việc thêm thông tin mặt/tóc. G3 so G0 chỉ mô tả khả năng truy hồi của phần pixel nhãn 4,9 trong khung C. Dù G3 cao hơn ngẫu nhiên hoặc thấp hơn G0, điều đó không chứng minh mô hình nhận diện người mẫu hoặc không dùng tín hiệu người, vì không có nhãn định danh người.

## Giới hạn
Khung cố định kiểm soát bbox/scale/padding giữa các G nhưng thao tác tô đen vẫn thay tương phản, đường biên và phân bố pixel. G2 vẫn dùng mask C để xác định bbox, nên không phải đối chứng hoàn toàn không dùng mask. Tên vùng dựa trên mapping thực nghiệm chưa có tài liệu chính thức, mask có thể sai. “Nền/vùng ngoài C” gồm cả nhãn0 và nhãn14 (nền và da), không phải nền thuần túy. Không quy chênh lệch cho một cơ chế cụ thể. Ground truth là cùng sản phẩm khác ảnh, không là sở thích hay giao dịch. CI theo query có điều kiện trên gallery cố định, không bao phủ biến thiên split, seed hay đổi gallery. Không có nhãn người để kiểm chứng nhận diện người mẫu.

Chưa kiểm tra được: nguyên nhân/cơ chế, hiệu quả người dùng hay lượt mua, nhãn người, độ bền nhiều split/seed và peak RAM toàn đợt. Test đã được thấy trước đây nên không xem CI là bằng chứng độc lập ngoài chuỗi hậu nghiệm.

Diễn giải thực thi: G0 tái dùng cache C sau kiểm tra200 ảnh thay vì trích lại toàn bộ; G1–G4 trích mới. B/F giữ nguyên ranking5b làm đối chiếu. Fallback theo bbox C rỗng được áp dụng cả ảnh gốc G2/G4, đúng fallback đen pipeline cũ. Tính thêm CI và McNemar cho G3−G0 để minh bạch đối chiếu mô tả, không đặt ngưỡng. NDCG@10 được giữ để đủ metric retrieval của protocol. Checkpoint mỗi256 ảnh, batch vẫn8. Báo cáo dùng tên ket_qua_tuan_05c.md, còn các artifact khác hậu tố _5c; ZIP hồ sơ không phải gói phần mềm/demo. Thiết kế G0–G3 cũ và màu trung bình ImageNet đã được thay thế theo quyết định thiết kế.

## Hình, minh chứng và tái chạy
Hình 1. Recall@5 G0–G4 cùng B/F tham chiếu trên validation/test, CI95%, trục y từ0; nguồn metrics_valid_5c.csv và metrics_test_5c.csv. Hình 2. Forest plot hiệu ghép cặp Recall@5 theo các tương phản; nguồn paired_valid_5c.json và paired_test_5c.json. Cả hai hình nền trắng, tiếng Việt, 180dpi, rộng không quá1600px; ghi rõ hậu nghiệm và baseline không đổi.

Xem manifest_5c.json, doi_chieu_so_lieu_5c.csv, verification_5c.txt, hash_cu_truoc_5c.json và hash_cu_sau_5c.json. Mã nguồn nằm trong frame_control_5c, không sửa mã cũ. commands_5c.txt ghi thứ tự đã chạy. Checkpoint có hash vector/audit/source/design, có thể tiếp tục trong trường hợp chỉ checkpoint đã commit được tái dùng. Không chạy lại vào thư mục giao nộp; dùng thư mục sibling mới cùng độ sâu. ZIP chứa toàn bộ thư mục5c, gồm vector/checkpoint, không chứa chính ZIP và không thay gói5b cũ.
