# Kết quả tuần 05 — bản đầy đủ

## Phạm vi và baseline
Hoàn tất phần còn lại của tuần 5: đóng băng trước test, tính metric test một lần, đối chiếu độc lập toàn bộ query và CI, so sánh ghép cặp với CenterCrop, mức ngẫu nhiên và mức nền popularity, mô tả phân nhóm, montage validation, hình và hồ sơ. Không triển khai fusion, trọng số, đối chứng ảnh gốc hay demo.

Baseline chính: B2-v1 theo luật validation đã đóng băng và đã được xác nhận trong user_decision.json. B2-v1 dùng bộ lọc nhãn 1–12, giữ nhãn 4 và 9 (mặt, tóc) và bỏ 13,15 (quần, giày) mà v2 giữ, nên KHÔNG đúng ngữ nghĩa. Kết quả chỉ là retrieval proxy trên validation, chưa có giải thích nguyên nhân. B2p và B2-v2 luôn là cấu hình đối chiếu; B0 là tham chiếu popularity giả lập.

Lưu ý trung thực: luật chọn được đặt sau khi đã thấy sanity tuần 4 trên test, trong đó v1 hơn v2 về Recall@5. Do đó test ở đây chỉ là xác nhận, không phải bằng chứng độc lập. Thứ tự user_decision → freeze → tính metric test chứng minh đợt chạy tiếp không dùng metric test mới để thay quyết định; không xóa được việc đã thấy test trước đó.

## Protocol, cấu hình và kiểm chứng
Không sửa protocol_v1.md hoặc category_rule_v2.json. SHA-256 được đối chiếu trong freeze.json, manifest_day_du.json. Split, ground truth cùng item_id khác ảnh, gallery theo thứ tự CSV, Top-10 stable và loại chính ảnh hỏi giữ nguyên. Recall macro cấp ảnh; Top-1 là Hit@1, không đồng nhất Recall@1. NDCG gain nhị phân/log2; MRR giới hạn Top-10. CI 95% percentile, bootstrap 10.000 lần seed 42 theo query. Môi trường Python 3.11.9, torch 2.14.1, torchvision 0.29.1, NumPy 2.2.1, Pillow 12.2.0; CPU, 4 luồng tính số, không trích vector mới trong đợt tiếp tục (batch 8 giữ ở lượt trích trước).

NormPop từ tuần 4 và t_q theo hash cố định của query. Popularity MAE: lỗi tuyệt đối trên các sản phẩm duy nhất xuất hiện trong Top-10 rồi trung bình query; xếp hạng vẫn ở cấp ảnh. B0 giảm dần NormPop và SHA-256 khi hòa, không tối ưu độ gần t_q. Vì vậy B0 không mặc nhiên có MAE nhỏ.

Em kiểm tra độc lập: 37210 phép đối chiếu, trong đó 18893 thuộc test, 0 không khớp. Mã độc lập không import module metric chính; tái tính query, aggregate, CI, bảng ghép cặp và mức nền. Tái tính kiểm chứng không phải một lần chọn/chạy test chính thức mới.

## Kết quả validation và lựa chọn baseline
Bảng chính dùng tỉ lệ 0–1; mỗi ô là ước lượng [CI thấp; CI cao]. Popularity MAE cũng ở thang 0–1.

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. B2p và B2-v2 được giữ làm đối chiếu.

| Metric | B0 | B2p | B2-v1 (sai ngữ nghĩa) | B2-v2 |
|---|---|---|---|---|
| Top-1 | 0.001761 [0.000000; 0.005282] | 0.308099 [0.269366; 0.346831] | 0.355634 [0.315141; 0.396127] | 0.322183 [0.283451; 0.360915] |
| Recall@1 | 0.001761 [0.000000; 0.005282] | 0.179677 [0.152275; 0.207613] | 0.204758 [0.175642; 0.233996] | 0.179109 [0.151967; 0.206095] |
| Recall@5 | 0.004464 [0.000440; 0.010249] | 0.318726 [0.285167; 0.352923] | 0.353902 [0.319082; 0.388688] | 0.313876 [0.281037; 0.347134] |
| Recall@10 | 0.008572 [0.002388; 0.016118] | 0.385775 [0.350585; 0.421711] | 0.413952 [0.378373; 0.450110] | 0.374174 [0.340192; 0.409420] |
| NDCG@5 | 0.003460 [0.000266; 0.008149] | 0.302086 [0.270849; 0.333423] | 0.334858 [0.303398; 0.366388] | 0.303799 [0.273039; 0.334425] |
| NDCG@10 | 0.004909 [0.001287; 0.009843] | 0.321716 [0.291110; 0.352180] | 0.350320 [0.319531; 0.381234] | 0.318556 [0.288615; 0.348189] |
| MRR@10 | 0.004025 [0.000787; 0.008715] | 0.376025 [0.339292; 0.412273] | 0.418478 [0.381302; 0.455242] | 0.385505 [0.348199; 0.421581] |
| Popularity MAE | 0.490073 [0.466753; 0.514187] | 0.329618 [0.320423; 0.338793] | 0.332787 [0.323713; 0.341695] | 0.338056 [0.328526; 0.347580] |

Bảng quy đổi sang % (MAE là lỗi tuyệt đối nhân 100, không phải accuracy); CI cũng nhân 100.

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. B2p và B2-v2 được giữ làm đối chiếu.

| Metric | B0 | B2p | B2-v1 (sai ngữ nghĩa) | B2-v2 |
|---|---|---|---|---|
| Top-1 | 0.18 [0.00; 0.53] | 30.81 [26.94; 34.68] | 35.56 [31.51; 39.61] | 32.22 [28.35; 36.09] |
| Recall@1 | 0.18 [0.00; 0.53] | 17.97 [15.23; 20.76] | 20.48 [17.56; 23.40] | 17.91 [15.20; 20.61] |
| Recall@5 | 0.45 [0.04; 1.02] | 31.87 [28.52; 35.29] | 35.39 [31.91; 38.87] | 31.39 [28.10; 34.71] |
| Recall@10 | 0.86 [0.24; 1.61] | 38.58 [35.06; 42.17] | 41.40 [37.84; 45.01] | 37.42 [34.02; 40.94] |
| NDCG@5 | 0.35 [0.03; 0.81] | 30.21 [27.08; 33.34] | 33.49 [30.34; 36.64] | 30.38 [27.30; 33.44] |
| NDCG@10 | 0.49 [0.13; 0.98] | 32.17 [29.11; 35.22] | 35.03 [31.95; 38.12] | 31.86 [28.86; 34.82] |
| MRR@10 | 0.40 [0.08; 0.87] | 37.60 [33.93; 41.23] | 41.85 [38.13; 45.52] | 38.55 [34.82; 42.16] |
| Popularity MAE | 49.01 [46.68; 51.42] | 32.96 [32.04; 33.88] | 33.28 [32.37; 34.17] | 33.81 [32.85; 34.76] |

CI hiệu Recall@5 v2−v1 trên validation [-0.073237; -0.007275] không chứa 0, hiệu -0.040026. Theo luật, chọn B2-v1 có Recall@5 cao hơn. Lựa chọn đã được xác nhận; không chọn lại dựa trên test và không coi mapping v1 đúng.

## Kết quả test xác nhận, ghép cặp và các mức nền
Bảng dùng tỉ lệ 0–1 và CI 95%. Test chính thức chạy một lần sau freeze, từ CSV Top-10 cũ đã xác nhận hash và số dòng.

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. B2p và B2-v2 được giữ làm đối chiếu.

| Metric | B0 | B2p | B2-v1 (sai ngữ nghĩa) | B2-v2 |
|---|---|---|---|---|
| Top-1 | 0.001706 [0.000000; 0.005119] | 0.302048 [0.264505; 0.339590] | 0.308874 [0.273038; 0.344710] | 0.312287 [0.274744; 0.349829] |
| Recall@1 | 0.000569 [0.000000; 0.001706] | 0.168588 [0.142931; 0.195091] | 0.175199 [0.149595; 0.201470] | 0.160367 [0.136258; 0.185751] |
| Recall@5 | 0.006257 [0.001138; 0.013083] | 0.321968 [0.289776; 0.355246] | 0.335115 [0.301963; 0.368201] | 0.296054 [0.264956; 0.327611] |
| Recall@10 | 0.009443 [0.002730; 0.017634] | 0.402118 [0.367568; 0.436520] | 0.396384 [0.362332; 0.431315] | 0.376889 [0.343190; 0.411069] |
| NDCG@5 | 0.003554 [0.000660; 0.007591] | 0.300317 [0.271060; 0.330336] | 0.312676 [0.282815; 0.343251] | 0.290541 [0.261394; 0.320445] |
| NDCG@10 | 0.004802 [0.001273; 0.009383] | 0.326294 [0.297213; 0.356195] | 0.331111 [0.301782; 0.360878] | 0.313626 [0.285090; 0.343015] |
| MRR@10 | 0.003671 [0.000782; 0.007941] | 0.374949 [0.339955; 0.410379] | 0.376222 [0.341759; 0.411012] | 0.379083 [0.344105; 0.414601] |
| Popularity MAE | 0.521174 [0.498227; 0.544318] | 0.337424 [0.328580; 0.346419] | 0.343670 [0.334861; 0.352457] | 0.335580 [0.326940; 0.344320] |

Quy đổi sang % như bảng validation (MAE nhân 100 chỉ là đổi thang).

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. B2p và B2-v2 được giữ làm đối chiếu.

| Metric | B0 | B2p | B2-v1 (sai ngữ nghĩa) | B2-v2 |
|---|---|---|---|---|
| Top-1 | 0.17 [0.00; 0.51] | 30.20 [26.45; 33.96] | 30.89 [27.30; 34.47] | 31.23 [27.47; 34.98] |
| Recall@1 | 0.06 [0.00; 0.17] | 16.86 [14.29; 19.51] | 17.52 [14.96; 20.15] | 16.04 [13.63; 18.58] |
| Recall@5 | 0.63 [0.11; 1.31] | 32.20 [28.98; 35.52] | 33.51 [30.20; 36.82] | 29.61 [26.50; 32.76] |
| Recall@10 | 0.94 [0.27; 1.76] | 40.21 [36.76; 43.65] | 39.64 [36.23; 43.13] | 37.69 [34.32; 41.11] |
| NDCG@5 | 0.36 [0.07; 0.76] | 30.03 [27.11; 33.03] | 31.27 [28.28; 34.33] | 29.05 [26.14; 32.04] |
| NDCG@10 | 0.48 [0.13; 0.94] | 32.63 [29.72; 35.62] | 33.11 [30.18; 36.09] | 31.36 [28.51; 34.30] |
| MRR@10 | 0.37 [0.08; 0.79] | 37.49 [34.00; 41.04] | 37.62 [34.18; 41.10] | 37.91 [34.41; 41.46] |
| Popularity MAE | 52.12 [49.82; 54.43] | 33.74 [32.86; 34.64] | 34.37 [33.49; 35.25] | 33.56 [32.69; 34.43] |

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. Các so sánh dưới đây là mô tả, không đổi baseline.

| Tập | Hiệu cấu hình | Metric | Hiệu (tỉ lệ) | CI 95% |
|---|---|---|---:|---|
| valid | B2-v2 minus B2-v1 | Recall@5 | -0.040026 | [-0.073237; -0.007275] |
| valid | B2-v2 minus B2-v1 | NDCG@5 | -0.031059 | [-0.058727; -0.003501] |
| valid | B2-v2 minus B2-v1 | MRR@10 | -0.032972 | [-0.064379; -0.001677] |
| valid | B2-v1 minus B2p | Recall@5 | 0.035176 | [0.004873; 0.065944] |
| valid | B2-v1 minus B2p | NDCG@5 | 0.032772 | [0.007872; 0.058527] |
| valid | B2-v1 minus B2p | MRR@10 | 0.042453 | [0.012997; 0.072871] |
| valid | B2-v2 minus B2p | Recall@5 | -0.004850 | [-0.040038; 0.029471] |
| valid | B2-v2 minus B2p | NDCG@5 | 0.001713 | [-0.027571; 0.031150] |
| valid | B2-v2 minus B2p | MRR@10 | 0.009480 | [-0.023935; 0.042259] |
| test | B2-v2 minus B2-v1 | Recall@5 | -0.039061 | [-0.071118; -0.007575] |
| test | B2-v2 minus B2-v1 | NDCG@5 | -0.022135 | [-0.049621; 0.004989] |
| test | B2-v2 minus B2-v1 | MRR@10 | 0.002860 | [-0.028081; 0.034387] |
| test | B2-v1 minus B2p | Recall@5 | 0.013146 | [-0.016582; 0.042920] |
| test | B2-v1 minus B2p | NDCG@5 | 0.012359 | [-0.012788; 0.038237] |
| test | B2-v1 minus B2p | MRR@10 | 0.001274 | [-0.027894; 0.031641] |
| test | B2-v2 minus B2p | Recall@5 | -0.025914 | [-0.056872; 0.004286] |
| test | B2-v2 minus B2p | NDCG@5 | -0.009776 | [-0.036160; 0.015501] |
| test | B2-v2 minus B2p | MRR@10 | 0.004134 | [-0.025808; 0.034181] |

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. McNemar exact hai phía; bảng hàng = cấu hình bị trừ, cột = cấu hình đứng trước, thứ tự sai/đúng. Không điều chỉnh đa kiểm định.

| Tập | Hiệu cấu hình | Bảng 2×2 | p exact |
|---|---|---|---:|
| valid | B2-v2 minus B2-v1 | [[314, 52], [71, 131]] | 0.104204 |
| valid | B2-v1 minus B2p | [[322, 71], [44, 131]] | 0.014965 |
| valid | B2-v2 minus B2p | [[330, 63], [55, 120]] | 0.519491 |
| test | B2-v2 minus B2-v1 | [[342, 63], [61, 120]] | 0.928492 |
| test | B2-v1 minus B2p | [[350, 59], [55, 122]] | 0.778876 |
| test | B2-v2 minus B2p | [[350, 59], [53, 124]] | 0.636800 |

Trên test, CI hiệu Recall@5 v2−v1 hoàn toàn âm; CI NDCG@5 và MRR@10 chứa 0 nên không kết luận cấu hình nào tốt hơn ở hai metric đó. McNemar cũng không cho bằng chứng khác biệt Top-1. So với B2p, các CI hiệu trên test đều chứa 0. Trên validation, v1−B2p có CI dương ở các metric ghép cặp, còn v2−B2p có CI chứa 0. Không suy kết quả test độc lập hoặc suy nguyên nhân từ các phép so sánh này.

Mức ngẫu nhiên: E[Top-1] = tổng Rq/(Q·G), E[Recall@5] = 5/G. Mức nền MAE lấy trung bình trên mọi sản phẩm gallery khác sản phẩm của query, rồi macro theo query; không trọng số theo số ảnh. Đây là đường tham chiếu, không phải một cấu hình xếp hạng được đánh giá. B0 so với mức ngẫu nhiên/mức nền chỉ mô tả, chưa kiểm định.

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. B2p/v2 giữ trong các bảng metric phía trên; bảng sau chỉ đối chiếu riêng B0 theo yêu cầu.

| Tập | Q | G | Cặp đúng | E[Top-1] | B0 Top-1 | E[Recall@5] | B0 Recall@5 | MAE nền | B0 MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 568 | 1597 | 1474 | 0.001625 | 0.001761 | 0.003131 | 0.004464 | 0.340432 | 0.490073 |
| test | 586 | 1543 | 1439 | 0.001591 | 0.001706 | 0.003240 | 0.006257 | 0.336287 | 0.521174 |

## Mô tả v1/v2, hình và category
Các nhóm dùng cùng danh sách query cho hai cấu hình. Trục nhóm chồng lấp, không cộng số query giữa các trục. Bbox nhỏ/lớn lấy đúng phần đầu/cuối của bbox_extremes.csv đã lưu; “không trong danh sách” không có nghĩa bbox bình thường. Không đặt lại ngưỡng sau khi xem metric. Query đen và gallery đen được xác định riêng theo audit bbox v1/v2.

B2-v1 KHÔNG đúng ngữ nghĩa: giữ 4,9; bỏ 13,15. Retrieval proxy; chưa giải thích nguyên nhân. Bảng nhóm chỉ so v1/v2 theo yêu cầu; B2p là đối chiếu ở bảng tổng thể.

| Tập | Trục / nhóm | n | Top-1 v1 | Top-1 v2 | Recall@5 v1 | Recall@5 v2 | Lưu ý |
|---|---|---:|---:|---:|---:|---:|---|
| valid | n_answers / 1 | 350 | 0.237143 | 0.200000 | 0.388571 | 0.322857 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | n_answers / 2 | 44 | 0.477273 | 0.409091 | 0.375000 | 0.397727 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | n_answers / 3+ | 174 | 0.563218 | 0.545977 | 0.278829 | 0.274607 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | query_black_v1 / black | 1 | 0.000000 | 0.000000 | 0.000000 | 0.333333 | n nhỏ, không diễn giải |
| valid | query_black_v1 / not_black | 567 | 0.356261 | 0.322751 | 0.354526 | 0.313842 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | query_black_v2 / black | 0 | — | — | — | — | n nhỏ, không diễn giải |
| valid | query_black_v2 / not_black | 568 | 0.355634 | 0.322183 | 0.353902 | 0.313876 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | bbox_v2 / small_extreme | 0 | — | — | — | — | n nhỏ, không diễn giải |
| valid | bbox_v2 / large_extreme | 1 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | n nhỏ, không diễn giải |
| valid | bbox_v2 / not_listed | 567 | 0.356261 | 0.320988 | 0.352762 | 0.312666 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | gallery_product_black_v1 / has_black | 4 | 0.500000 | 0.500000 | 0.100000 | 0.150000 | n nhỏ, không diễn giải |
| valid | gallery_product_black_v1 / no_black | 564 | 0.354610 | 0.320922 | 0.355703 | 0.315038 | Chỉ mô tả, chưa kiểm định nhóm |
| valid | gallery_product_black_v2 / has_black | 3 | 0.666667 | 0.666667 | 0.177778 | 0.188889 | n nhỏ, không diễn giải |
| valid | gallery_product_black_v2 / no_black | 565 | 0.353982 | 0.320354 | 0.354837 | 0.314540 | Chỉ mô tả, chưa kiểm định nhóm |
| test | n_answers / 1 | 342 | 0.201754 | 0.169591 | 0.339181 | 0.280702 | Chỉ mô tả, chưa kiểm định nhóm |
| test | n_answers / 2 | 63 | 0.365079 | 0.380952 | 0.373016 | 0.333333 | Chỉ mô tả, chưa kiểm định nhóm |
| test | n_answers / 3+ | 181 | 0.491713 | 0.558011 | 0.314239 | 0.312087 | Chỉ mô tả, chưa kiểm định nhóm |
| test | query_black_v1 / black | 3 | 0.000000 | 0.333333 | 0.444444 | 0.111111 | n nhỏ, không diễn giải |
| test | query_black_v1 / not_black | 583 | 0.310463 | 0.312178 | 0.334552 | 0.297006 | Chỉ mô tả, chưa kiểm định nhóm |
| test | query_black_v2 / black | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | n nhỏ, không diễn giải |
| test | query_black_v2 / not_black | 585 | 0.309402 | 0.312821 | 0.335688 | 0.296560 | Chỉ mô tả, chưa kiểm định nhóm |
| test | bbox_v2 / small_extreme | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | n nhỏ, không diễn giải |
| test | bbox_v2 / large_extreme | 1 | 0.000000 | 0.000000 | 0.000000 | 0.142857 | n nhỏ, không diễn giải |
| test | bbox_v2 / not_listed | 584 | 0.309932 | 0.313356 | 0.336262 | 0.296823 | Chỉ mô tả, chưa kiểm định nhóm |
| test | gallery_product_black_v1 / has_black | 6 | 0.166667 | 0.500000 | 0.335573 | 0.174283 | n nhỏ, không diễn giải |
| test | gallery_product_black_v1 / no_black | 580 | 0.310345 | 0.310345 | 0.335110 | 0.297314 | Chỉ mô tả, chưa kiểm định nhóm |
| test | gallery_product_black_v2 / has_black | 3 | 0.333333 | 0.666667 | 0.226703 | 0.237455 | n nhỏ, không diễn giải |
| test | gallery_product_black_v2 / no_black | 583 | 0.308748 | 0.310463 | 0.335673 | 0.296356 | Chỉ mô tả, chưa kiểm định nhóm |

Quan sát mô tả: ở nhóm một đáp án, Recall@5 v2 thấp hơn v1 trên cả validation và test. Ở nhóm nhiều đáp án trên test, Top-1 v2 cao hơn trong khi Recall@5 gần nhau; không gọi đây là khác biệt có ý nghĩa thống kê vì chưa kiểm định nhóm. Các nhóm ảnh đen/bbox cực trị quá ít query để diễn giải. Độ lớn chênh lệch và n nằm trong bảng; không chọn các nhóm này để thay baseline.

Giả thuyết chưa kiểm tra: hai mask giữ các vùng khác nhau, cụ thể v1 giữ mặt/tóc còn v2 giữ quần/giày; điều này có thể liên quan biểu diễn. Chưa có đối chứng để xác định ảnh hưởng hay quan hệ nhân quả. Không chạy đối chứng chỉ vùng mặt/tóc hoặc ảnh không mask ở tuần này.

Hình 1: Recall@5/NDCG@5 và CI validation/test. Hình 2: chuyển trạng thái Top-1. Hình 3 category cũ giữ nguyên. Hình 4: montage validation lấy query_id tăng dần trong mỗi chiều chuyển trạng thái, không chọn tay, không ngẫu nhiên. Hình 5: MAE cùng đường mức nền. Nguồn/caption và quy tắc chọn nằm ở figures/mo_ta_hinh_day_du.md.

Category: 88/100 KHÔNG ĐẠT, bỏ Category Match; w_c = 0 là trạng thái loại thành phần category, chưa triển khai/chỉnh trọng số mô hình fusion. Có 12 sản phẩm đánh dấu sai hoặc chưa xác nhận được. Đây là đánh giá sơ bộ tự động; vẫn cần con người xác nhận lại montage.

Phân loại ca sai từ ghi chú đã có (không chấm lại): 5 ca bộ đồ cả áo và quần, 2 ca mask đen, 5 ca váy/váy liền/playsuit, 1 ca váy liền/áo dài phối quần. Ca item 3304 nằm trong nhóm váy/váy liền/playsuit và nhóm váy liền/áo dài phối quần; đây là phân nhóm chồng lấp, không cộng thành số sản phẩm. Tổng sản phẩm duy nhất vẫn là 12. Xem category_error_groups_day_du.csv.

## Giới hạn và các số cần đính chính
Popularity và t đều giả lập; MAE thấp chỉ là gần t giả lập hơn, không phải lượt mua hay sở thích thật. Ground truth cùng sản phẩm khác ảnh đánh giá nhận diện/truy hồi sản phẩm. Mapping v2 do thực nghiệm tự xác định, chưa có tài liệu nhãn chính thức. Category một nhãn không phù hợp nhiều ảnh cả bộ đồ; việc nhìn một ảnh đại diện có thể không khớp majority sản phẩm, nên ca chưa xác nhận được đã tính sai bảo thủ. Giữ nguyên kết quả kiểm tra cũ, không dùng metric category.

Các số Category Match 55,72% / 80,50% / 79,75% trong báo cáo tuần 1 và Excel E15; 78% / 54% trong báo cáo tuần 3 và doi_chieu_cau_hinh_P.json cần đính chính vì dựa trên mapping cũ sai. Đường dẫn đầy đủ: can_dinh_chinh.md. Không sửa file nguồn, không thay bằng category metric mới.

README.md của data_protocol khác hash lịch sử trong validation_report.json nhưng không đổi giữa đầu/cuối lượt chạy trước hoặc lượt này. CSV split/inventory/ground truth vẫn khớp hash lịch sử. Không dùng README làm dữ liệu tính metric.

Chưa kiểm tra được: peak RAM toàn đợt; nguyên nhân chênh lệch v1/v2; ý nghĩa với người mua; độ đúng category do con người; khả năng tổng quát nhiều split/seed; định nghĩa trong paper trực tuyến. Nội dung API/PostgreSQL lịch sử trong Excel chép từ hồ sơ trước, chưa kiểm thử API lại ở lượt này. Không kết luận khác biệt khi CI chứa 0; các nhóm chỉ mô tả, nhiều phép so sánh chưa hiệu chỉnh đa kiểm định.

Quy tắc thực hiện: (i) phân loại ca category có một trường hợp giao nhau để giữ đúng tổng sản phẩm đã kiểm tra, không thêm ca thứ 13; (ii) “không đóng gói/demo” được hiểu là không xây gói phần mềm hoặc demo, vẫn tạo ZIP hồ sơ theo mục E.4; (iii) tên freeze.json, user_decision.json, t_q_test.csv, independent_test_checks.csv theo tên chỉ định, các đầu ra còn lại hậu tố _day_du; (iv) các bảng phân nhóm chỉ v1/v2 và bảng mức nền riêng B0 trong phạm vi đánh giá, mọi bảng tổng thể đều giữ B2p/v2; (v) không tự tạo một lần test mới khi kiểm chứng, chỉ tính độc lập từ cùng CSV đóng băng. Protocol, luật chọn và mọi artifact cũ không bị sửa.

## Minh chứng, toàn vẹn và tái chạy
Các file mới gồm freeze.json, user_decision.json, metrics_test_day_du.csv, paired_day_du.json, reference_levels_day_du.csv, independent_test_checks.csv, group_metrics_day_du.csv, montage_selection_day_du.csv, ảnh _day_du, bản sao Excel và manifest_day_du.json. Các báo cáo “đã dừng” cũ là lịch sử của lần bàn giao trước; bản đầy đủ này là kết quả chạy tiếp.

verification_day_du.txt ghi thứ tự thời gian user_decision → freeze → test, kèm timestamp nội dung và timestamp filesystem. hash_cu_truoc_day_du.json / hash_cu_sau_day_du.json chứng minh nguồn tuần 1–4 không đổi; hash_week5_truoc_day_du.json / hash_week5_sau_day_du.json bảo vệ toàn bộ artifact cũ tuần 5. doi_chieu_so_lieu_day_du.csv đối chiếu số báo cáo với triển khai độc lập; các giá trị lịch sử cần đính chính được đánh dấu nguồn lịch sử, không trình bày như metric mới được tính lại.

commands_day_du.txt ghi lệnh và thứ tự đã chạy. Các script dùng tạo file độc quyền để tránh ghi đè; không chạy lại vào thư mục giao nộp. Khi cần tái lập, dùng một thư mục sibling mới với bản sao đầu vào cần thiết. Không trích lại vector hoặc đổi ranking trong lượt này. Manifest liệt kê input/output, SHA-256, số dòng, môi trường, thời gian; RAM ghi chưa kiểm tra được. ZIP hồ sơ chứa cả file cũ và mới, kể cả ZIP cũ; không thể chứa chính ZIP đang được tạo.

## Dự kiến tuần 6
Chỉ cập nhật kế hoạch: cấu hình P visual + popularity trên baseline chính B2-v1 đã được chấp nhận, category hiện bị loại; lựa chọn trọng số trên validation, xem đánh đổi retrieval và MAE giả lập, test một lần theo protocol riêng phù hợp. Chưa triển khai các bước đó. Kế hoạch tuần 8 thêm đối chứng vùng mặt/tóc và crop/mask theo kế hoạch đã ghi; chưa chạy ở tuần này.
