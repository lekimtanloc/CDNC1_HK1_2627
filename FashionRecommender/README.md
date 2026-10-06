# FashionRecommender - huong A da chay duoc

Repo nay duoc dung lam **huong chinh** cho de tai khuyen nghi san pham thoi trang dua tren anh san pham va tin hieu popularity/trendiness.

Trang thai hien tai: **da chay end-to-end duoc tren dataset that** bang runner tai lap da sua loi, va da tao split/ground truth doc lap cho cac buoc danh gia tiep theo.

## Ket Qua Da Xac Nhan

| Hang muc | Ket qua |
|---|---|
| Dataset | Kaggle Adjusted archive, co cap `images/*.jpg` va `masks/*.png` |
| So file anh trong archive | 13.752 |
| Item ID duy nhat co du anh + mask | 4.600 |
| Image/mask thieu cap | 0 |
| Image/mask sai kich thuoc | 0 |
| Moi truong da chay | macOS ARM64, Python 3.11.9, PyTorch 2.14.1, Apple MPS |
| Backbone chinh | ResNet-50 pretrained |
| Seed | 42 |
| So user tong hop | 50 |
| Top-K | 5 |
| Thoi gian end-to-end | 216,49 giay |
| Peak RAM | Khoang 767 MiB |

## Ket Qua Smoke / Reproduction Run

Ket qua chay huong A tren 4.600 item ID:

| Bien the | Category exact match | Popularity MAE |
|---|---:|---:|
| Popularity-only | 55,72% | 0,0023 |
| Visual-only | **80,50%** | 0,3451 |
| Visual + popularity | 79,75% | **0,1913** |

Nhan xet ngan:

- Visual-only dat category match cao nhat trong smoke run.
- Visual + popularity giam Popularity MAE khoang 44,6% so voi visual-only, nhung category match giam nhe 0,75 diem.
- Ket qua nay xac nhan pipeline chay thong suot, chua du de ket luan khoa hoc cuoi cung vi ground truth hanh vi that chua co.

## Cac Sua Loi / Dieu Chinh Chinh

Repo goc co mot so diem khong khop voi mo ta va gay sai lech khi bao cao ket qua. Ban hien tai dung runner rieng `direction_a_repro.py` de tai lap toi thieu va tranh cac loi do luong.

| Van de | Trang thai |
|---|---|
| `tensorflow==2.8.0` khong cai duoc tren Python 3.11 | Khong dung trong runner moi |
| Gender Alignment do sai item | Khong bao cao metric nay trong smoke run |
| Category score trong repo goc khong tham gia ranking | Runner moi bao cao ro cac bien the scoring |
| `--P_threshold` khai bao `int` nhung mac dinh `0.15` | Can sua neu dung `final_recommandation.py` goc |
| README goc khac code ve so user / purchase / Top-K | README nay ghi theo ket qua da chay thuc te |

## Cau Truc File Quan Trong

| File / thu muc | Vai tro |
|---|---|
| `direction_a_repro.py` | Runner tai lap huong A da chay duoc tren archive that |
| `data_check.py` | Kiem tra archive, cap anh/mask, phan phoi category |
| `split_items.py` | Tao split train/val/test co dinh, stratified theo category |
| `make_ground_truth.py` | Tao ground truth doc lap dua tren category, khong dung popularity |
| `data_check_report.json` | Bao cao kiem tra du lieu da sinh |
| `week3_splits/` | Split va manifest phuc vu danh gia tu tuan 4 tro di |
| `../bao_cao_xac_thuc_6_buoc.md` | Bao cao tong hop 6 buoc va quyet dinh chon huong |

## Cai Dat Moi Truong

Khuyen nghi tao moi truong rieng:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision pillow numpy
```

Neu muon dung them cac script goc, co the cai:

```bash
pip install -r req.txt
```

Luu y: file `req.txt` goc co the khong tuong thich hoan toan voi Python 3.11/macOS ARM64 do phu thuoc cu.

## Chuan Bi Du Lieu

Runner moi doc truc tiep file ZIP, khong bat buoc giai nen toan bo dataset.

Archive can co cau truc:

```text
archive.zip
├── images/
│   ├── <item_id>_*.jpg
│   └── ...
└── masks/
    ├── <item_id>_*.png
    └── ...
```

Trong lan chay da bao cao, archive hop le gom 13.752 anh va 13.752 mask, kich thuoc 2.239.158.848 byte, SHA-256 bat dau bang `a7d59c2b`.

## Lenh Da Chay Duoc

Kiem tra du lieu:

```bash
python data_check.py --archive ../archive.zip --sample 500
```

Chay reproduction huong A:

```bash
python direction_a_repro.py \
  --archive ../archive.zip \
  --output-dir repro_outputs \
  --sample-items 4600 \
  --num-users 50 \
  --max-purchases 3 \
  --top-k 5 \
  --seed 42 \
  --backbone resnet50
```

Tao split co dinh:

```bash
python split_items.py --archive ../archive.zip --output-dir week3_splits --seed 42
```

Tao ground truth doc lap:

```bash
python make_ground_truth.py --split-dir week3_splits --output-dir week3_splits
```

## Split Va Ground Truth Tuan 3

Split co dinh da tao:

| Split | So item | Ty le |
|---|---:|---:|
| Train | 3.221 | 70,0% |
| Val | 689 | 15,0% |
| Test | 690 | 15,0% |
| Tong | 4.600 | 100% |

Ground truth doc lap:

| Hang muc | Ket qua |
|---|---|
| Phuong phap | Category-based binary relevance |
| Queries | 690 test items |
| Queries khong co relevant | 0 |
| Avg relevant / query | 1.829,5 |
| Min / Max relevant | 1 / 2.367 |
| Luu tai | `week3_splits/ground_truth.json` |

Ground truth nay khong dung popularity va khong dung hanh vi tong hop, nen tranh circular evaluation cho cac chi so Recall@K, NDCG@K, MRR o cac tuan tiep theo.

## Han Che Hien Tai

- Ket qua reproduction hien tai moi xac nhan tinh kha thi va pipeline, chua phai ket qua paper tai lap day du.
- Hanh vi nguoi dung trong pipeline van la tong hop, khong duoc coi la bang chung ca nhan hoa ngoai doi that.
- Dataset bi imbalance nang, 500 mau scan co 77,0% `Tees_Tanks`; can bao cao chi so theo category.
- Can chay them toi thieu 3 seed va danh gia tren split/ground truth co dinh truoc khi rut ket luan khoa hoc.

## Ket Luan Trang Thai

Huong A: **Go** cho giai doan trien khai tiep theo.

Ly do:

- Da chay duoc end-to-end tren dataset that.
- Da co log thoi gian, peak RAM, metric CSV, manifest va checksum.
- Da tao split/ground truth doc lap de tiep tuc danh gia nghiem tuc.
- Co cac loi repo goc duoc ghi nhan ro trong `../bao_cao_xac_thuc_6_buoc.md`.

Huong C (`../outfit-transformer`) duoc giu lam baseline doi chieu smoke. Huong B (NGNN) da dong vi moi truong Python 2/TensorFlow 1.5 va du lieu tien xu ly thieu, fake qua nhieu nen khong co gia tri kiem chung.

## Tai Lieu Lien Quan

- Bao cao tong hop: `../bao_cao_xac_thuc_6_buoc.md`
- Baseline OutfitTransformer: `../outfit-transformer/README.md`
- Split/ground truth: `week3_splits/`
