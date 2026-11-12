from copy import copy
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "week1"
SOURCE = RESULT_DIR / "khao_sat_xu_huong_co_nguon.xlsx"
OUTPUT = RESULT_DIR / "khao_sat_xu_huong_da_xac_thuc.xlsx"

NAVY = "123B6D"
TEAL = "13A8A8"
LIGHT_TEAL = "DDF5F5"
ORANGE = "FF7A21"
LIGHT_ORANGE = "FFF0E2"
GREEN = "D9EAD3"
YELLOW = "FFF2CC"
RED = "F4CCCC"
WHITE = "FFFFFF"
GRID = Side(style="thin", color="D9E2F3")


def style_sheet(ws, widths):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(color=WHITE, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(left=GRID, right=GRID, top=GRID, bottom=GRID)


def status_fill(cell):
    text = str(cell.value or "").lower()
    color = GREEN if "đạt" in text and "chưa" not in text else YELLOW
    if "chưa đạt" in text or "thất bại" in text:
        color = RED
    cell.fill = PatternFill("solid", fgColor=color)
    cell.font = Font(bold=True)


wb = load_workbook(SOURCE)
original = wb[wb.sheetnames[0]]
original.title = "Khao_sat_goc"

if "Xac_thuc_6_buoc" in wb.sheetnames:
    del wb["Xac_thuc_6_buoc"]
ws = wb.create_sheet("Xac_thuc_6_buoc")
ws.append(["Bước", "Kết quả xác thực", "Trạng thái", "Hành động/Minh chứng tiếp theo"])
rows = [
    ("1. Xác định phạm vi", "Bài toán và đầu ra rõ; 1 người, pretrained CNN. Lịch ảnh là 12 tuần nhưng Excel ghi 13 tuần (26/09–25/12/2026).", "Đạt có điều kiện", "Chọn 12 tuần hoặc đổi tên kế hoạch thành 13 tuần."),
    ("2. Khảo sát bằng công cụ", "A gần phạm vi nhất; OutfitTransformer và CR-BPR là hai hướng đối chiếu. CFALR/TATTOO chưa có repo chính thức tìm thấy.", "Đạt sau hiệu chỉnh", "Không dùng cụm 'hướng duy nhất'; ghi rõ tiêu chí gần phạm vi + khả thi."),
    ("3. Đọc và đối chiếu", "Paper/repo có thật nhưng code lệch mô tả: gender filter và category weight bị comment; metric gender ghi sai trường; README khác code; paper mâu thuẫn MAE.", "Đạt một phần", "Xem báo cáo Markdown, mục Bước 3."),
    ("4. Đặt câu hỏi", "Đã chốt câu hỏi tái lập: visual + popularity có cải thiện Top-K sau khi sửa lỗi và dùng ground truth độc lập hay không?", "Đạt", "Đo Recall/NDCG/MRR hoặc mAP@K, thêm latency/RAM/VRAM."),
    ("5. Chạy thử nhỏ", "Đã chạy end-to-end runner sửa lỗi trên toàn bộ 4.600 item ID từ archive Kaggle, ResNet-50 pretrained, 50 users tổng hợp, Top-5. Sinh đủ feature/CSV/manifest/checksum; 216,49 giây; peak RSS ~767 MiB.", "Đạt", "Bổ sung nhiều seed và split độc lập trước khi kết luận khoa học."),
    ("6. Chốt hướng", "GO cho Hướng A dưới dạng đề tài tái lập và đánh giá độ tin cậy. Không tuyên bố repo gốc chạy nguyên trạng hoặc đã tái lập full paper.", "Đạt có điều kiện", "Tiếp tục sửa metric, metadata và đánh giá full-dataset."),
]
for row in rows:
    ws.append(row)
style_sheet(ws, [24, 74, 26, 64])
for row in range(2, ws.max_row + 1):
    status_fill(ws.cell(row, 3))

if "Doi_chieu_nguon" in wb.sheetnames:
    del wb["Doi_chieu_nguon"]
ws = wb.create_sheet("Doi_chieu_nguon")
ws.append(["Hướng", "Năm", "Paper", "Repo/Dataset", "Kết luận"])
sources = [
    ("A. Trend-Aware Fashion Recommendation", 2025, "https://arxiv.org/abs/2506.07773", "https://github.com/meddjilani/FashionRecommender", "Đã chạy end-to-end bằng runner sửa lỗi trên 4.600 item ID; chọn có điều kiện."),
    ("B. NGNN", 2019, "https://dl.acm.org/doi/10.1145/3308558.3313444", "https://github.com/CRIPAC-DIG/NGNN", "SKIP: TensorFlow 1.5 không có bản cho Python 3.11/macOS ARM64; repo còn thiếu feature/pickle đầu vào."),
    ("C. OutfitTransformer", 2022, "https://arxiv.org/abs/2204.04812", "https://github.com/essalim/outfit-transformer", "Đã chạy smoke end-to-end 1 epoch trên fixture 24 ảnh; không dùng metric này làm kết quả khoa học."),
    ("F. FitCheck.AI", 2025, "Không có paper học thuật", "https://github.com/daniel-mehta/FitCheck.AI", "Chỉ tham khảo demo VLM/Streamlit."),
    ("CFALR", 2026, "https://arxiv.org/abs/2606.13001", "Chưa thấy repo chính thức", "Mới nhưng chưa thỏa tiêu chí code kiểm chứng."),
    ("TATTOO", 2025, "https://arxiv.org/abs/2509.23242", "Chưa thấy repo chính thức", "Mới nhưng chưa thỏa tiêu chí code kiểm chứng."),
    ("CR-BPR", 2026, "https://doi.org/10.1016/j.asoc.2026.115069", "https://github.com/asaander719/CR-BPR", "Mới + code; lệch sang complementary clothing recommendation."),
    ("Fashion-MNIST (smoke data)", 2017, "https://arxiv.org/abs/1708.07747", "https://github.com/zalandoresearch/fashion-mnist", "MIT; 40 cặp ảnh/mask đã đóng gói chỉ để kiểm tra kỹ thuật."),
]
for row in sources:
    ws.append(row)
style_sheet(ws, [36, 10, 48, 52, 54])
for row in range(2, ws.max_row + 1):
    for col in (3, 4):
        cell = ws.cell(row, col)
        if str(cell.value).startswith("http"):
            cell.hyperlink = cell.value
            cell.style = "Hyperlink"

if "Nhat_ky_chay" in wb.sheetnames:
    del wb["Nhat_ky_chay"]
ws = wb.create_sheet("Nhat_ky_chay")
ws.append(["Kiểm tra", "Lệnh/Bằng chứng", "Kết quả", "Đánh giá"])
logs = [
    ("Repo", "git clone --depth 1 https://github.com/meddjilani/FashionRecommender.git", "Thành công; commit ddd62684e9b51d9fbe5a48d3bab9c1e27be832db", "Đạt"),
    ("Cú pháp", "python3 -m py_compile FashionRecommender/*.py", "Exit 0", "Đạt"),
    ("Entry point", "python3 final_recommandation.py --help", "ModuleNotFoundError: No module named 'torchvision'", "Thất bại"),
    ("Dependency resolver", "python3 -m pip install --dry-run -r req.txt", "Không có tensorflow==2.8.0 phù hợp Python 3.11", "Thất bại"),
    ("Dataset smoke", "prepare_smoke_dataset.py", "40 ảnh + 40 mask; manifest; MIT; ZIP SHA-256 e27dfea2…", "Đạt cho smoke test"),
    ("Dataset nghiên cứu", "archive.zip do người dùng tải từ hserdaraltan", "13.752 cặp ảnh/mask; SHA-256 a7d59c2b…", "Đạt"),
    ("End-to-end", "direction_a_repro.py; 4.600 item ID; ResNet-50 pretrained", "Thành công; CSV + manifest + checksum; 216,49 giây; peak RSS ~767 MiB", "Đạt"),
    ("B / NGNN", "pip install tensorflow==1.5.0; python3 main_score.py; py_compile Config.py", "Không có distribution TF 1.5 cho Python 3.11/macOS ARM64; code Python 2; thiếu feature JSON, pickle và dữ liệu *_1000", "SKIP"),
    ("C / OutfitTransformer", "Cài timm/labml-nn/faiss-cpu/sentence-transformers; tạo fixture 24 ảnh; train_cp.py 1 epoch", "Thành công trên MPS; validation AUC 0,73/accuracy 62,5%; test AUC 0,6667/accuracy 62,5%; có checkpoint + report", "Đạt smoke"),
]
for row in logs:
    ws.append(row)
style_sheet(ws, [24, 68, 70, 18])
for row in range(2, ws.max_row + 1):
    status_fill(ws.cell(row, 4))

if "Kiem_tra_B_C" in wb.sheetnames:
    del wb["Kiem_tra_B_C"]
ws = wb.create_sheet("Kiem_tra_B_C")
ws.append(["Hướng", "Repo/commit", "Kiểm tra đã chạy", "Kết quả", "Trạng thái", "Điều kiện để chạy tiếp"])
checks = [
    (
        "B. NGNN",
        "CRIPAC-DIG/NGNN @ 554585c42d50b4f4b5c26da33269e73f11c2e3cc",
        "Cài tensorflow==1.5.0; entry point Python 3; biên dịch cú pháp; kiểm kê dữ liệu",
        "Pip xác nhận không có TF 1.5 cho nền tảng hiện tại; Config.py là Python 2. Repo còn thiếu fill_in_blank_1000_from_test_score.pkl, feature JSON và dữ liệu *_1000; sample ảnh chỉ có 2 outfit.",
        "SKIP",
        "Cần máy/container x86 cũ chạy Python 2.7 + TensorFlow 1.5 và gói dữ liệu tiền xử lý đầy đủ của tác giả.",
    ),
    (
        "C. OutfitTransformer",
        "essalim/outfit-transformer @ 90f26efd349f18de81d656a00fe9a888c42ab098",
        "Cài đủ dependency; tạo fixture schema Polyvore từ sample NGNN; train/valid/test 1 epoch",
        "Chạy end-to-end thành công trên MPS với ResNet-18 pretrained; 24 ảnh, 8 outfit, 16 compatibility rows/split. Test AUC 0,6667; accuracy 62,5%; sinh checkpoint/report.",
        "Đạt smoke",
        "Muốn so sánh khoa học phải chạy lại trên Polyvore thật, split chính thức và nhiều seed; không dùng metric fixture tổng hợp.",
    ),
]
for row in checks:
    ws.append(row)
style_sheet(ws, [24, 58, 48, 82, 16, 72])
for row in range(2, ws.max_row + 1):
    status_fill(ws.cell(row, 5))

if "De_cuong_12_tuan" in wb.sheetnames:
    del wb["De_cuong_12_tuan"]
ws = wb.create_sheet("De_cuong_12_tuan")
ws.append(["Tuần", "Công việc", "Minh chứng nghiệm thu"])
plan = [
    (1, "Chốt phạm vi, license, nguồn dữ liệu", "Manifest + biên bản phạm vi"),
    (2, "Tái lập môi trường, smoke test nhỏ; quyết định Go/No-Go", "Lockfile + log chạy"),
    (3, "Chuẩn hóa dữ liệu, split và ground truth", "Script kiểm tra dữ liệu"),
    (4, "Baseline popularity-only và visual-only", "Bảng baseline"),
    (5, "Sửa masking và cache feature", "Tests + cache"),
    (6, "Sửa scoring và metric gender/category", "Test chống hồi quy"),
    (7, "Ablation masking/popularity", "CSV kết quả"),
    (8, "So sánh backbone nếu đủ tài nguyên", "Bảng chất lượng/thời gian/bộ nhớ"),
    (9, "Chạy ít nhất 3 seeds, phân tích sai số", "Mean ± std"),
    (10, "Xây API/demo web tối thiểu", "Demo Top-K"),
    (11, "Viết báo cáo và threat to validity", "Bản thảo"),
    (12, "Đóng gói tái lập và phương án dự phòng", "README + lệnh chạy + video"),
]
for row in plan:
    ws.append(row)
style_sheet(ws, [10, 66, 54])

for sheet in wb.worksheets[1:]:
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 32

wb.save(OUTPUT)
print(OUTPUT)
