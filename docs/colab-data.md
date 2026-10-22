# Tải và kết nối dữ liệu trên Google Colab

Repository đã có đủ ba file dữ liệu trong thư mục `rmit-hackathon-2026/`:

| File | Số dòng | Vai trò |
|---|---:|---|
| `train.csv` | 14.000 | Huấn luyện và tạo validation split |
| `test.csv` | 6.000 | Tạo dự đoán để nộp Kaggle |
| `sample_submission.csv` | 6.000 | Giữ đúng ID, thứ tự và schema submission |

File [rmit-hackathon-2026.zip](../rmit-hackathon-2026.zip) ở thư mục gốc chứa đúng ba CSV này. Notebook hiện tại là **[final_97.ipynb](../final_97.ipynb)** (v5), hỗ trợ upload trực tiếp ZIP. Các notebook trong `notebooks/archive/` là bản lịch sử.

## Cách 1: upload ZIP trực tiếp lên Colab

1. Tải hai file về máy:
   - `final_97.ipynb` ở thư mục gốc;
   - `rmit-hackathon-2026.zip`.
2. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook** và chọn file `.ipynb`.
3. Chọn **Runtime → Change runtime type → T4 GPU**.
4. Trong cell cấu hình, giữ nguyên:

```python
DATA_MODE = "upload"
DATA_DIR = "/content/rmit-hackathon-2026"
MOUNT_DRIVE = False
MODEL_PRESET = "wolf"
WOLF_LOCAL_PATH = ""  # Tải đúng model Kaggle bằng KaggleHub khi tới cell model
WOLF_LOADING_MODE = "v3_legacy"
USE_FULL_TRAIN = False  # Xem kết quả selection trước, chưa refit 14.000 dòng
```

5. Chạy các cell từ đầu trong runtime mới. Khi cell **Upload dữ liệu** mở hộp chọn file, chọn `rmit-hackathon-2026.zip`.
6. Notebook tự giải nén ba CSV vào `/content/rmit-hackathon-2026` và kiểm tra schema, số dòng, ID cùng nhãn.

Dữ liệu trong `/content` sẽ mất khi Colab runtime bị xóa hoặc reset. Tải CSV và artifacts của lần chạy cũ về trước khi đổi runtime. V5 dùng cấu hình v3: 256 token đầu, train batch 8/accumulation 4, tối đa 4 epoch và tự chọn checkpoint tốt nhất. Mặc định chỉ chạy selection vì `USE_FULL_TRAIN=False`; chưa có kết quả GPU v5.

`submission_selection.csv` có dự đoán sau bước hiệu chỉnh được chấp nhận; `submission_selection_global.csv` giữ kết quả trước bước hiệu chỉnh BENIGN/JAILBREAK tiếng Bồ Đào Nha. Với mặc định tắt refit, `submission.csv` giống kết quả selection. Nếu muốn refit, đặt `USE_FULL_TRAIN=True` từ đầu; sau refit, `submission.csv` dùng model mới train trên 14.000 dòng còn file selection được giữ nguyên. Nộp CSV, không nộp ZIP artifacts. Xem `pt_pair_tuning.json` để biết bước hiệu chỉnh có được bật và vì sao.

Hiệu chỉnh v5 diễn ra sau training: bảng Macro F1 từng epoch vẫn đo model thô và không bắt buộc vượt v3. Đánh giá thử nghiệm bằng gain nested cross-fit và điều kiện bật/tắt trong `pt_pair_tuning.json`, sau đó xem CSV selection. Không suy ra mức tăng Kaggle chắc chắn từ điểm validation.

Hiện lần chạy cũ chỉ còn `submission.csv`, không còn logits hoặc ZIP artifacts. Các nhãn trong CSV không đủ để khôi phục score của từng lớp hoặc đánh giá bias trên validation, nên lần này cần train v5 lại từ cấu hình v3.

Sau mục 10, mở bảng **Files** và tải cả CSV lẫn **`/content/selection_predictions.zip`**. Đây là gói nhỏ chứa logits validation/test của selection, ID tương ứng, lexical scores nếu dùng, metadata lần chạy/cách nạp model, label mapping, tham số hậu xử lý và CSV. Gói này không chứa weights, giúp các lần chỉnh hậu xử lý tiếp theo có dữ liệu để tái sử dụng mà không train GPU lại. Gói lớn **`/content/rmit_c1_artifacts.zip`** vẫn dùng lưu model và báo cáo; không dùng hai ZIP này làm prediction file để submit Kaggle.

## Model Wolf được tải riêng với dữ liệu

ZIP dữ liệu không chứa model. Với cấu hình mặc định `MODEL_PRESET="wolf"` và `WOLF_LOCAL_PATH=""`, cell model sẽ gọi KaggleHub để tải đúng `aisuko/wolf-defender-prompt-injection/transformers/default/1`, cùng model/version trong notebook Melbourne. Cần Internet và dung lượng lưu model; nếu Kaggle yêu cầu đăng nhập thì dùng tài khoản của bạn trong Colab, không ghi credentials vào notebook hoặc Git. V3 đã chạy Wolf trên T4; v5 cần được chạy lại từ đầu để kiểm chứng kết quả mới. Giữ `WOLF_LOADING_MODE="v3_legacy"` để so sánh với baseline; `checkpoint_config` là phép thử cách nạp khác, chưa có bằng chứng tốt hơn.

Nếu đã có weights trên Drive hoặc đã gắn model trên Kaggle, đặt đường dẫn **thư mục model chứa `config.json`** trước khi chạy cell cấu hình:

```python
MODEL_PRESET = "wolf"
WOLF_LOCAL_PATH = "/content/drive/MyDrive/RMIT_SGS_Sigma69_C1/models/wolf"
```

Ví dụ trên chỉ có tác dụng sau khi bạn đã đặt đúng model ở thư mục đó và mount Drive. Trên Kaggle, lấy đường dẫn thực từ panel Input; notebook sẽ kiểm tra model type `modernbert`. Model này đã nằm trong [danh sách BTC được lưu](../demo/competition-challenge.json).

Nếu muốn thử lại DistilBERT QLoRA, đổi `MODEL_PRESET="distilbert"` rồi chạy lại notebook từ đầu. Không dùng logits/model còn sót từ preset Wolf. CSV và ZIP dữ liệu dùng chung cho cả hai preset.

## Cách 2: dùng Google Drive để dữ liệu tồn tại lâu dài

Trên Google Drive, tạo thư mục:

```text
MyDrive/RMIT_SGS_Sigma69_C1/data/
```

Upload trực tiếp ba file CSV vào thư mục `data`. Sau đó sửa cell cấu hình của notebook:

```python
DATA_MODE = "path"
MOUNT_DRIVE = True
DATA_DIR = "/content/drive/MyDrive/RMIT_SGS_Sigma69_C1/data"
OUTPUT_ROOT = "/content/drive/MyDrive/RMIT_SGS_Sigma69_C1/runs"
```

Khi chạy cell cấu hình, Colab yêu cầu cấp quyền Drive. Sau khi mount thành công, cell dữ liệu đọc ba CSV từ `DATA_DIR`; không hiện hộp upload.

Nếu chỉ lưu ZIP trên Drive, giải nén một lần trước khi chạy cell đọc dữ liệu:

```python
from pathlib import Path
from zipfile import ZipFile

zip_path = Path("/content/drive/MyDrive/RMIT_SGS_Sigma69_C1/rmit-hackathon-2026.zip")
data_dir = Path("/content/drive/MyDrive/RMIT_SGS_Sigma69_C1/data")
data_dir.mkdir(parents=True, exist_ok=True)

with ZipFile(zip_path) as archive:
    archive.extractall(data_dir)
```

Sau đó giữ `DATA_MODE = "path"` và `DATA_DIR` trỏ tới thư mục `data`.

## Kiểm tra kết nối dữ liệu

Sau khi cell dữ liệu chạy thành công, Colab phải hiển thị:

```text
Train: 14000 | Test: 6000
```

Có thể kiểm tra thủ công bằng cell sau:

```python
from pathlib import Path
import pandas as pd

data_dir = Path(DATA_DIR)
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample = pd.read_csv(data_dir / "sample_submission.csv")

assert train.shape == (14000, 4)
assert test.shape == (6000, 3)
assert sample.shape == (6000, 2)
assert list(train.columns) == ["id", "text", "language", "label"]
assert list(test.columns) == ["id", "text", "language"]
assert list(sample.columns) == ["id", "label"]
print("Data connected successfully")
```

Không dùng `sample_submission.csv` làm nhãn test. Cột `label` trong file này chỉ là placeholder để minh họa định dạng nộp bài.
