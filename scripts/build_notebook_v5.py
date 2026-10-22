"""Build V5: restore the observed V3 training profile; test one PT correction.

Uses immutable V3/V4 archives and writes only the active final_97.ipynb. Never
executes training. The resulting notebook is standalone on Colab.
"""
import hashlib
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "notebooks/archive/final_97_executed.ipynb"
V4 = ROOT / "notebooks/archive/final_97_v4_source.ipynb"
V3_SHA = "cea7a8332988234fcf2a232c3b25d9256964519cd2c7a2f90d11131a6d02e9a5"
V4_SHA = "99a5236478c1f3705a66cf1983de646042b0ef836dde444cb80911ba88363359"
assert hashlib.sha256(V3.read_bytes()).hexdigest() == V3_SHA
assert hashlib.sha256(V4.read_bytes()).hexdigest() == V4_SHA
baseline = nbformat.read(V3, as_version=4)
notebook = nbformat.read(V4, as_version=4)


def replace(index, old, new):
    source = notebook.cells[index].source
    assert source.count(old) == 1, (index, old)
    notebook.cells[index].source = source.replace(old, new)


notebook.cells[0].source = """# RMIT_SGS_Sigma69_C1 — V5, nền V3 và thử hiệu chỉnh Portuguese

File đang chạy: **`final_97.ipynb`**. Wolf Defender/ModernBERT được fine-tune toàn bộ,
nguồn Kaggle đã chọn: `aisuko/wolf-defender-prompt-injection/transformers/default/1`.

V3 có raw validation Macro F1 **0.972933**, tốt nhất ở epoch 4.
V4 người dùng báo **0.969274**, tốt nhất ở epoch 3. V4 đổi context, batch và cách
nạp config cùng lúc; các số này không xác định riêng nguyên nhân giảm điểm.

V5 trở lại cấu hình train và cách nạp V3: **256 token đầu, batch 8 × accumulation 4,
4 epoch, LR 3e-5, weighted CE power 0.5**. Thử thêm một hiệu chỉnh nhỏ cho các
dự đoán Portuguese BENIGN/JAILBREAK sát nhau. Chỉ bật nếu nested cross-fit qua
điều kiện; nếu không, giữ kết quả global ban đầu. **Chưa có điểm GPU/Kaggle V5.**

Upload notebook và chọn GPU mới, chạy từ đầu. Mục 3 nhận `rmit-hackathon-2026.zip`.
Chỉ còn `submission.csv` cũ thì cần train lại: CSV nhãn không chứa logits/validation.
Mặc định `USE_FULL_TRAIN=False` để xem selection trước. Khi đã xem kết quả và muốn
refit, bật tùy chọn này cho bước 11; không đổi các cấu hình khác giữa hai bước.

Sau mục 10 tải **`submission_selection.csv` và `selection_predictions.zip`**.
ZIP nhỏ giữ logits/ID/báo cáo để lần sau thử hậu xử lý mà không phải train lại.
`rmit_c1_artifacts.zip` cuối notebook mới chứa cả model. Mục tiêu 0.99 cần đo thực tế.
"""
notebook.cells[3].source = """## 2. Cấu hình

Giữ `DATA_MODE="upload"` và chọn ZIP dữ liệu. Dữ liệu đã có trên Colab/Drive thì
dùng `"path"` và sửa `DATA_DIR`; `OUTPUT_ROOT` trên Drive giúp giữ checkpoint.

Wolf mặc định khôi phục cấu hình V3: 256 token `head`, batch 8 × accumulation 4 = 32.
Tự chọn epoch có Macro F1 cao nhất; không ép epoch 3 (V3 tốt nhất ở epoch 4).
Giữ class-weight power 0.5. Tăng power cũng làm đổi trọng số Prompt Injection,
nên không gộp thay đổi loss vào thử nghiệm hiệu chỉnh Portuguese này.

`WOLF_LOADING_MODE="v3_legacy"` tái lập cách đọc config/tokenizer của V3 bằng thư viện
đã khóa; đây không phải bảo đảm tái lập đúng từng bit hoặc cùng score.
`"checkpoint_config"` là cách chuẩn hóa RoPE ở V4, dành cho thử nghiệm riêng.
`ENABLE_PT_PAIR_TUNING=False` tắt hiệu chỉnh mới để chạy đối chứng cùng cấu hình.
`USE_FULL_TRAIN=False` mặc định: xuất selection trước, không tự train thêm 14.000 dòng.
"""

# Restore all the known training choices together, including original global
# postprocessing grids. The experimental change is the guarded PT correction.
notebook.cells[4].source = baseline.cells[4].source
replace(4, 'USE_FULL_TRAIN = True                      # 90/10 selection -> fresh refit tren 14k',
        'USE_FULL_TRAIN = False                     # Xem selection truoc; refit la tuy chon')
replace(4, 'RUN_NAME = f"{MODEL_PRESET}_v3"',
        'NOTEBOOK_VERSION = "v5_v3baseline_pt"\n'
        'WOLF_LOADING_MODE = "v3_legacy"            # "checkpoint_config" la thu nghiem rieng\n'
        'RUN_NAME = f"{MODEL_PRESET}_{NOTEBOOK_VERSION}"')
replace(4, 'MIN_POSTPROCESS_GAIN = 0.002',
        'MIN_POSTPROCESS_GAIN = 0.002\n'
        'ENABLE_PT_PAIR_TUNING = True\n'
        'PT_PAIR_BIAS_VALUES = (-0.30, -0.15, 0.0, 0.15, 0.30)\n'
        'PT_PAIR_MAX_MARGIN = 0.75\n'
        'PT_PAIR_MIN_GAIN = 0.002')
replace(4, 'print("Preset:", MODEL_PRESET, "| full refit:", USE_FULL_TRAIN)',
        'print("Notebook:", NOTEBOOK_VERSION, "| preset:", MODEL_PRESET, "| full refit:", USE_FULL_TRAIN)\n'
        'print("Loading:", WOLF_LOADING_MODE, "| input:", MAX_LENGTH, LONG_TEXT_MODE)')

notebook.cells[13].source = """## 5. Nạp model theo cấu hình đã chọn

Mặc định `v3_legacy` giữ cách AutoConfig 4.57.6 đọc config nguồn như V3 và cùng
cách sửa metadata tokenizer của V3. Chỉ sao chép các file config/tokenizer nhỏ,
không sao chép weights. Các giá trị RoPE thực sự được ghi vào `model_provenance.json`.
`checkpoint_config` giữ cách ánh xạ metadata mới ở V4 cho thử nghiệm riêng.

256 token `head` là baseline quan sát; báo cáo `token_lengths.csv` vẫn lưu phần
bị cắt theo ngôn ngữ/lớp. Chưa kết luận context dài hơn luôn tốt hoặc xấu hơn.
"""
notebook.cells[14].source = (ROOT / "scripts/_v5_tokenizer_cell.py").read_text(encoding="utf-8")
notebook.cells[20].source = baseline.cells[20].source
notebook.cells[26].source = """### Hậu xử lý global và thử hiệu chỉnh Portuguese

Giữ quy trình global của V3: blend/bias chỉ bật nếu cross-fit tăng ít nhất 0.002.
Sau đó thử bias nhỏ cho Portuguese ở những dự đoán Benign/Jailbreak sát nhau.
Các fold ngoài đánh giá trên những dòng không tham gia chọn global policy hoặc
bias của chính fold đó. Chỉ dùng hiệu chỉnh mới nếu gain overall ≥ 0.002, gain
Portuguese dương, ít nhất 2/3 fold cải thiện và kiểm tra cuối không làm giảm điểm.
Nếu không đạt, giữ nguyên kết quả global. Bước PT không đổi dự đoán PI/ngôn ngữ khác.

Checkpoint vẫn đã được chọn trên validation này, nên nested cross-fit hậu xử lý
không biến các số trên thành đánh giá model độc lập. Cả điểm raw, global và PT
được lưu riêng; điểm fit trên toàn validation không phải dự báo điểm Kaggle.
"""
notebook.cells[27].source = (
    baseline.cells[27].source.rstrip() + "\n\n"
    + (ROOT / "scripts/_v5_postprocess_cell.py").read_text(encoding="utf-8")
)
notebook.cells[28].source = """### Xuất CSV selection và gói dự đoán nhỏ

`submission_selection.csv` là kết quả đã chọn sau các điều kiện kiểm tra.
`submission_selection_global.csv` giữ kết quả trước bước PT;
`submission_selection_raw.csv` giữ dự đoán neural thô trong artifacts.
Nếu bước PT không đạt điều kiện, hai file selection và selection_global giống nhau.

Tải **`/content/submission_selection.csv`** để nộp và giữ thêm
**`/content/selection_predictions.zip`** để thử hậu xử lý lần sau mà không train lại.
ZIP này không chứa weights. `submission.csv` hiện có cùng nội dung selection;
nếu bật refit thì bước sau mới cập nhật nó bằng dự đoán model refit.
"""
replace(29, '# Luu selection CSV truoc full refit de co ket qua neu runtime bi ngat.',
        '# Luu selection CSV truoc full refit de co ket qua neu runtime bi ngat.\n'
        'make_submission(global_test_pred, ARTIFACT_DIR / "submission_selection_global.csv")\n'
        'shutil.copy2(ARTIFACT_DIR / "submission_selection_global.csv", "/content/submission_selection_global.csv")')
notebook.cells[29].source += '\n' + (ROOT / "scripts/_v5_selection_bundle.py").read_text(encoding="utf-8")

notebook.cells[30].source = """## 11. Refit tùy chọn trên toàn bộ dữ liệu có nhãn

Mặc định **`USE_FULL_TRAIN=False`**: bước này chỉ lưu metadata của selection và không
train thêm. Muốn refit sau khi xem kết quả, đặt `USE_FULL_TRAIN=True` trong một cell
ngay trước bước này, giữ nguyên các cấu hình đã dùng cho selection.

Refit khởi tạo mới từ pretrained, train 14.000 dòng trong số epoch checkpoint tốt
nhất đã chọn. Không đánh giá trên holdout cũ đã học, không chọn lại bias.
Hiệu chỉnh global/PT được giữ nguyên và áp dụng theo đúng language/ID của test.
Điểm raw/cross-fit trước đó thuộc selection; refit không bảo đảm tăng điểm test.
"""
replace(31, '    if FROZEN_POSTPROCESS_CONFIG["enabled"]:\n'
            '        final_test_scores = postprocess_log_scores(\n'
            '            test_logits, test_lexical_scores, FROZEN_POSTPROCESS_CONFIG["parameters"],\n'
            '        )\n'
            '    else:\n'
            '        final_test_scores = log_softmax_np(test_logits, 1.0)',
        '    final_test_scores = apply_selected_postprocess(\n'
        '        test_logits, test_lexical_scores, FROZEN_POSTPROCESS_CONFIG,\n'
        '        test_df["language"].astype(str).to_numpy(),\n'
        '    )')
replace(34, '    "model_provenance": MODEL_PROVENANCE,',
        '    "model_provenance": MODEL_PROVENANCE,\n'
        '    "wolf_loading_mode": WOLF_LOADING_MODE if MODEL_PRESET == "wolf" else None,\n'
        '    "v4_best_raw_validation_user_reported": 0.969274,')
notebook.cells[35].source = """## 13. Tải kết quả

Mở Files bên trái Colab → Refresh. Tải `submission_selection.csv` và
`selection_predictions.zip` để giữ kết quả selection cùng logits/ID.
`submission.csv` là file của model đang dùng (selection nếu không refit, final nếu
đã refit). `rmit_c1_artifacts.zip` chứa model và báo cáo nên lớn hơn nhiều.
Các cell dưới chỉ in đường dẫn; đặt cờ `True` nếu muốn thử tải tự động.
"""
notebook.cells[36].source += '\nprint("Giu them ZIP nho de tinh chinh lan sau:", SELECTION_PREDICTIONS_PATH)\n'
notebook.cells[38].source = """## Nguồn và cách đọc kết quả

- [Cuộc thi](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview).
- [Wolf trên Kaggle](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/transformers/default/1).
- [AutoTokenizer 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/auto/tokenization_auto.py).
- [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

`selection_comparison.json` tách raw Macro F1 hiện tại, mốc V3/V4 và gain hiệu chỉnh.
`pt_pair_tuning.json` lưu điều kiện bật/tắt cùng các fold. Cross-fit chỉ đánh giá
phần hậu xử lý; model vẫn từng dùng holdout này để chọn checkpoint.

Nạp lại inference phải dùng `apply_selected_postprocess` với config đầy đủ và
language đúng thứ tự ID. Trường `enabled` ở gốc config chỉ mô tả global policy;
hiệu chỉnh PT nằm trong `portuguese_pair`. Refit áp dụng cả hai mà không fit lại.
`selection_predictions.zip` chỉ giữ dự đoán selection; model nằm trong ZIP artifacts lớn.
Bản V3 đã chạy và source V4 được giữ riêng trong `notebooks/archive/`.
"""

for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []
        compile(cell.source, f"final_97.ipynb:cell_{index}", "exec")
nbformat.validate(notebook)
nbformat.write(notebook, ROOT / "final_97.ipynb")
print("Wrote V5 final_97.ipynb; V3 and V4 archives unchanged. No training executed.")
