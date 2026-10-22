"""Build the standalone V4 notebook from the preserved, executed Wolf run.

This edits only final_97.ipynb. It never trains, edits archives, or recreates the
old root notebook copies. Review the generated notebook before running Colab.
"""
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "notebooks/archive/final_97_executed.ipynb"
DESTINATION = ROOT / "final_97.ipynb"
BASELINE_SHA256 = "cea7a8332988234fcf2a232c3b25d9256964519cd2c7a2f90d11131a6d02e9a5"
assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256
notebook = nbformat.read(BASELINE, as_version=4)


def replace(index, old, new):
    source = notebook.cells[index].source
    assert source.count(old) == 1, (index, old)
    notebook.cells[index].source = source.replace(old, new)


notebook.cells[0].source = """# RMIT_SGS_Sigma69_C1 — V4 Wolf 512, nâng cấp từ bản đạt 0.97

File đang chạy: **`final_97.ipynb`**. Backbone duy nhất: Wolf Defender/ModernBERT,
fine-tune toàn bộ model BTC đã duyệt, Kaggle version **`default/1`**.

Bản V3 đã chạy có raw validation Macro F1 **0.9729325**, sai **35/1.400** mẫu;
27 lỗi là Benign/Jailbreak tiếng Bồ Đào Nha. Public ~0.97 là kết quả người dùng báo.
V4 thử **512 token, giữ đầu + cuối** thay vì chỉ 256 token đầu; giữ loss và lịch train
đã cho kết quả tốt. Đồng thời sửa tương thích tokenizer/config Wolf với Transformers.
**0.99 là mục tiêu; chưa có kết quả GPU hoặc điểm Kaggle cho V4.**

Colab: tạo **runtime GPU mới**, upload notebook này, chạy từ đầu; đến mục 3 chọn
`rmit-hackathon-2026.zip` (train/test/sample_submission). Không cần upload scripts.
T4 dùng batch 4, accumulation 8. Đầu vào dài hơn sẽ tốn thời gian hơn V3.
`USE_FULL_TRAIN=True`: chọn checkpoint rồi refit đủ 14.000 dòng.
Nếu chỉ cần kết quả selection sớm, đặt `USE_FULL_TRAIN=False` **trước khi chạy**.
CSV ở mục 10 đã hợp lệ để submit; ZIP chỉ lưu model/báo cáo, không phải prediction file.
"""

notebook.cells[3].source = """## 2. Cấu hình

Giữ `DATA_MODE="upload"` để chọn ZIP/CSV. Nếu dữ liệu đã có trên Colab, chọn
`"path"` và sửa `DATA_DIR`. Dùng Drive: bật `MOUNT_DRIVE` và đặt `DATA_DIR` hoặc
`OUTPUT_ROOT` trong thư mục Drive.

V4 mặc định Wolf, 512 token, `head_tail`, batch 4 × accumulation 8 = 32.
Hết VRAM khi train: giảm batch 4 → 2, tăng accumulation 8 → 16 rồi chạy lại từ đầu.
Hết VRAM khi predict: giảm `EVAL_BATCH_SIZE` từ 8 xuống 4.
`USE_FULL_TRAIN=False` chỉ chạy selection, vẫn xuất đủ 6.000 dự đoán để submit.
"""

replace(4, 'TRAIN_BATCH_SIZE = 8                       # OOM: giam 4, tang GRAD_ACCUM_STEPS len 8',
        'TRAIN_BATCH_SIZE = 4                       # T4/512 tokens; OOM: 2 va accum=16')
replace(4, 'EVAL_BATCH_SIZE = 16', 'EVAL_BATCH_SIZE = 8                    # OOM luc predict: giam xuong 4')
replace(4, 'GRAD_ACCUM_STEPS = 4', 'GRAD_ACCUM_STEPS = 8                    # Giu effective batch = 32')
replace(4, 'MAX_LENGTH, LONG_TEXT_MODE = 256, "head"', 'MAX_LENGTH, LONG_TEXT_MODE = 512, "head_tail"')
replace(4, 'RUN_NAME = f"{MODEL_PRESET}_v3"',
        'NOTEBOOK_VERSION = "v4_wolf512"\nRUN_NAME = f"{MODEL_PRESET}_{NOTEBOOK_VERSION}"')
replace(4, '# Giu cac moc bias nho V2 va them range rong Melbourne; KHONG copy bias da fit cua ho.',
        '# Wolf da manh: thu trong so lexical nho va bias toan cuc hep; gate cross-fit van bat.')
replace(4, '    "jailbreak_bias_values": [-1., -.75, -.5, -.3, -.15, 0., .15, .3, .5, .75, 1.],\n'
           '    "injection_bias_values": [-1., -.75, -.5, -.3, -.15, 0., .15, .3, .5, .75, 1., 1.5, 2.],',
        '    "lexical_weights": [0., .05, .10, .20, .35, .50],\n'
        '    "jailbreak_bias_values": [-.5, -.3, -.15, 0., .15, .3, .5],\n'
        '    "injection_bias_values": [-.5, -.3, -.15, 0., .15, .3, .5],')
replace(4, 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
        'if MODEL_PRESET == "distilbert":\n'
        '    POSTPROCESS_FIT_KWARGS.update(\n'
        '        lexical_weights=[0., .25, .50, .75, .90],\n'
        '        jailbreak_bias_values=[-1., -.75, -.5, -.3, -.15, 0., .15, .3, .5, .75, 1.],\n'
        '        injection_bias_values=[-1., -.75, -.5, -.3, -.15, 0., .15, .3, .5, .75, 1., 1.5, 2.],\n'
        '    )\n'
        'os.environ["CUDA_VISIBLE_DEVICES"] = "0"')
replace(4, 'print("Preset:", MODEL_PRESET, "| full refit:", USE_FULL_TRAIN)',
        'print("Notebook:", NOTEBOOK_VERSION, "| preset:", MODEL_PRESET, "| full refit:", USE_FULL_TRAIN)\n'
        'print("Input:", MAX_LENGTH, "tokens |", LONG_TEXT_MODE)')

notebook.cells[13].source = """## 5. Nạp đúng Wolf/tokenizer và giữ cả đầu + cuối văn bản

Đọc config từ đúng model Kaggle đã chọn. Nếu config dùng định dạng Transformers 5,
ánh xạ RoPE sang runtime 4.57.6; tokenizer dùng `tokenizer.json` thật và giữ special tokens.
Chỉ chuẩn hóa các file tokenizer nhỏ, không sao chép toàn bộ weights.

512 token giúp giảm cắt văn bản; nếu vẫn dài hơn, giữ nửa đầu và nửa cuối. Báo cáo
`token_lengths.csv` ghi số mẫu bị cắt theo ngôn ngữ/lớp. Không đổi cách token hóa
giữa selection, validation, test và refit.
"""
notebook.cells[14].source = (ROOT / "scripts/_v4_tokenizer_cell.py").read_text(encoding="utf-8")
notebook.cells[23].source = (ROOT / "scripts/_v4_validation_cell.py").read_text(encoding="utf-8")
replace(20, 'Hết VRAM: giảm batch 8 → 4 và tăng accumulation 4 → 8, giữ effective batch 32.',
        'Hết VRAM: giảm batch 4 → 2 và tăng accumulation 8 → 16, giữ effective batch 32.')
notebook.cells[26].source = """### Chọn hậu xử lý bằng validation rồi xuất submission selection

Thử đóng góp nhỏ của TF-IDF/SVM và bias toàn cục. Chỉ dùng nếu Macro F1 cross-fit
tăng ít nhất `0.002`; không đủ thì giữ dự đoán Wolf thô. Không tune theo test hoặc
điểm public. Điểm fit bias trên toàn validation chỉ là chẩn đoán trong mẫu.
Validation đã dùng chọn checkpoint, nên cả điểm raw và cross-fit vẫn có thiên lệch chọn model.
"""
notebook.cells[28].source += """

**Có thể tải `/content/submission_selection.csv` ngay sau cell này để submit.**
Không cần chờ ZIP hoặc refit. Nếu chạy tiếp refit, `/content/submission.csv` sẽ được
cập nhật bằng model cuối; tên `selection` luôn giữ dự đoán từ model 90/10.
"""
replace(29, 'final_errors = raw_errors.copy()\n'
            'final_errors["predicted_label"] = [ID2LABEL[int(i)] for i in final_valid_pred]',
        'final_errors = build_prediction_audit(\n'
        '    valid_df, valid_true, final_valid_scores, valid_dataset.raw_lengths,\n'
        '    budget, LABELS, stage="final",\n'
        ')\n'
        'final_groups, final_lengths = summarize_prediction_audit(final_errors, stage="final")\n'
        'final_groups.to_csv(ARTIFACT_DIR / "validation_errors_by_group_final.csv", index=False)\n'
        'final_lengths.to_csv(ARTIFACT_DIR / "validation_errors_by_length_final.csv", index=False)\n'
        'final_errors.to_csv(ARTIFACT_DIR / "validation_prediction_audit_final.csv", index=False)')
notebook.cells[29].source += '\nprint("Co the submit ngay: /content/submission_selection.csv (Files panel > Refresh > Download).")\n'

replace(35, '    "notebook_version": "v3",', '    "notebook_version": NOTEBOOK_VERSION,')
replace(35, '    "baseline_public_score_user_reported": 0.87,',
        '    "baseline_public_score_user_reported": 0.87,\n'
        '    "v3_wolf_raw_validation_macro_f1": 0.9729325186,\n'
        '    "v3_wolf_public_score_user_reported": 0.97,\n'
        '    "v3_archive_sha256": "' + BASELINE_SHA256 + '",')
notebook.cells[35].source = '''# Re-read and synchronize the final CSV immediately before making the ZIP.
validate_submission(pd.read_csv(SUBMISSION_PATH, dtype=str, keep_default_na=False), test_df, sample_df)
shutil.copy2(SUBMISSION_PATH, ARTIFACT_DIR / "submission.csv")
''' + notebook.cells[35].source
replace(35, 'print("Artifacts:", ARTIFACT_DIR)',
        'with zipfile.ZipFile(ARCHIVE_PATH) as packed:\n'
        '    packed_csv = packed.read("submission.csv")\n'
        '    packed_config = json.loads(packed.read("run_config.json"))\n'
        '    packed_hash = hashlib.sha256(packed_csv).hexdigest()\n'
        '    assert packed_hash == run_config["submission_sha256"] == packed_config["submission_sha256"]\n'
        '    assert packed_csv == SUBMISSION_PATH.read_bytes(), "CSV trong ZIP khac file hien tai."\n'
        'print("ZIP/CSV SHA256 da khop:", packed_hash)\n'
        'print("Artifacts:", ARTIFACT_DIR)')

notebook.cells[38].source += """

V4 chuẩn hóa config theo [ModernBERT 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/configuration_modernbert.py).
`model_provenance` ghi config/tokenizer đã nạp; `validation_errors_raw.csv` ghi score
và margin thô, không phải độ tin cậy đã hiệu chỉnh. Bản V3 có output được lưu nguyên
tại `notebooks/archive/final_97_executed.ipynb` trong repository.
"""

# The saved run placed the download heading before the archive code. Restore
# sequential sections: export CSV -> archive -> download.
notebook.cells[34], notebook.cells[35] = notebook.cells[35], notebook.cells[34]
notebook.metadata.colab = {"name": "final_97.ipynb", "provenance": [], "gpuType": "T4"}
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []
    # Saved Colab execution IDs/timing are stale after rewriting the source.
    cell.metadata = {key: value for key, value in cell.metadata.items() if key in {"id", "tags"}}
nbformat.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"{DESTINATION.name}:cell_{index}", "exec")
nbformat.write(notebook, DESTINATION)
print("Wrote", DESTINATION.name, "| preserved V3 SHA256:", BASELINE_SHA256)
