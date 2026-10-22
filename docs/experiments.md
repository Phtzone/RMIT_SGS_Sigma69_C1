# Nhật ký thí nghiệm và kế hoạch cải tiến

Tài liệu này tách kết quả đã chạy khỏi cấu hình đang chuẩn bị. Điểm Public Leaderboard `0.87` của baseline và khoảng `0.97` của Wolf v3 là số người dùng báo lại sau khi nộp Kaggle; repository không có phản hồi chấm điểm của Kaggle để kiểm chứng độc lập. Người dùng đã cung cấp log train v4 nhưng chưa có notebook v4 đã chạy hoặc public score v4 tại repo. Notebook hiện tại là [final_97.ipynb](../final_97.ipynb), bản v5 chưa có kết quả GPU.

## Baseline đã chạy

Notebook baseline dùng DistilBERT multilingual 4-bit NF4 và QLoRA trên Tesla T4. Cấu hình chính gồm ba epoch, độ dài tối đa 512 token, chỉ giữ phần đầu văn bản, learning rate `2e-4`, LoRA rank 16, alpha 32 và hai target module `q_lin`, `v_lin`. Loss dùng class weight nghịch đảo tần suất:

| Nhãn | Số mẫu train | Class weight |
|---|---:|---:|
| `BENIGN` | 7.811 | 0,537703 |
| `JAILBREAK` | 3.907 | 1,074994 |
| `PROMPT_INJECTION` | 882 | 4,761905 |

Kết quả trên cùng validation cố định gồm 1.400 mẫu:

| Epoch | Validation loss | Macro F1 |
|---:|---:|---:|
| 1 | 0,306169 | 0,833761 |
| 2 | 0,229407 | 0,867555 |
| 3 | 0,228582 | **0,886366** |

Checkpoint tốt nhất nằm ở cuối epoch 3. Macro F1 vẫn tăng từ epoch 2 sang epoch 3, vì vậy baseline chưa cho thấy đã hội tụ hoàn toàn. Thời gian `32:46` chỉ là số quan sát của lần chạy baseline trên T4, không dùng để cam kết thời gian cho phiên bản sau.

Kết quả theo lớp:

| Nhãn | Precision | Recall | F1 |
|---|---:|---:|---:|
| `BENIGN` | 0,954717 | 0,873418 | 0,912260 |
| `JAILBREAK` | 0,797217 | 0,926097 | 0,856838 |
| `PROMPT_INJECTION` | 0,872549 | 0,908163 | 0,890000 |

Ma trận nhầm lẫn, hàng là nhãn thật và cột là nhãn dự đoán:

| Nhãn thật \ Nhãn dự đoán | `BENIGN` | `JAILBREAK` | `PROMPT_INJECTION` |
|---|---:|---:|---:|
| `BENIGN` | 759 | 98 | 12 |
| `JAILBREAK` | 31 | 401 | 1 |
| `PROMPT_INJECTION` | 5 | 4 | 89 |

Lỗi lớn nhất là 98 mẫu `BENIGN` bị đoán thành `JAILBREAK`. Model dự đoán 503 mẫu `JAILBREAK` trong khi validation chỉ có 433 mẫu thật. Đây là căn cứ để thử giảm class weight; chưa chứng minh class weight là nguyên nhân duy nhất. Theo ngôn ngữ, Macro F1 lần lượt là `pt=0,836839`, `en=0,856261`, `fr=0,939601` và `it=0,952737`. Hai nhóm cần ưu tiên là phân biệt `BENIGN`/`JAILBREAK` trong tiếng Bồ Đào Nha và `PROMPT_INJECTION` trong tiếng Anh.

Output baseline ghi compute dtype là `torch.bfloat16` trên Tesla T4. T4 không có BF16 phần cứng gốc, nên phiên bản mới sẽ chọn dtype theo kiến trúc GPU và ép FP16 trên T4. Việc này nhằm tránh đường tính BF16 không phù hợp; không được xem là bằng chứng tự thân rằng điểm F1 sẽ tăng.

## Thử nghiệm CPU mang tính thăm dò

Các thử nghiệm tuyến tính đều dùng đúng split cố định: 12.600 mẫu fit và 1.400 mẫu validation. Đặc trưng là TF-IDF ký tự; model chỉ được fit trên 12.600 mẫu, không dùng `test.csv` hoặc nhãn ẩn. Kết quả quan sát:

| Mô hình thăm dò | Validation Macro F1 |
|---|---:|
| SVM toàn cục, `char_wb` 3–5 | 0,922968 |
| SVM toàn cục, `char_wb` 2–6 | 0,929554 |
| SVM riêng theo từng ngôn ngữ | 0,937808 |
| Trộn điểm SVM toàn cục và theo ngôn ngữ, tỷ lệ 0,5/0,5 | 0,936607 |

Các số trên cho thấy đặc trưng ký tự có tín hiệu bổ sung cho DistilBERT, nhất là với mẫu đa ngôn ngữ, chuỗi mã, ký tự lạ và mẫu câu lặp. Chúng chưa phải ước lượng độc lập: khoảng n-gram, `C`, class weight và cách trộn đều được khảo sát dựa trên cùng validation. Một cấu hình tiếp tục được tinh chỉnh trên tập này từng cho điểm quan sát `0,938752`, nhưng điểm đó bị thiên lệch do chọn cấu hình trên chính tập đánh giá và không được dùng làm kết quả đại diện. Kết quả CPU cũng không bảo đảm mức điểm tương ứng trên Public hoặc Private Leaderboard.

## Phiên bản v2 đã chạy trên Colab

Phiên bản v2 giữ yêu cầu DistilBERT multilingual và QLoRA, với cấu hình:

| Thành phần | Cấu hình v2 |
|---|---|
| Xử lý văn bản dài | `head_tail` |
| Class weight | Căn bậc hai của inverse-frequency weight, sau đó chuẩn hóa |
| Số epoch | 5 |
| Learning rate | `1e-4` |
| LoRA | rank 32, alpha 64 |
| Target module | `q_lin`, `k_lin`, `v_lin`, `out_lin`, `lin1`, `lin2` |
| Tesla T4 | FP16 |

Nhánh TF-IDF/SVM là tùy chọn nhưng được bật mặc định trong v2. Nhánh này dùng `char_wb` 2–6, SVM `C=2`, `class_weight="balanced"`, gồm model toàn cục và model riêng theo ngôn ngữ. Hai decision score được trộn với trọng số `0,5/0,5`. Mọi vectorizer và SVM chỉ được fit trên 12.600 mẫu fit; validation không được đưa vào quá trình fit các model này.

Hậu xử lý trộn log-score của hai nhánh, thử trọng số SVM `0/0,25/0,5/0,75/0,9` và bias toàn cục. Cross-fitting ba phần bên trong validation giảm nguy cơ một mẫu vừa học tham số hậu xử lý vừa tự chấm chính nó. Chỉ bật hậu xử lý nếu điểm cross-fit tăng ít nhất `0,002` so với QLoRA thô; nếu không, `submission.csv` dùng QLoRA thô. `submission_distilbert_raw.csv` luôn được giữ trong artifacts để đối chiếu.

Cơ chế này chỉ là lớp bảo vệ cục bộ: cùng tập validation vẫn đã được dùng để chọn checkpoint, siêu tham số và cách trộn, nên điểm cuối vẫn có thiên lệch chọn model. Muốn có ước lượng chắc hơn cần một holdout chưa từng dùng hoặc đánh giá ngoài mẫu theo nhiều fold. `test_final_probabilities.npy` là score chuẩn hóa để lấy argmax, không phải xác suất độ tin cậy đã được hiệu chỉnh.

Notebook v2 người dùng cung cấp đã được lưu tại [RMIT_SGS_Sigma69_C1_Colab_v2_executed.ipynb](../notebooks/archive/RMIT_SGS_Sigma69_C1_Colab_v2_executed.ipynb). Kết quả quan sát từ output này:

| Phép đo | Macro F1 | Giới hạn |
|---|---:|---|
| DistilBERT raw, checkpoint epoch 5 | **0,919710** | Validation cũng dùng để chọn checkpoint |
| Ensemble/bias cross-fit | **0,952069** | Cross-fit chỉ áp dụng cho hậu xử lý; vẫn có thiên lệch từ chọn checkpoint/cấu hình |
| Ensemble/bias fit trên toàn validation | **0,958402** | Fit tham số và chấm trên cùng 1.400 mẫu; không phải holdout độc lập |

Tham số hậu xử lý đã lưu: neural temperature `2.0`, lexical temperature `0.5`, lexical weight `0.5`, class bias `[0.0, -0.3, 0.3]`. Mức tăng cross-fit so với raw là `0,032358`. Public `0.87` ở đầu tài liệu thuộc v1; chưa có public score v2 được kiểm chứng trong repository. Không dùng chênh lệch validation v2 và public v1 để kết luận test lệch phân bố.

Nhánh DistilBERT raw v2 còn yếu ở JAILBREAK tiếng Bồ Đào Nha: F1 `0,802030`, Macro F1 theo ngôn ngữ `0,883464`. English PROMPT_INJECTION đạt F1 `0,705882` trên nhóm validation rất nhỏ; cần xem số lỗi và support trước khi kết luận nguyên nhân. Kết quả theo ngôn ngữ còn lại: `fr=0,949102`, `it=0,975692`, `en=0,888671`.

## Tham khảo notebook Melbourne

[rmit-mel-segfault-c1.ipynb](../notebooks/archive/rmit-mel-segfault-c1.ipynb) dùng `aisuko/wolf-defender-prompt-injection`, được log nhận diện là `ModernBertForSequenceClassification`, và fine-tune toàn bộ model. Output lưu có raw validation `0,960082` và bias fit `0,964711`; public khoảng `0.97` là thông tin người dùng báo, không có phản hồi Kaggle trong file. Code hiện khai báo bốn epoch nhưng output chỉ ghi hai epoch; tên artifact cũng khác. Không xem output đó là bằng chứng đã chạy đúng code hiện tại. Chi tiết ở [melbourne-reference-review.md](melbourne-reference-review.md).

## Wolf v3 đã chạy: raw validation 0.972933

Người dùng đã cho phép đổi sang Wolf Defender nếu BTC duyệt. Ngày 22/10/2026, kiểm tra trực tiếp trang Challenge qua `competitions.PageService/ListPages` (competition `169360`, page `877551`) xác nhận `patronus-studio/wolf-defender-prompt-injection` nằm trong danh sách 10 model được duyệt, trỏ tới [Kaggle model của BTC](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default). Hướng hiện tại ưu tiên Wolf; không trộn Wolf và DistilBERT thành ensemble nhiều backbone. [Nguồn: Challenge chính thức](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview/challenge).

File người dùng đã chạy được giữ nguyên tại [final_97_executed.ipynb](../notebooks/archive/final_97_executed.ipynb). Wolf v3 fine-tune toàn bộ model: 4 epoch, LR `3e-5`, patience 2, 256 token đầu, class-weight power `0.5`, focal gamma `0`, batch 8/accumulation 4. Model dùng gradient checkpointing, SDPA, trọng số FP32 với mixed precision FP16 trên T4, tắt reference compile. Bằng chứng phê duyệt được lưu trong [competition-challenge.json](../demo/competition-challenge.json).

| Epoch | Raw validation Macro F1 |
|---:|---:|
| 1 | 0.945174 |
| 2 | 0.967531 |
| 3 | 0.972018 |
| 4 | **0.972933** |

Checkpoint epoch 4 sai **35/1.400 mẫu**; 27 lỗi là nhầm giữa BENIGN/JAILBREAK tiếng Bồ Đào Nha. Thống kê token cho thấy 45/195 mẫu JAILBREAK tiếng Bồ Đào Nha trong validation bị cắt ở context 256. File cũ không lưu ID của từng lỗi nên chưa thể khẳng định các lỗi này do truncation. Đây là căn cứ ưu tiên thử context dài hơn, không phải bằng chứng chắc chắn tăng điểm.

Lần chạy T4 này ghi khoảng **27:37** cho selection và **26:48** cho refit toàn bộ 14.000 mẫu, đều 4 epoch. Hai cell hậu xử lý/xuất selection trong file đã lưu không có output hay execution count, dù cell refit đã chạy và dùng biến từ chúng; vì vậy không khôi phục được điểm cross-fit cuối cùng từ notebook. Raw validation ở trên là số có bằng chứng trực tiếp. Public khoảng `0.97` là thông tin người dùng báo lại.

## V4 — kết quả từ log người dùng cung cấp

V4 giữ một backbone Wolf cùng version `/1`, cùng split 12.600/1.400, LR, số epoch và weighted CE đã chạy ở v3. Source được giữ tại [final_97_v4_source.ipynb](../notebooks/archive/final_97_v4_source.ipynb); file này không chứa output của lần train Colab. Các thay đổi so với v3:

- Tăng context từ 256 token đầu lên **512 token `head_tail`**, giữ đầu/cuối prompt và special tokens. Train batch 4/accumulation 8 giữ effective batch 32; eval batch 8.
- Chuẩn hóa metadata tokenizer của artifact Transformers 5 cho Transformers `4.57.6`, không sao chép weights; chuyển RoPE global/local đúng trường của phiên bản đang dùng.
- Nhánh TF-IDF/SVM thử lexical weight nhỏ hơn `[0, 0.05, 0.10, 0.20, 0.35, 0.50]` và bias toàn cục hẹp hơn `[-0.5, 0.5]`, giữ các mức nhỏ ở giữa. Chỉ bật hậu xử lý nếu cross-fit tăng ít nhất `0.002`; không thêm backbone hoặc ensemble nhiều seed. DistilBERT giữ grid cũ.
- Bổ sung báo cáo lỗi theo ngôn ngữ/độ dài và kiểm tra hash CSV trong artifacts để đối chiếu lần chạy.

Người dùng cung cấp bảng kết quả sau trong hội thoại:

| Epoch | Training loss | Validation loss | Macro F1 | Accuracy | F1 BENIGN | F1 JAILBREAK | F1 PROMPT_INJECTION |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.132100 | 0.147707 | 0.938744 | 0.945000 | 0.955908 | 0.926991 | 0.933333 |
| 2 | 0.077100 | 0.207038 | 0.960004 | 0.965000 | 0.973491 | 0.948626 | 0.957895 |
| 3 | 0.001900 | 0.179243 | **0.969274** | 0.970714 | 0.977650 | 0.955814 | 0.974359 |
| 4 | 0.000500 | 0.176366 | 0.967447 | 0.970000 | 0.977038 | 0.955916 | 0.969388 |

V4 tốt nhất ở epoch 3, thấp hơn v3 epoch 4 khoảng `0.003659` Macro F1. V4 chưa chứng minh cải thiện validation; chưa có dữ liệu kết luận public/private. V4 thay context, cách nạp RoPE/tokenizer và batch cùng lúc, nên không thể quy riêng việc giảm điểm cho context 512 hoặc suy ra mọi cấu hình 512 đều kém. Không ép model mới chọn epoch 3 chỉ vì lần này epoch 3 tốt nhất.

## V5 — trở về cấu hình v3, thử hậu xử lý có điều kiện

[final_97.ipynb](../final_97.ipynb) dùng 256 token đầu, batch 8/eval 16/accumulation 4, LR `3e-5`, tối đa 4 epoch, weighted CE power `0.5`, gamma `0`, seed/split seed 42. Checkpoint được chọn tự động bằng Macro F1. `WOLF_LOADING_MODE="v3_legacy"` tái hiện cách nạp config/tokenizer của v3 trong Transformers 4.57.6; `checkpoint_config` giữ cách ánh xạ RoPE của v4 làm một lựa chọn thử nghiệm riêng. Khôi phục cấu hình không đồng nghĩa tái lập đúng từng trọng số hoặc điểm v3.

Giữ hậu xử lý toàn cục v3 và ngưỡng cross-fit gain `0.002`. Bước mới chỉ hiệu chỉnh BENIGN/JAILBREAK trên mẫu tiếng Bồ Đào Nha có dự đoán trước hiệu chỉnh thuộc hai lớp này và chênh lệch tuyệt đối log-score hai lớp không quá `0.75`. Delta chỉ thử `[-0.3, -0.15, 0, 0.15, 0.3]`; không sửa dự đoán PROMPT_INJECTION hay ngôn ngữ khác.

Nested cross-fit so sánh pipeline có và không có hiệu chỉnh, fit tham số bằng phần train của từng outer fold. Chỉ bật nếu Macro F1 tổng tăng ít nhất `0.002`, Macro F1 tiếng Bồ Đào Nha tăng, ít nhất 2/3 outer fold tăng và phép fit trên toàn validation không giảm điểm. Không đạt thì giữ delta 0. Báo cáo `pt_pair_tuning.json` ghi kết quả/điều kiện; `submission_selection_global.csv` giữ nhánh trước hiệu chỉnh, `submission_selection.csv` giữ nhánh được chọn. Các score fit trên toàn validation chỉ là mô tả; cross-fit vẫn có thiên lệch do validation đã dùng chọn checkpoint và lựa chọn phương pháp thử nghiệm.

V5 mặc định `USE_FULL_TRAIN=False` để xem kết quả selection trước khi tốn thêm lượt train. Chưa có score GPU hoặc Kaggle v5; `0.99` là mục tiêu, không phải mức tăng dự báo. Không tăng class-weight power cùng lúc; đây sẽ là thí nghiệm riêng nếu báo cáo lỗi có căn cứ.

Phần thử nghiệm v5 diễn ra sau training, vì vậy Macro F1 thô trong bảng epoch không bắt buộc cao hơn v3. Bằng chứng cần xem là so sánh baseline/candidate bằng nested cross-fit và quyết định guard trong `pt_pair_tuning.json`, rồi các CSV selection. Không dùng điểm fit trên toàn validation để thay thế điểm cross-fit; ngay cả cross-fit tăng cũng không bảo đảm điểm public/private tăng.

Người dùng xác nhận hiện chỉ còn `submission.csv` của lần cũ, không còn logits hay ZIP artifacts. Nhãn cuối trong CSV không chứa score ba lớp, dữ liệu dự đoán validation hoặc weights, nên không đủ để fit/kiểm chứng bias và không thể thay cho artifacts. Lần v5 này cần train lại từ cấu hình v3.

V5 bổ sung gói nhỏ **`/content/selection_predictions.zip`** ngay ở mục 10, chứa logits validation/test của selection, ID, lexical scores nếu dùng, metadata run/loading, label mapping, tham số hậu xử lý và CSV; không chứa model weights. Tải gói này cùng CSV để các thử nghiệm hậu xử lý sau có thể dùng lại dự đoán mà không train GPU. Gói này không cho phép resume training hoặc suy luận trên dữ liệu mới nếu không có model; muốn giữ model phải tải thêm gói lớn `rmit_c1_artifacts.zip`.

[Kế hoạch nâng cấp](upgrade-099-plan.md) đặt mục tiêu `0.99`; mục tiêu không phải mức điểm được bảo đảm. Phương án DistilBERT QLoRA đề xuất trước khi đổi hướng là tối đa 8 epoch, learning rate `5e-5`, early-stopping patience 3, LoRA rank 64/alpha 128, class-weight power `0.65` và focal gamma `2.0`. Đây không phải cấu hình Wolf full fine-tuning.

Focal loss phải tính `p_t = exp(-CE)` từ **CE chưa nhân class weight**, rồi tính `weight[y] * (1-p_t)^gamma * CE`; không `detach()` `p_t`. Tăng class weight và dùng focal cùng lúc là một giả thuyết cần kiểm chứng qua precision/recall, không phải bảo đảm tăng F1 lớp hiếm.

`USE_FULL_TRAIN=True` phải thực hiện hai giai đoạn: (1) selection trên split 12.600/1.400 để chọn epoch và khóa tham số ensemble, (2) khởi tạo model mới từ pretrained checkpoint để refit trên cả 14.000 mẫu trong số epoch đã chọn. Final refit tắt evaluation, `load_best_model_at_end` và early stopping; không truyền holdout cũ làm eval vì model đã học những dòng đó. SVM/vectorizer được fit lại trên toàn train khi refit; tham số ensemble giữ nguyên từ selection. Lưu submission selection để đối chiếu với final refit.

Việc chuyển tham số bias từ selection sang model refit là giả định chưa được kiểm chứng độc lập. V5 mặc định dùng model selection (`USE_FULL_TRAIN=False`); chỉ bật refit nếu muốn thử riêng và đủ thời gian. Full refit không bảo đảm tốt hơn selection trên test.

## Quản lý notebook và chạy lại trên Colab

Baseline được lưu tại `notebooks/archive/RMIT_SGS_Sigma69_C1_Colab_baseline_087.ipynb`; v2 tại `notebooks/archive/RMIT_SGS_Sigma69_C1_Colab_v2_executed.ipynb`; v3 tại `notebooks/archive/final_97_executed.ipynb`. Source v4 được giữ tại `notebooks/archive/final_97_v4_source.ipynb`, không chứa output Colab do người dùng báo trong hội thoại. Notebook đang phát triển duy nhất ở thư mục gốc là **`final_97.ipynb`** (v5). Không ghi đè output lịch sử trong archive.

Trước khi thay notebook trong Colab, cần giữ lại kết quả của runtime cũ:

1. Mở bảng **Files** của Colab và tải trực tiếp `/content/submission.csv`.
2. Với v5, tải `/content/selection_predictions.zip` ngay sau mục 10; tải thêm `/content/rmit_c1_artifacts.zip` nếu muốn giữ model và file này đã được tạo.
3. Kiểm tra các file đã có trên máy trước khi ngắt hoặc khởi động lại runtime.
4. Upload `final_97.ipynb` ở thư mục gốc đã cập nhật lên Colab.
5. Dùng runtime GPU mới, chạy từ đầu theo đúng thứ tự cell và không tái sử dụng biến còn sót từ phiên cũ.
6. Sau khi tất cả cell hoàn tất, tải `submission.csv` từ thư mục gốc `/content` bằng bảng **Files**. File ZIP chỉ dùng lưu model và báo cáo, không dùng làm prediction file trên Kaggle.

Có thể kiểm tra phần CPU bằng `python scripts/check_notebook.py` sau khi cài các thư viện cần thiết. Các kiểm tra CPU/cú pháp không thay thế chạy end-to-end trên GPU; score mô phỏng chỉ phục vụ kiểm tra code.

## Kiểm tra source v5 ngày 22/10/2026

Kiểm tra CPU trên notebook v5 hiện tại đã qua: `check_v3_training.py`, `check_refit.py` (16 trường hợp giả lập), `check_validation.py`, `check_exports.py`, `check_v5_postprocess.py` và `check_selection_bundle.py`. Các kiểm tra bao gồm training helpers, refit, báo cáo validation, nested cross-fit PT và gói nhỏ `selection_predictions.zip`; chúng xác nhận quy trình/điều kiện bật tắt, không phải phép đo chất lượng model mới.

Đối chiếu cách nạp tokenizer legacy bằng Transformers **4.57.6** đã qua với các file metadata/tokenizer từ [commit nhà phát hành đã khóa](https://huggingface.co/patronus-studio/wolf-defender-prompt-injection/tree/a77e7b669f079c3ad7e72e4ad10f3bd173ac4f4d). Chế độ legacy giữ RoPE global/local thực tế `160000/10000` như cách phiên bản thư viện này diễn giải config; không tự áp dụng ánh xạ RoPE của v4. Phép kiểm tra không tải/train weights. Chưa train GPU v5 hoặc train lại baseline v3 trong lần sửa này; chưa có score v5.

## Kiểm tra đã thực hiện cho source v4 ngày 22/10/2026

Các script `check_notebook.py`, `check_v3_training.py`, `check_refit.py`, `check_validation.py` và `check_exports.py` trong `scripts/` đã chạy thành công trên source v4 ở lần sửa trước. Kiểm tra bao gồm schema/cú pháp, loss và gradient accumulation bằng mô phỏng NumPy, nhánh lexical trên split thật, cross-fit không phụ thuộc test scores, refit bằng Trainer giả lập, báo cáo lỗi và xuất CSV 6.000 dòng. Test export cố tình đưa CSV cũ vào artifacts rồi xác nhận code đóng ZIP đồng bộ lại đúng CSV cuối, giữ nguyên CSV selection và hash khớp nhau. Lần kiểm tra code này không train Wolf/DistilBERT trên GPU; người dùng sau đó chạy Colab và cung cấp log v4 ở trên.

`check_wolf_compat.py --download-upstream --dataset rmit-hackathon-2026/train.csv` cũng chạy thành công trong môi trường tạm riêng với **Transformers 4.57.6 / tokenizers 0.22.2**. Kiểm tra dùng ba file JSON nhỏ của [nhà phát hành Wolf tại commit đã khóa](https://huggingface.co/patronus-studio/wolf-defender-prompt-injection/tree/a77e7b669f079c3ad7e72e4ad10f3bd173ac4f4d), không tải weights. ID token, special tokens, head-tail, attention masks và lưu/nạp tokenizer đều giữ đúng; config RoPE global/local được ánh xạ từ nguồn. Cell tokenizer trong notebook được kiểm tra trùng chính xác với mã đã test.

Với tokenizer nhà phát hành này, trong 14.000 văn bản train có **1.376** mẫu vượt 256 token, **272** mẫu vượt 512 token và **59** mẫu vượt 1.024 token (đã tính hai special tokens). Context 512 giữ trọn thêm 1.104 prompt so với 256; đây là thống kê độ dài, không phải cải thiện F1 đã đo. Không giả định fixture nhà phát hành giống từng byte với Kaggle `/1`: notebook vẫn đọc và ghi hash của config/tokenizer Kaggle thực tế lúc chạy. SHA256 tokenizer fixture: `7e426c3929b44e6ab4c931770b5f22b913280633f5a1c67c81e9ad64decef55c`.

API đối chiếu: [LoRA config](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora), [PyTorch BF16 support](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html), [LinearSVC 1.6.1](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html).
