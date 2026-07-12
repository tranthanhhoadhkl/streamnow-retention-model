# StreamNow — Báo cáo Dự đoán Churn Thuê bao

## 0. Cấu trúc repo

Repo gồm:

- `coursework.ipynb` — bản nộp chính, bám theo đúng cấu trúc/yêu cầu của mark scheme (9 section + Extension), từ tìm hiểu bài toán, load/clean/join dữ liệu, EDA, feature engineering, huấn luyện 2 model bắt buộc (Logistic Regression + Random Forest), đánh giá, tính business impact, đề xuất giám sát drift, và 2 phần mở rộng (survival analysis + SHAP).
- `model.pkl` — model cuối cùng được chọn để triển khai (Random Forest), lưu bằng `joblib`.
- `data/` — 4 file dữ liệu nguồn: `subscribers.csv`, `viewing_history.csv`, `support_tickets.csv`, `billing.csv`.
- `data_dictionary.md` — mô tả schema từng bảng và các lưu ý dữ liệu quan trọng (đặc biệt là rủi ro leakage của `cancellation_intent`).

Báo cáo này tóm tắt toàn bộ pipeline trong `coursework.ipynb`: bài toán, dữ liệu, EDA, feature engineering, 2 model, kết quả, business impact, giám sát drift, và phần mở rộng.

## 1. Bài toán

StreamNow muốn dự đoán subscriber nào có khả năng **churn** (huỷ gói) trong **30 ngày tới**, để đội Retention có thể chủ động can thiệp (ưu đãi giữ chân, chăm sóc...) trước khi subscriber thực sự huỷ. Đây là bài toán **phân loại nhị phân**, nhãn `churned_30d` (1 = huỷ, 0 = còn ở lại), mỗi dòng dữ liệu là một subscriber tại thời điểm snapshot (`2026-06-20`).

Ràng buộc nghiệp vụ quan trọng: đội Retention chỉ có năng lực liên hệ chủ động với **20% subscriber rủi ro cao nhất**, và tỷ lệ nền churn rất thấp (~5.8%) nên **accuracy không có ý nghĩa** — một mô hình "luôn đoán không churn" đã đạt ~94% accuracy nhưng vô dụng. Vì vậy, cách đánh giá xuyên suốt project dùng:

- **ROC-AUC** để so sánh/chọn model (không phụ thuộc ngưỡng, đo khả năng xếp hạng đúng rủi ro).
- **Business impact tại đúng ngưỡng vận hành thực tế (top 20%)**, quy đổi ra doanh thu giữ được, thay vì các ngưỡng phân loại mặc định (0.5).

## 2. Dữ liệu

### 2.1 Các bảng nguồn

| File | Vai trò | Số dòng | Số cột chính |
|---|---|---|---|
| `subscribers.csv` | Bảng chính, 1 dòng/subscriber | 8,030 | `signup_date`, `plan`, `payment_method`, `age`, `gender`, `country`, `device_primary`, `churned_30d` |
| `viewing_history.csv` | 1 dòng/phiên xem, 90 ngày gần nhất | 92,196 | `session_date`, `duration_mins`, `content_type`, `completed`, `device` |
| `support_tickets.csv` | 1 dòng/ticket hỗ trợ, 6 tháng gần nhất | 9,978 | `ticket_date`, `category`, `resolved`, `resolution_days` |
| `billing.csv` | 1 dòng/chu kỳ thanh toán, tối đa 12 tháng | 72,219 | `billing_date`, `amount_due`, `payment_status`, `retry_count` |

### 2.2 Chất lượng dữ liệu

- **Trùng lặp**: `subscribers` có 30 dòng trùng `subscriber_id` — loại bỏ, giữ bản ghi đầu tiên, đảm bảo mỗi subscriber chỉ xuất hiện đúng một lần.
- **Toàn vẹn tham chiếu**: không có `subscriber_id` mồ côi (orphan) giữa `viewing`/`tickets`/`billing` và `subscribers` → join an toàn bằng left-join.
- **Lỗi ngày tháng**: 4,266 phiên xem và một số ticket có `session_date`/`ticket_date` **trước** `signup_date` của chính subscriber đó — về logic là không hợp lệ (không thể xem nội dung hay tạo ticket trước khi tồn tại trong hệ thống). Các dòng này bị loại **trước** bước tổng hợp feature ở Section 4, để tránh làm méo các feature như `total_sessions_90d`.
- **Missing có ý nghĩa vs missing do lỗi**: `resolution_days` thiếu ở 3,328/9,978 ticket (ticket chưa được xử lý xong) — khác bản chất với missing ở `age`/`gender` (lỗi nhập liệu), nên **không** impute bằng median mà giữ NaN/xử lý riêng ở bước feature engineering.
- **Không hoạt động ≠ thiếu dữ liệu**: theo đúng lưu ý ở `data_dictionary.md`, subscriber không có phiên xem nào trong 90 ngày gần nhất được coi là tín hiệu thật (không hoạt động), không phải dữ liệu bị thiếu — `total_sessions_90d`/`total_watch_mins_90d` được điền `0` chứ không drop.
- **Mất cân bằng lớp**: churn rate tổng thể chỉ **5.8%** — không phải lỗi dữ liệu, nhưng là lý do chính chi phối lựa chọn metric (ROC-AUC thay vì accuracy) và cách xử lý imbalance khi huấn luyện (`class_weight='balanced'`).

### 2.3 Join & tổng hợp dữ liệu

Ba bảng giao dịch (`viewing_history`, `support_tickets`, `billing`) được lọc lỗi ngày tháng rồi tổng hợp về **1 dòng/subscriber**, sau đó join với `subscribers` bằng left-join (join với **toàn bộ** danh sách subscriber, kể cả người không có hoạt động nào) để đảm bảo không mất subscriber không hoạt động:

- **Viewing** → `total_sessions_90d`, `total_watch_mins_90d`, `completion_rate`, `days_since_last_session`, `view_trend_mins` (watch time 30 ngày gần nhất trừ 30 ngày trước đó).
- **Support tickets** → `total_tickets`, `pct_unresolved`, `avg_resolution_days`, `num_cancel_intent_tickets` (đếm riêng ticket loại `cancellation_intent`).
- **Billing** → `total_billing_cycles`, `billing_failure_rate`, `avg_retry_count`, `total_amount_due`.

Kết quả là bảng `master` — 1 dòng/subscriber, đầy đủ đặc trưng hành vi/thanh toán/hỗ trợ.

## 3. Khám phá dữ liệu (EDA) — các phát hiện chính

Xếp theo mức độ liên quan đến churn:

1. **Số lần thanh toán thất bại là tín hiệu rõ ràng và đơn điệu nhất**: churn rate tăng gần như tuyến tính từ ~4.4% (0 lần thất bại) lên ~15.3% (5 lần) — tăng hơn 3 lần, không nhiễu, không ngoại lệ.
2. **Tenure (thời gian gắn bó)** rất mạnh: tenure trung vị của nhóm churn chỉ **122 ngày**, so với **427 ngày** ở nhóm không churn — chênh lệch hơn 3.5 lần.
3. **Hoạt động xem gần đây (30 ngày)** rất mạnh: số phiên xem trung vị của nhóm churn gần bằng 0, so với 2 ở nhóm không churn — phần lớn subscriber sắp huỷ đã gần như ngừng dùng dịch vụ *trước khi* chính thức huỷ.
4. **Loại gói (`plan`)** là mối quan hệ yếu nhất trong 4 biến trên: Premium có churn cao nhất (8.3%) so với Basic/Standard (~4.7–5.0%), nhưng chênh lệch tuyệt đối chỉ 3–4 điểm phần trăm.
5. **Phân khúc rủi ro cao nhất**: Premium (8.33% — cao nhất trong mọi phân khúc) và Voucher (7.16% — cao nhất trong các phương thức thanh toán). Premium đáng lo vì đây cũng là nhóm mang lại doanh thu/subscriber cao nhất; Voucher đáng lo vì nhóm này thường thiếu cam kết tài chính dài hạn (không gắn thẻ tín dụng/ghi nợ định kỳ).
6. **Phân phối tenure** có một đỉnh bất thường ở giá trị lớn nhất (~1750–1830 ngày), tách biệt hẳn khỏi phần đuôi giảm dần — dấu hiệu "left-censoring": có một ngày ra mắt dịch vụ cụ thể, mọi subscriber đăng ký từ ngày đầu đều bị dồn về cùng một giá trị tenure tối đa.
7. **Vé hỗ trợ liên quan billing** là loại phổ biến nhất (~2,500 vé) — nhiều hơn cả vé kỹ thuật, dù với nền tảng streaming người ta thường kỳ vọng lỗi kỹ thuật (buffering, app lỗi...) mới là nguồn vé lớn nhất. Đây là gợi ý về ma sát thanh toán là vấn đề vận hành đáng chú ý.

## 4. Feature engineering

Từ bảng `master`, xây dựng thêm các feature mới, mỗi feature bám theo một phát hiện cụ thể từ EDA:

| Feature | Ý nghĩa | Vì sao dự đoán churn |
|---|---|---|
| `tenure_bucket` | Nhóm tenure: `new` (<3 tháng) / `growing` (3–12 tháng) / `mature` (>12 tháng) | Subscriber mới chưa hình thành thói quen sử dụng/cam kết, dễ huỷ sớm nếu trải nghiệm ban đầu không như kỳ vọng |
| `view_accel` | Tốc độ thay đổi watch time 30 ngày gần nhất so với 30 ngày trước đó | Giá trị âm/thấp (xem ngày càng ít) báo hiệu subscriber đang dần rời xa dịch vụ trước khi chính thức huỷ |
| `billing_failure_rate` | Tỷ lệ chu kỳ thanh toán thất bại/tranh chấp trên tổng số chu kỳ | Thanh toán thất bại vừa là nguyên nhân trực tiếp gây huỷ tự động (thẻ hết hạn, thiếu tiền), vừa phản ánh gián tiếp mức độ gắn bó thấp |
| `has_cancel_intent_ticket` | Đã từng tạo ticket loại `cancellation_intent` hay chưa (0/1) | Tín hiệu ý định huỷ trực tiếp và mạnh nhất trong toàn bộ dữ liệu — gần như một tuyên bố ý định từ khách hàng |
| `avg_session_duration` | Watch time trung bình mỗi phiên = tổng phút / tổng số phiên | Giá trị thấp có thể phản ánh nội dung không đủ hấp dẫn hoặc subscriber thử rồi bỏ giữa chừng — khác với `completion_rate` (đo mức độ hoàn thành từng lượt xem) |
| `is_inactive_90d` | Cờ đánh dấu subscriber không có phiên xem nào trong 90 ngày | Tín hiệu "ngắt kết nối" mạnh nhất — vẫn trả tiền nhưng không dùng dịch vụ thường là nhóm dễ huỷ nhất khi họ rà soát lại chi tiêu hàng tháng |
| `days_since_last_session` | Số ngày kể từ phiên xem gần nhất | Càng lâu không xem, khả năng subscriber đã "rời bỏ về mặt tinh thần" trước khi chính thức huỷ càng cao |

`num_cancel_intent_tickets` (đếm số ticket ý định huỷ) cũng được giữ lại bên cạnh cờ nhị phân `has_cancel_intent_ticket`, theo đúng lưu ý ở `data_dictionary.md` về việc cân nhắc kỹ trước khi đưa tín hiệu này vào model để tránh leakage — nhóm quyết định **giữ lại** vì đây là tín hiệu hành vi đã xảy ra tại thời điểm snapshot (không phải thông tin từ tương lai), không phải leakage theo đúng nghĩa thời gian.

## 5. Huấn luyện mô hình

### 5.1 Chia train/validation

Dữ liệu được chia **80/20**, stratify theo `churned_30d` để giữ đúng tỷ lệ churn ở cả hai tập (`train_test_split(..., stratify=y)`), đánh giá bằng `StratifiedKFold` 5-fold trên tập train.

### 5.2 Hai model bắt buộc

| Model | Xử lý imbalance | Regularisation/Tuning |
|---|---|---|
| **Logistic Regression** | `class_weight='balanced'` | L2 (Ridge), `C` tune qua `GridSearchCV` trên `{0.01, 0.1, 1.0, 10.0}` — ưu tiên vì nhiều feature tổng hợp từ `viewing_history` có khả năng tương quan với nhau (ví dụ tổng watch time và số phiên) |
| **Random Forest** | `class_weight='balanced'` | `GridSearchCV` trên `n_estimators∈{100,200,300,500}`, `max_depth∈{4,6,8}`, `min_samples_leaf∈{10,20}` — giới hạn độ sâu và kích thước lá vì tập train chỉ ~6,400 dòng, tránh cây "học thuộc" nhiễu |

Cả hai được đánh giá bằng ROC-AUC (CV mean và validation), không dùng ngưỡng 0.5 mặc định vì mục tiêu cuối cùng là **xếp hạng** subscriber theo rủi ro để chọn top 20%, không phải phân loại nhị phân đơn thuần.

### 5.3 Kết quả so sánh & model được chọn

| Model | CV AUC | Val AUC | Overfit gap |
|---|---|---|---|
| Random Forest | 0.9126 | **0.9138** | **−0.0011** |
| Logistic Regression | 0.9129 | 0.9094 | +0.0035 |

Hai model gần như ngang nhau ở CV AUC (0.9126 vs 0.9129), nhưng Random Forest có Validation AUC cao hơn và **overfit gap âm** (thực ra hoạt động tốt hơn một chút trên tập validation so với trong cross-validation) — dấu hiệu `max_depth`/`min_samples_leaf` đã kiểm soát tốt việc overfit. Logistic Regression có overfit gap dương và lớn hơn, tức hoạt động kém hơn ngoài fold so với trong CV.

**Model được chọn triển khai: Random Forest.** Nhược điểm duy nhất là khả năng diễn giải (Logistic Regression có hệ số +/- trực quan hơn), nhưng được bù đắp bằng SHAP ở Extension E2, giải thích chi tiết từng subscriber rủi ro cao nhất thay vì chỉ importance tổng thể.

## 6. Đánh giá mô hình

- **ROC curve**: cả hai model nằm rõ trên đường chéo random-guess, AUC ~0.91 trên validation — mức phân tách mạnh so với tỷ lệ nền churn chỉ 5.8%.
- **Phân phối điểm dự đoán**: phần lớn non-churner dồn gần 0, trong khi churner trải rộng từ ~0.2 đến 0.9 thay vì dồn gần 0 như nhóm còn lại — đúng hình dạng kỳ vọng ở mức AUC ~0.91, cho thấy model thực sự đẩy churner thật lên điểm cao chứ không chỉ đoán an toàn cho tất cả.
- **SHAP (Extension E2)** trên top 5 subscriber rủi ro cao nhất (xác suất dự đoán 0.946–0.962) cho thấy `num_cancel_intent_tickets` là yếu tố đóng góp lớn nhất ở cả 5 trường hợp (+0.108 đến +0.254), tiếp theo là `has_cancel_intent_ticket` (+0.106 đến +0.180) — hai tín hiệu ý định huỷ chiếm phần lớn lý do các subscriber này đứng đầu danh sách rủi ro. `days_since_last_session` (+0.062 đến +0.092) đứng thứ ba, cho thấy sự mất kết nối cộng dồn với ý định huỷ đã nêu, chứ không phải hai tín hiệu này tách rời nhau. Các feature liên quan billing (`total_billing_cycles`, `avg_retry_count`, `billing_failure_rate`) là yếu tố phụ nhưng xuất hiện nhất quán ở phần lớn các trường hợp.

## 7. Tác động kinh doanh (Business Impact)

Trên tập validation, so sánh chính sách **liên hệ top 20% subscriber theo rủi ro dự đoán** với chính sách **liên hệ ngẫu nhiên 20%**, giả định tỷ lệ giữ chân thành công 35% khi liên hệ, quy đổi ra biên lợi nhuận/tháng theo từng gói (`basic`: £5.94, `standard`: £10.19, `premium`: £15.29):

| Chính sách | Doanh thu giữ được / tháng |
|---|---|
| Random 20% | (kỳ vọng theo tỷ lệ churner trong tập validation) |
| **Model (top 20% theo rủi ro)** | **+£210.66 / tháng cao hơn random (+321.9%)** |

Tương đương khoảng **£2,527.98/năm** giá trị tăng thêm so với chọn ngẫu nhiên, trên riêng slice validation này.

**Khuyến nghị**: đội Retention nên xếp hạng subscriber theo xác suất churn của model và ưu tiên liên hệ đúng top 20% rủi ro cao nhất, thay vì chọn ngẫu nhiên. Vì các gói khác nhau mang lại biên lợi nhuận khác nhau, nên ưu tiên thêm các subscriber gói **Premium** trong nhóm rủi ro cao (giữ được một Premium subscriber có giá trị hơn một Basic subscriber). Model cần được giám sát và huấn luyện lại định kỳ để duy trì hiệu quả khi hành vi khách hàng thay đổi.

## 8. Giám sát drift khi lên production

- **Input drift**: theo dõi phân phối từng feature theo cửa sổ trượt (tuần/tháng) so với phân phối lúc train — PSI hoặc Kolmogorov–Smirnov cho biến numeric (`total_watch_mins_90d`, `billing_failure_rate`, `days_since_last_session`...), chi-square cho biến categorical (`plan`, `payment_method`, `device_primary`). Nhóm dễ drift nhất: hành vi xem (thay đổi theo mùa/nội dung mới ra mắt) và mix `plan`/`payment_method` (do quyết định thương mại của StreamNow, ví dụ khuyến mãi mới).
- **Target drift**: theo dõi định kỳ (1) churn rate thực tế so với baseline ~5.8%, (2) ROC-AUC/calibration thực tế so với baseline ~0.91 (đo được sau khi biết kết quả churn thật của từng đợt snapshot). Kích hoạt retrain khi: AUC live giảm rõ rệt và kéo dài, capture rate ở ngưỡng top 20% giảm, hoặc có thay đổi cấu trúc mới (gói mới, thị trường mới) mà model chưa từng thấy.

## 9. Extension

- **E1 — Survival analysis (Kaplan–Meier)**: ước lượng đường cong sống sót theo từng gói (`basic`/`standard`/`premium`) bằng `lifelines`. Đúng như phát hiện ở Section 3 (tenure trung vị 122 ngày ở nhóm churn vs 427 ngày ở nhóm không churn), đường cong giảm dốc nhất ở vùng tenure sớm (vài trăm ngày đầu) rồi thoải dần với các subscriber đã "sống sót" đủ lâu để được coi là `mature` — nhất quán với logic của `tenure_bucket` ở Section 5.
- **E2 — SHAP**: xem Section 6 — hai tín hiệu ý định huỷ (`num_cancel_intent_tickets`, `has_cancel_intent_ticket`) và sự mất kết nối gần đây (`days_since_last_session`) là bộ ba yếu tố chi phối rủi ro của các subscriber cao nhất, đúng với trực giác nghiệp vụ.

## 10. Tổng kết

- **Model triển khai**: Random Forest (`class_weight='balanced'`, `max_depth`/`min_samples_leaf` tune qua `GridSearchCV`), Val ROC-AUC ≈ **0.914**, lưu tại `model.pkl`.
- **Tại ngưỡng vận hành thực tế (top 20%)**: chính sách theo model tạo thêm **£210.66/tháng** (≈ **£2,527.98/năm**) so với liên hệ ngẫu nhiên — lift **321.9%**.
- **Tín hiệu churn mạnh nhất**: ý định huỷ đã nêu rõ (`cancellation_intent` ticket), mất kết nối gần đây (không xem trong 90 ngày / `days_since_last_session` cao), tenure ngắn, và thanh toán thất bại lặp lại — nhất quán giữa EDA, hệ số/importance của model, và SHAP.
- Toàn bộ pipeline xử lý đúng các lưu ý trong `data_dictionary.md`: loại hoạt động trước ngày đăng ký, giữ "không hoạt động" như một tín hiệu thật (không impute bằng 0 một cách mù quáng cho mọi cột), và cân nhắc rõ ràng trước khi đưa `cancellation_intent` vào model.
