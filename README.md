# StreamNow — Báo cáo Dự đoán Churn Thuê bao

## 0. Cấu trúc repo

Repo gồm:

- `coursework.ipynb`: bản nộp chính, bao gồm tìm hiểu bài toán, load/clean/join dữ liệu, EDA, feature engineering, huấn luyện 2 model bắt buộc (Logistic Regression + Random Forest), đánh giá, tính business impact, đề xuất giám sát drift, và 2 phần mở rộng (survival analysis + SHAP).
- `model.pkl`: model cuối cùng được chọn để triển khai (Random Forest), lưu bằng `joblib`.
- `data/`: 4 file dữ liệu nguồn: `subscribers.csv`, `viewing_history.csv`, `support_tickets.csv`, `billing.csv`.
- `data_dictionary.md`: mô tả schema từng bảng và các lưu ý dữ liệu quan trọng (đặc biệt là rủi ro leakage của `cancellation_intent`).

Báo cáo này tóm tắt toàn bộ pipeline trong `coursework.ipynb`: bài toán, dữ liệu, EDA, feature engineering, 2 model, kết quả, business impact, giám sát drift, và phần mở rộng.

## 1. Bài toán

StreamNow muốn dự đoán subscriber nào có khả năng **churn** (huỷ gói) trong 30 ngày tới, để đội Retention có thể chủ động can thiệp (ưu đãi giữ chân, chăm sóc...) trước khi subscriber thực sự huỷ. Đây là bài toán phân loại nhị phân, nhãn `churned_30d` (1 = huỷ, 0 = còn ở lại), mỗi dòng dữ liệu là một subscriber tại thời điểm snapshot (`2026-06-20`).

Ràng buộc nghiệp vụ quan trọng: đội Retention chỉ có năng lực liên hệ chủ động với 20% subscriber rủi ro cao nhất, và tỷ lệ nền churn rất thấp (~5.8%) nên accuracy không có ý nghĩa — một mô hình "luôn đoán không churn" đã đạt ~94% accuracy nhưng vô dụng. Vì vậy, cách đánh giá xuyên suốt project dùng:

- ROC-AUC để so sánh/chọn model (không phụ thuộc ngưỡng, đo khả năng xếp hạng đúng rủi ro).
- Business impact tại đúng ngưỡng vận hành thực tế (top 20%), quy đổi ra doanh thu giữ được, thay vì các ngưỡng phân loại mặc định (0.5).

## 2. Dữ liệu

### 2.1 Các bảng nguồn

| File | Vai trò | Số dòng | Số cột chính |
|---|---|---|---|
| `subscribers.csv` | Bảng chính, 1 dòng/subscriber | 8,030 | `signup_date`, `plan`, `payment_method`, `age`, `gender`, `country`, `device_primary`, `churned_30d` |
| `viewing_history.csv` | 1 dòng/phiên xem, 90 ngày gần nhất | 92,196 | `session_date`, `duration_mins`, `content_type`, `completed`, `device` |
| `support_tickets.csv` | 1 dòng/ticket hỗ trợ, 6 tháng gần nhất | 9,978 | `ticket_date`, `category`, `resolved`, `resolution_days` |
| `billing.csv` | 1 dòng/chu kỳ thanh toán, tối đa 12 tháng | 72,219 | `billing_date`, `amount_due`, `payment_status`, `retry_count` |

### 2.2 Chất lượng dữ liệu

- **Trùng lặp**: `subscribers` có 30 dòng trùng `subscriber_id` loại bỏ, giữ bản ghi đầu tiên, đảm bảo mỗi subscriber chỉ xuất hiện đúng một lần.
- **Toàn vẹn tham chiếu**: không có `subscriber_id` mồ côi (orphan) giữa `viewing`/`tickets`/`billing` và `subscribers` → join an toàn bằng left-join.
- **Lỗi ngày tháng**: 4,266 phiên xem và một số ticket có `session_date`/`ticket_date` trước `signup_date` của chính subscriber đó về logic là không hợp lệ (không thể xem nội dung hay tạo ticket trước khi tồn tại trong hệ thống). Các dòng này bị loại trước bước tổng hợp feature ở Section 4, để tránh làm méo các feature như `total_sessions_90d`.
- **Missing có ý nghĩa vs missing do lỗi**: `resolution_days` thiếu ở 3,328/9,978 ticket (ticket chưa được xử lý xong) khác bản chất với missing ở `age`/`gender` (lỗi nhập liệu), nên **không** impute bằng median mà giữ NaN/xử lý riêng ở bước feature engineering.
- **Không hoạt động ≠ thiếu dữ liệu**: theo đúng lưu ý ở `data_dictionary.md`, subscriber không có phiên xem nào trong 90 ngày gần nhất được coi là tín hiệu thật (không hoạt động), không phải dữ liệu bị thiếu — `total_sessions_90d`/`total_watch_mins_90d` được điền `0` chứ không drop.
- **Mất cân bằng lớp**: churn rate tổng thể chỉ 5.8% không phải lỗi dữ liệu, nhưng là lý do chính chi phối lựa chọn metric (ROC-AUC thay vì accuracy) và cách xử lý imbalance khi huấn luyện (`class_weight='balanced'`).

### 2.3 Join & tổng hợp dữ liệu

Ba bảng giao dịch (`viewing_history`, `support_tickets`, `billing`) được lọc lỗi ngày tháng rồi tổng hợp về 1 dòng/subscriber, sau đó join với `subscribers` bằng left-join (join với toàn bộ danh sách subscriber, kể cả người không có hoạt động nào) để đảm bảo không mất subscriber không hoạt động:

- **Viewing** → `total_sessions_90d`, `total_watch_mins_90d`, `completion_rate`, `days_since_last_session`, `view_trend_mins` (watch time 30 ngày gần nhất trừ 30 ngày trước đó).
- **Support tickets** → `total_tickets`, `pct_unresolved`, `avg_resolution_days`, `num_cancel_intent_tickets` (đếm riêng ticket loại `cancellation_intent`).
- **Billing** → `total_billing_cycles`, `billing_failure_rate`, `avg_retry_count`, `total_amount_due`.

Kết quả là bảng `master` 1 dòng/subscriber, đầy đủ đặc trưng hành vi/thanh toán/hỗ trợ.
