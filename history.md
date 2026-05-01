# Nhật ký Thay đổi Dự án (Project History)

## [Current Version] - Nâng cấp Môi trường AircraftEnv (Single-Agent)

Trong giai đoạn này, chúng ta đã tinh chỉnh môi trường Reinforcement Learning để phản ánh vật lý bay thực tế và loại bỏ các yếu tố "cầm tay chỉ việc", giúp PPO Agent tự chủ hơn:

### 1. Cấu trúc Vật lý bay (Flight Dynamics)
* **Khởi tạo ở mặt đất:** Máy bay không còn tự động bắt đầu ở độ cao 10000m. Giá trị khởi tạo là `0.0`.
* **Cơ chế cất cánh (Take-off):** Agent bắt buộc phải chọn hành động `CLIMB` để nâng độ cao. Nếu đứng ở mặt đất mà chọn `CRUISE` hoặc `DESCEND`, máy bay sẽ không di chuyển, tiêu hao nhiên liệu (Idle) và bị trừ điểm.

### 2. Cấu trúc Phần thưởng (Reward System)
* **Xóa bỏ SAFE_RUL:** Đã loại bỏ hằng số `SAFE_RUL = 30` và các phần thưởng dẫn dắt (Shaping Reward) dựa trên khoảng cách khi RUL thấp.
* **Maintenance Reward tự nhiên:** Áp dụng công thức thưởng rủi ro cho việc hạ cánh bảo trì: `Reward = 10.0 + max(0.0, (150.0 - current_rul) * 0.1)`. Hạ cánh khi RUL càng cận kề số 0 (nguy hiểm cao) thì điểm thưởng nhận được càng lớn.

### 3. Quy mô Lộ trình & Dữ liệu
* **Dataset:** Tạm thời cố định môi trường chỉ sử dụng **Động cơ số 1 (Unit 1)** làm mốc huấn luyện (Proof of Concept). Với tuổi thọ 191 cycles, điều này đảm bảo máy bay không bị rơi lãng xẹt do động cơ quá yếu.
* **Quãng đường:** Tăng `TOTAL_DISTANCE` từ 10,000 lên **20,000** đơn vị.
* **Sân bay phụ:** Điều chỉnh số lượng sân bay phụ (`NUM_SUB_AIRPORTS`) xuống còn **6**.
* **Phân phối sân bay:** Sử dụng logic phân phối ngẫu nhiên có kiểm soát (Base points $\pm 500m$). Điều này giúp rải rác 6 sân bay trên quãng đường 20,000m với khoảng cách tối đa giữa 2 trạm là ~3857m (thấp hơn nhiều so với tầm bay an toàn ~4600m của Động cơ 1). Điều này đảm bảo tính khả thi của trò chơi trong mọi tập (episode).
