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

## [2026-05-01] - Chống Exploit & Tối Ưu Hóa Reward (Anti-Farm + Exploration Fix)

### Bối Cảnh
Sau các lần chạy huấn luyện trước, Agent PPO bị plateau ở ~500 steps do 2 nguyên nhân gốc rễ:
1. **"Bức tường nhiên liệu"**: `FUEL_CAPACITY=250` / `FUEL_RATE=0.5` → Agent chỉ sống được đúng 500 steps rồi hết xăng chết.
2. **Maintenance Reward phản tác dụng**: Công thức cũ `40 - (RUL*0.5) - (fuel/cap*20)` cho kết quả **âm** khi RUL/fuel còn cao → Agent bị phạt khi hạ cánh tiếp xăng đúng lúc → chọn "bay thẳng cho đến chết" thay vì bảo trì.

### Thay Đổi Trong `scripts/aircraft_env.py`

#### 1. `LANDING_THRESHOLD`: 400 → 300
* Giảm nhẹ so với giá trị 400 cũ nhưng đảm bảo cửa sổ hạ cánh đủ rộng (~21% khoảng cách giữa 2 sân bay).
* Tăng xác suất Agent khám phá hạ cánh thành công trong giai đoạn Exploration.

#### 2. `FUEL_CAPACITY`: 300 → 250
* Ép Agent phải hạ cánh tiếp xăng ít nhất 1 lần (12,500m tối đa < 20,000m tổng).
* Tạo bài toán có ý nghĩa cho Predictive Maintenance.

#### 3. Airport Noise: ±500 → ±200
* Giảm độ ngẫu nhiên vị trí sân bay để Value Function ổn định hơn.
* Giảm phương sai môi trường, giúp Agent học nhanh hơn trong giai đoạn đầu.

#### 4. Idling Penalty: -0.5 → -2.0
* Tăng hình phạt đứng im để ngăn Agent "ngồi đất chờ hết giờ".

#### 5. Thêm `MAX_STEPS = 1000` + biến đếm `self.current_step`
* Ngăn vòng lặp vô hạn nếu Agent không bao giờ kết thúc episode.
* Khi vượt quá → trả về `truncated=True` (không phải `done=True`).

#### 6. Hình phạt FIELD_CRASH / FUEL_EMPTY / CRASHED: đồng nhất về -40
* Cân bằng hình phạt: đủ nặng để Agent không cố tình chết, nhưng không quá nặng để Agent không sợ khám phá.

#### 7. Reward Shaping: `0.05` → `distance_covered / 1000.0`
* Thưởng mỗi bước tỷ lệ với tốc độ di chuyển (Cruise: +0.025, Climb: +0.02, Descend: +0.015).
* Cung cấp dense reward signal dẫn lối Agent đi về phía trước.

#### 8. [CHÍNH] Maintenance Reward — Chống Loop Exploit
* **Vấn đề cũ:** `40 - RUL*0.5 - fuel/cap*20` → cho kết quả âm khi khỏe → Agent bị phạt khi hạ cánh hợp lý.
* **Exploit mới phát sinh khi sửa:** Nếu reward luôn dương cố định → Agent có thể CLIMB 1 step + DESCEND 1 step tại chỗ (~35m) để farm maintenance reward vô hạn.
* **Giải pháp:** Reward tỷ lệ với **khoảng cách đã bay từ lần bảo trì trước**:
  ```
  progress_bonus = min(dist_since_last_maintenance / 100.0, 30.0)  # 0 → +30
  rul_bonus      = max(0.0, (80.0 - current_rul) * 0.3)            # 0 → +24
  fuel_bonus     = max(0.0, (1 - fuel/capacity) * 15.0)            # 0 → +15
  maintenance_reward = progress_bonus + rul_bonus + fuel_bonus      # Luôn >= 0
  ```
* Farm loop (~35m) chỉ nhận được **~0.35 điểm** → không đáng.
* Bay đúng 1 segment (~2857m) nhận được **tối đa +69 điểm** → rất hấp dẫn.
* Thêm biến `self.dist_since_last_maintenance` (reset về 0 sau mỗi lần bảo trì).

### Khuyến Nghị Hyperparameter (demo_flow.ipynb)
* Dùng **LR cố định `3e-4`** thay vì `linear_schedule` để đảm bảo LR vẫn đủ lớn khi Agent khám phá ra hành vi tốt ở giữa/cuối training.
* Giữ `ent_coef=0.05` để duy trì Exploration đủ lâu.

### Quản lý Repository
* Cập nhật `.gitignore`: Bổ sung `logs/` và `mlruns/` để loại bỏ các file kết quả huấn luyện nặng nề khỏi Git.
