[🏠 Home](./README.md) | [📜 History](./history.md)

# Project Evolution History

Dưới đây là lịch sử thay đổi và quá trình phát triển hệ thống **Aircraft Digital Twin - Predictive Maintenance**.

---

### 🕒 **2026-03-30 | 08:30 - 09:30 AM**

#### **Thay đổi: Tạo mô phỏng 2D đơn giản (Pygame)**

- **Tiếp cận:** Cung cấp một công cụ trực quan hóa nhanh chóng bằng Python để kiểm thử Logic của môi trường Gymnasium (`AircraftEnv`).
- **Nội dung:** Tạo file `scripts/sim_2d.py` sử dụng thư viện Pygame để vẽ máy bay, sân bay và các chỉ số RUL/Nhiên liệu.
- **Trạng thái:** Hoàn thành (Lưu trữ như một công cụ dev nội bộ).

---

### 🕒 **2026-03-30 | 01:15 - 02:00 PM**

#### **Thay đổi: Chuyển đổi sang kiến trúc WebSockets & FastAPI**

- **Tiếp cận:** Tách biệt Logic mô phỏng (Backend) và Giao diện người dùng (Frontend) để hỗ trợ điều khiển thời gian thực và giao diện hiện đại hơn.
- **Nội dung:**
  - Tạo `server.py`: Server FastAPI sử dụng WebSockets để stream dữ liệu vật lý từ `AircraftEnv` với tần suất ~3 FPS.
  - Xử lý các lệnh điều khiển (`FLY`, `LAND`, `RESET`) gửi ngược từ Client về Server.

---

### 🕒 **2026-03-30 | 02:00 - 02:10 PM**

#### **Thay đổi: Xây dựng Dashboard Next.js (Premium UI)**

- **Tiếp cận:** Sử dụng Next.js, Tailwind CSS và Framer Motion để tạo một bảng điều khiển Dashboard chuyên nghiệp theo phong cách "Glassmorphism".
- **Nội dung:**
  - `useSimulation.ts`: Hook quản lý kết nối WebSocket.
  - `Environment2D.tsx`: Hiển thị đồ họa máy bay bằng SVG (mượt mà và nhẹ hơn).
  - `Dashboard.tsx`: Bảng điều khiển telemetry (Altitude, Fuel, RUL) và các nút điều hướng.
  - `globals.css`: Tùy chỉnh màu sắc Dark mode cao cấp.

---

### 🕒 **2026-04-03 | 03:00 - 03:30 PM**

#### **Thay đổi: Giải quyết xung đột Dependency (LangGraph)**

- **Tiếp cận:** Điều chỉnh phiên bản thư viện trong môi trường Python để đảm bảo tính ổn định của hệ thống đa tác vụ (Multi-agent).
- **Nội dung:**
  - Cập nhật `pyproject.toml`: Kiểm soát phiên bản `langchain` và `langgraph` để tránh lỗi "dependency resolver".
  - Refactor các module bị ảnh hưởng bởi thay đổi phiên bản.

---

### 🕒 **2026-04-05 | 09:30 - 11:00 AM**

#### **Thay đổi: Nâng cấp quy trình huấn luyện và đánh giá mô hình**

- **Tiếp cận:** Xây dựng khung đánh giá (Evaluation Framework) chuẩn cho mô hình LSTM và RL Agent để đảm bảo độ tin cậy của dự đoán RUL.
- **Nội dung:**
  - `scripts/eval_utils.py` & `scripts/rl_eval.py`: Module hóa các tác vụ tính toán metric (MAE, RMSE) và mô phỏng đánh giá.
  - `demo_flow.ipynb`: Cập nhật Notebook với luồng công việc hoàn chỉnh từ Data Preprocessing, Training đến Post-training Evaluation.
  - `scripts/lstm_model.py`: Cải tiến kiến trúc lưu trữ mô hình và xử lý dữ liệu đầu vào.
- **Trạng thái:** Hoàn thành (Đã tích hợp vào luồng demo chính).

---

## 🛠 **Hướng dẫn vận hành (Rút gọn)**

Hệ thống hiện tại chạy trên kiến trúc **Client-Server**:

1. **Khởi động Backend (AI & Logic):**

   ```powershell
   .\.venv\Scripts\python.exe server.py
   ```

   _(Server sẽ chạy tại http://localhost:8000)_

2. **Khởi động Frontend (Dashboard):**
   ```powershell
   cd frontend
   npm run dev
   ```
   _(Truy cập http://localhost:3000 để bắt đầu điều khiển)_

---

_Lịch sử này sẽ được cập nhật khi có các thay đổi lớn tiếp theo._

---

### 🕒 **2026-04-06 | 12:10 - 12:25 AM**

#### **Thay đổi: Kiểm thử toàn hệ thống & Sửa lỗi (System Audit)**

- **Tiếp cận:** Rà soát toàn bộ mã nguồn (logic, cú pháp, tính thực tế) và áp dụng tất cả các bản sửa lỗi được xác định trong một phiên duy nhất.

- **Nội dung sửa lỗi (`scripts/rl_eval.py` — 2 lỗi nghiêm trọng):**
  - **BUG 1 (Runtime Crash):** Sửa tham chiếu thuộc tính `eval_env.total_steps` không tồn tại → thay bằng biến đếm `step_count` trong vòng lặp.
  - **BUG 2 (Metric sai):** Sửa chuỗi sự kiện `'LANDED'` không khớp với giá trị thực tế mà môi trường phát ra (`'ARRIVED'`, `'MAINTAINED'`). Điều này khiến tỷ lệ thành công luôn bằng 0%.

- **Nội dung sửa lỗi (`scripts/aircraft_env.py` — 4 vấn đề):**
  - **BUG 3 (Docstring sai):** Cập nhật tài liệu từ "2 hành động" lên "3 hành động" (`CRUISE`, `DESCEND`, `CLIMB`).
  - **LOGIC 1 (Quan sát vô nghĩa):** Cập nhật `twin.velocity` sau mỗi hành động để phản ánh tốc độ thực tế. Trước đây `obs[1]` luôn là `250.0` bất kể hành động nào.
  - **LOGIC 2 (Trạng thái không nhất quán):** Thêm `self.flight_phase = "CRUISING"` và `self.twin.velocity = self.V_CRUISE` vào `_reset_to_new_engine()`.
  - **LOGIC 3 (Dead code):** Thêm cập nhật `self.flight_phase` trong `step()` cho cả 3 hành động (`"CRUISING"` / `"DESCENDING"` / `"CLIMBING"`). Điều kiện trong `sim_2d.py` không còn là dead code nữa.
  - **LOGIC 4 (Cân bằng nhiên liệu):** Tăng `FUEL_CAPACITY` từ `100` → `200`. Với 200 chu kỳ CRUISE tối thiểu để hoàn thành hành trình, mức `100` không đủ biên độ an toàn khi agent leo cao.

- **Nội dung sửa lỗi (`scripts/sim_2d.py`):**
  - Sửa nhãn hành động hiển thị: thêm `"CLIMB"` cho `action == 2` (trước đây chỉ có `"FLY"`/`"LAND"`).

- **Trạng thái:** Hoàn thành. Tất cả 6 vấn đề (3 bugs + 4 logic) đã được áp dụng.

---

### 🕒 **2026-04-06 | 05:50 - 06:10 PM**

#### **Thay đổi: Tối ưu hóa hiệu năng huấn luyện PPO (Training Stability)**

- **Tiếp cận:** Cải thiện tính ổn định số học và khả năng hội tụ của thuật toán PPO bằng cách chuẩn hóa các thành phần của MDP (Markov Decision Process).

- **Nội dung (`scripts/aircraft_env.py`):**
  - **Reward Scaling (Tối quan trọng):** Chia toàn bộ thang điểm thưởng/phạt cho 100. Terminal reward giảm từ `±5000` xuống `±50`. Các phần thưởng theo bước (`step reward`) chuyển về mức `0.1` và `0.02`. 
  - **Mục tiêu:** Giảm `value_loss` từ mức bùng nổ (`10^5`) xuống mức ổn định để mạng Neural dễ dàng học được hàm giá trị mà không bị "sụp đổ" gradient.

- **Nội dung (`scripts/train_ppo.py`):**
  - **Tăng cường kinh nghiệm:** Tăng `total_timesteps` từ `100,000` lên **`500,000`**. 
  - **Lý do:** Với 4 nhân CPU song song (`SubprocVecEnv`), mỗi vòng rollout thu thập ~8k bước. Mốc 100k là quá ít để Agent học được các quy luật phức tạp của AircraftEnv (bay hành trình, hạ cánh, quản lý nhiên liệu).

- **Khuyến nghị bổ sung:** Đề xuất sử dụng `VecNormalize` trong Notebook để chuẩn hóa Observation (đầu vào) do sự chênh lệch lớn giữa các tham số (Altitude 12k vs Fuel 200).

- **Trạng thái:** Hoàn thành cập nhật mã nguồn Backend. Đang chờ cập nhật thủ công vào Notebook.

---

### 🕒 **2026-04-06 | 08:30 - 09:45 PM**

#### **Thay đổi: Nâng cấp môi trường & Chuẩn hóa PPO (Scaling & Normalization)**

- **Tiếp cận:** Mở rộng quy mô thử thách cho Agent bằng cách tăng gấp đôi khoảng cách hành trình và tích duy trì sự ổn định thông qua chuẩn hóa dữ liệu đầu vào.

- **Nội dung (`scripts/aircraft_env.py`):**
  - **Mở rộng quy mô (Environment Scaling):**
    - Tăng `TOTAL_DISTANCE` từ `5,000` → **`10,000`**.
    - Tăng `NUM_SUB_AIRPORTS` từ `3` → **`5`**.
  - **Ngẫu nhiên hóa (Randomized Airports):** 
    - Chuyển từ đặt sân bay cố định sang **Ngẫu nhiên hóa theo phân đoạn (Segmented Randomization)**. 
    - Lộ trình được chia làm 5 đoạn (2.000 đơn vị), mỗi đoạn đặt 1 sân bay ngẫu nhiên. Cách này đảm bảo bản đồ luôn mới lạ nhưng máy bay vẫn luôn có "trạm xăng" trong tầm bay cho phép.

- **Nội dung (`demo_flow.ipynb`):**
  - **Tích hợp `VecNormalize`:** Sửa lỗi logic block PPO để tự động lưu và nạp thông số chuẩn hóa (`_vec_normalize.pkl`). Điều này đảm bảo Agent nhận diện đúng tỷ lệ dữ liệu (Scaling) khi Inference.
  - **Tối ưu hóa thời gian học:** 
    - Xác định mốc **200,000 steps** là điểm dừng lý tưởng (thay vì 500k).
    - Kết quả thực nghiệm: Agent đạt **68.8/70** điểm thưởng (gần như tối đa) và rút ngắn thời gian bay xuống còn **213 cycles** (nhanh hơn 13% so với giai đoạn đầu).

- **Trạng thái:** Hoàn thành. Hệ thống đã sẵn sàng cho bài toán hành trình dài với sự linh hoạt cao.

---

### 🕒 **2026-04-06 | 10:15 - 10:45 PM**

#### **Thay đổi: Tích hợp MLflow và Cấu trúc hóa Evaluation (Experiment Tracking & Robustness)**

- **Tiếp cận:** Cải tiến quy trình theo dõi thí nghiệm, lưu trữ model và xử lý các lỗi khi thiết lập môi trường chuẩn hóa bằng MLflow và Custom Callbacks.

- **Nội dung (`scripts/rl_callbacks.py` - TẠO MỚI):**
  - **`SaveVecNormalizeCallback`:** Callback tuỳ chỉnh lắng nghe tín hiệu `EvalCallback` của SB3, đồng bộ hoá thông số chuẩn hóa `VecNormalize` lưu thành file `.pkl` ngay khi mô hình tốt nhất (Best Model) được tìm thấy. 
  - **`MLflowLoggingCallback`:** Callback đẩy tự động toàn bộ training metrics (loss, reward, fps, v.v.) từ `stable_baselines3` (sử dụng hàm `hasattr` kiểm tra `name_to_value` đảm bảo tương thích mọi version SB3).

- **Nội dung (`scripts/rl_eval.py` - FIX BUGS TÍNH CHẤT NỀN TẢNG):**
  - **Khắc phục lỗi "Mù dở đầu vào" (Blind Input Bug):** Thuật toán PPO được train qua bộ tiền xử lý `VecNormalize` nhưng lúc Evaluation lại bị truyền observation ở dạng thô (raw). Cải tiến hàm `evaluate_rl_agent` chấp nhận `stats_path` và thiết lập `DummyVecEnv` để map đúng thang đo dữ liệu chuẩn hóa.
  - **Sửa cấu trúc Output Vectorized Env:** Cập nhật toàn bộ logic xử lý step lấy `reward[0]`, `done_array[0]`, và chuyển fallback heuristic hành động theo dạng tham chiếu mảng của môi trường Vectorized (Ví dụ: `action=[0]`). Trích xuất `landed_rul` an toàn từ Info Object.

- **Nội dung (`demo_flow.ipynb`):**
  - Tích hợp logic đẩy artifact (đóng gói trực tiếp file nén `best_model.zip` và `vec_normalize.pkl`) lên Artifact Store của hệ thống **MLflow** tracking (`http://localhost:5000`).
  - Thiết lập luồng Evaluate chặt chẽ với `eval_freq=5000` liên kết các custom metrics trên tập 200,000 steps tối ưu.

- **Trạng thái:** Hoàn tất toàn diện. Dự án đã sẵn sàng cho Experiment Tracking và Machine Learning Operations (MLOps) chuyên nghiệp.
