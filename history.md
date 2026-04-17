[🏠 Home](./README.md) | [📜 History](./history.md)

# Project Evolution History

Dưới đây là lịch sử thay đổi và quá trình phát triển hệ thống **Aircraft Digital Twin - Predictive Maintenance** (được sắp xếp từ mới nhất đến cũ nhất).

---

### 🕒 **2026-04-17 | 11:30 - 11:55 PM**

#### **Thay đổi: Triển khai hạ tầng huấn luyện trên Google Cloud Platform (GCP)**

- **Tiếp cận:** Chuyển luồng huấn luyện từ máy cục bộ lên GCP để tận dụng tài nguyên tính toán mạnh mẽ hơn và bảo vệ phần cứng cá nhân. Sử dụng kiến trúc Dockerized để đảm bảo tính nhất quán.
- **Nội dung:**
  - **Containerization**: Tạo `Dockerfile` tối ưu cho môi trường RL (SB3, TensorFlow) và cập nhật `docker-compose.yaml` để quản lý cả MLflow và Training service.
  - **Infrastructure as Code (Terraform)**:
    - Bổ sung cấu hình cho máy ảo GCE (`e2-standard-4`) tại vùng `us-east1-b`.
    - Thiết lập Firewall mở cổng `5000` cho phép truy cập MLflow Dashboard từ xa.
    - Cấu hình `metadata_startup_script` để tự động cài đặt Docker và chuẩn bị môi trường khi VM khởi động.
  - **Automation Scripts**: 
    - Tạo `scripts/start_training_vm.ps1` và `scripts/stop_training_vm.ps1` để người dùng chủ động bật/tắt máy ảo nhằm tối ưu chi phí mà không làm mất dữ liệu.
  - **Logic Support**: Cập nhật `scripts/train_ppo.py` hỗ trợ biến môi trường `MLFLOW_TRACKING_URI` giúp Agent tự động kết nối đúng server MLflow trên Cloud.
- **Trạng thái:** Hoàn thành. Hệ thống đã sẵn sàng để triển khai thực tế.

---

### 🕒 **2026-04-10 | 12:30 - 02:00 PM**

#### **Thay đổi: Cấu trúc lại Environment & Thiết kế Reward Shaping (Dense Reward & Precision)**

- **Tiếp cận:** Giải quyết vấn đề Reward âm kéo dài và Agent "không chịu học" bằng cách cung cấp tín hiệu thưởng dày đặc (Dense Reward) và mở rộng khả năng quan sát (Observation Space) để Agent có cái nhìn toàn cảnh về bản đồ.
- **Nội dung (`scripts/aircraft_env.py` — Refactor quy mô lớn):**
  - **Mở rộng Observation Space (6D → 10D):** Thêm 5 giá trị **Signed Distances** tới các sân bay phụ.
  - **Thiết kế Reward Shaping (RUL × Proximity):** Thưởng liên tục dựa trên công thức: `Phần thưởng = Mức độ khẩn cấp (RUL thấp) × Độ gần sân bay (Proximity)`.
  - **Cân bằng Hành động (Action Neutralization):** Thiết lập base reward bằng nhau (`+0.05/step`) cho cả 3 hành động.
  - **Sửa lỗi Logic & Độ ổn định:** Fix Altitude, khởi tạo vật lý tường minh trong `reset()`, và ưu tiên Terminal logic.
- **Trạng thái:** Hoàn thành. Đã sẵn sàng cho đợt huấn luyện PPO Version 2.

---

### 🕒 **2026-04-07 | 03:20 - 03:50 PM**

#### **Thay đổi: Đảm bảo tính toàn vẹn của Metrics và Quản lý Experiment (MLflow Robustness)**

- **Tiếp cận:** Khắc phục triệt để hiện tượng mất metrics và lỗi crash khi quản lý Experiment trên MLflow. Chuyển đổi cơ chế từ "Polling Callback" sang "Logger Plugin".
- **Nội dung (`scripts/rl_callbacks.py`):**
  - **Chế tạo `MLflowOutputFormat` (KVWriter):** Cắm trực tiếp vào lõi của Stable-Baselines3 Logger để bắt 100% metrics trước khi bị xóa.
  - **Dọn dẹp cú pháp:** Tích hợp Regex để tự động loại bỏ các ký tự lạ trong Key của metric.
- **Nội dung (`demo_flow.ipynb`):** Xử lý "Deleted Experiment" Error.
- **Trạng thái:** Hoàn tất. Hệ thống Logging đã đạt độ tin cậy tuyệt đối.

---

### 🕒 **2026-04-06 | 10:15 - 10:45 PM**

#### **Thay đổi: Tích hợp MLflow và Cấu trúc hóa Evaluation (Experiment Tracking & Robustness)**

- **Tiếp cận:** Cải tiến quy trình theo dõi thí nghiệm, lưu trữ model và xử lý các lỗi khi thiết lập môi trường chuẩn hóa bằng MLflow và Custom Callbacks.
- **Nội dung:**
  - **`scripts/rl_callbacks.py` (TẠO MỚI)**: Tạo `SaveVecNormalizeCallback` và `MLflowLoggingCallback`.
  - **`scripts/rl_eval.py`**: Khắc phục lỗi "Mù dở đầu vào" (Blind Input Bug) bằng cách tích hợp stats của `VecNormalize`.
  - **`demo_flow.ipynb`**: Tích hợp logic đẩy artifact lên MLflow server.
- **Trạng thái:** Hoàn tất toàn diện. Dự án đã sẵn sàng cho MLOps chuyên nghiệp.

---

### 🕒 **2026-04-06 | 08:30 - 09:45 PM**

#### **Thay đổi: Nâng cấp môi trường & Chuẩn hóa PPO (Scaling & Normalization)**

- **Tiếp cận:** Mở rộng quy mô thử thách cho Agent bằng cách tăng gấp đôi khoảng cách hành trình và duy trì sự ổn định thông qua chuẩn hóa dữ liệu đầu vào.
- **Nội dung (`scripts/aircraft_env.py`):**
  - **Mở rộng quy mô**: Tăng `TOTAL_DISTANCE` lên `10,000` và `NUM_SUB_AIRPORTS` lên `5`.
  - **Ngẫu nhiên hóa**: Chuyển sang **Ngẫu nhiên hóa theo phân đoạn (Segmented Randomization)**.
- **Nội dung (`demo_flow.ipynb`):** Tích hợp `VecNormalize` và tối ưu hóa thời gian học (200k steps).
- **Trạng thái:** Hoàn thành. Hệ thống đã sẵn sàng cho bài toán hành trình dài.

---

### 🕒 **2026-04-06 | 05:50 - 06:10 PM**

#### **Thay đổi: Tối ưu hóa hiệu năng huấn luyện PPO (Training Stability)**

- **Tiếp cận:** Cải thiện tính ổn định số học và khả năng hội tụ của thuật toán PPO bằng cách chuẩn hóa các thành phần của MDP.
- **Nội dung:**
  - **Reward Scaling (`scripts/aircraft_env.py`)**: Chia thang điểm thưởng/phạt cho 100 để ổn định gradient.
  - **Tăng cường kinh nghiệm (`scripts/train_ppo.py`)**: Tăng `total_timesteps` lên `500,000`.
- **Trạng thái:** Hoàn thành cập nhật mã nguồn Backend.

---

### 🕒 **2026-04-06 | 12:10 - 12:25 AM**

#### **Thay đổi: Kiểm thử toàn hệ thống & Sửa lỗi (System Audit)**

- **Tiếp cận:** Rà soát toàn bộ mã nguồn và áp dụng tất cả các bản sửa lỗi xác định trong một phiên duy nhất.
- **Nội dung:**
  - **Sửa lỗi `scripts/rl_eval.py`**: Fix lỗi `total_steps` crash và mismatch event `'LANDED'`.
  - **Sửa lỗi `scripts/aircraft_env.py`**: Cập nhật docstring, velocity logic, và tăng `FUEL_CAPACITY` lên `200`.
  - **Sửa lỗi `scripts/sim_2d.py`**: Cập nhật nhãn hành động `CLIMB`.
- **Trạng thái:** Hoàn thành.

---

### 🕒 **2026-04-05 | 09:30 - 11:00 AM**

#### **Thay đổi: Nâng cấp quy trình huấn luyện và đánh giá mô hình**

- **Tiếp cận:** Xây dựng khung đánh giá (Evaluation Framework) chuẩn cho mô hình LSTM và RL Agent.
- **Nội dung:**
  - Tạo `scripts/eval_utils.py` & `scripts/rl_eval.py`.
  - Cập nhật `demo_flow.ipynb` với luồng workflow hoàn chỉnh.
- **Trạng thái:** Hoàn thành.

---

### 🕒 **2026-04-03 | 03:00 - 03:30 PM**

#### **Thay đổi: Giải quyết xung đột Dependency (LangGraph)**

- **Tiếp cận:** Điều chỉnh phiên bản thư viện trong môi trường Python để đảm bảo tính ổn định.
- **Nội dung:** Cập nhật `pyproject.toml` để kiểm soát phiên bản `langchain` và `langgraph`.

---

### 🕒 **2026-03-30 | 02:00 - 02:10 PM**

#### **Thay đổi: Xây dựng Dashboard Next.js (Premium UI)**

- **Tiếp cận:** Sử dụng Next.js, Tailwind CSS và Framer Motion để tạo một bảng điều khiển Dashboard chuyên nghiệp.
- **Nội dung:** Tạo `Dashboard.tsx`, `Environment2D.tsx` (SVG logic), và `useSimulation.ts`.

---

### 🕒 **2026-03-30 | 01:15 - 02:00 PM**

#### **Thay đổi: Chuyển đổi sang kiến trúc WebSockets & FastAPI**

- **Tiếp cận:** Tách biệt Logic mô phỏng (Backend) và Giao diện người dùng (Frontend).
- **Nội dung:** Tạo `server.py` sử dụng WebSockets để stream dữ liệu telemetry.

---

### 🕒 **2026-03-30 | 08:30 - 09:30 AM**

#### **Thay đổi: Tạo mô phỏng 2D đơn giản (Pygame)**

- **Tiếp cận:** Cung cấp công cụ trực quan hóa nhanh chóng để kiểm thử Logic môi trường Gymnasium.
- **Nội dung:** Tạo file `scripts/sim_2d.py`.
- **Trạng thái:** Đã lưu trữ (Dev tool).

---

## 🛠 **Hướng dẫn vận hành (Rút gọn)**

Hệ thống hiện tại chạy trên kiến trúc **Client-Server**:

1. **Khởi động Backend (AI & Logic):**
   ```powershell
   .\.venv\Scripts\python.exe server.py
   ```

2. **Khởi động Frontend (Dashboard):**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Huấn luyện trên Cloud (GCP):**
   ```powershell
   # Khởi động VM
   .\scripts\start_training_vm.ps1
   # Dừng VM sau khi xong
   .\scripts\stop_training_vm.ps1
   ```
