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

## 🛠 **Hướng dẫn vận hành (Rút gọn)**

Hệ thống hiện tại chạy trên kiến trúc **Client-Server**:

1. **Khởi động Backend (AI & Logic):**
   ```powershell
   .\.venv\Scripts\python.exe server.py
   ```
   *(Server sẽ chạy tại http://localhost:8000)*

2. **Khởi động Frontend (Dashboard):**
   ```powershell
   cd frontend
   npm run dev
   ```
   *(Truy cập http://localhost:3000 để bắt đầu điều khiển)*

---
*Lịch sử này sẽ được cập nhật khi có các thay đổi lớn tiếp theo.*
