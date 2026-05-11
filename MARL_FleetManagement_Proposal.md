# Proposal: Fleet Management - Multi-Agent Reinforcement Learning

## 1. Giới thiệu & Động lực
Thay vì điều khiển một máy bay duy nhất, chúng ta nâng cấp bài toán thành **Bài toán Quản lý Đội Bay (Fleet Management)**. Đây là một trong những lớp bài toán MARL phổ biến nhất trong công nghiệp, áp dụng trong logistics, taxi tự lái, giao thông thông minh...

Mỗi **Agent = 1 Máy bay** với một động cơ riêng biệt, độc lập hoạt động nhưng phải **cạnh tranh** tài nguyên bảo trì trên cùng một hành lang bay 20,000m.

---

## 2. Bối cảnh Bài toán

### Thiết lập
- **Số lượng Agent (Máy bay):** 3 đến 5 chiếc, bay **song song** trên cùng tuyến đường.
- **Hành lang bay:** 20,000 đơn vị, có **6 sân bay phụ** (slot bảo trì) và 1 đích đến.
- **Ràng buộc cốt lõi:** Mỗi sân bay phụ chỉ có **1 slot bảo trì duy nhất**. Nếu 2 máy bay cùng yêu cầu hạ cánh xuống cùng một sân bay trong cùng một bước, chỉ **chiếc có RUL thấp hơn** được ưu tiên. Chiếc còn lại bị từ chối và buộc phải tiếp tục bay.

### Tình huống xung đột điển hình
> Máy bay A (RUL=20) và Máy bay B (RUL=30) cùng tiếp cận Sân bay số 3. Cả hai đều muốn hạ cánh. Nếu cả hai cùng chọn `DESCEND`, Máy bay B sẽ bị từ chối slot, buộc phải leo lên lại và tìm Sân bay số 4. Máy bay B phải **tự học** được rằng: "Đối thủ A sắp chết hơn mình, mình nhường slot này và tính toán xem mình còn đủ RUL đến sân bay tiếp theo không."

---

## 3. Không gian Quan sát (Observation Space)

Mỗi Agent quan sát một vector gồm 2 phần:

### 3.1. Quan sát Cục bộ (Private - chỉ thấy bản thân)
| Biến | Ý nghĩa |
|---|---|
| `altitude` | Độ cao hiện tại |
| `velocity` | Vận tốc |
| `fuel` | Lượng nhiên liệu còn lại |
| `my_rul` | RUL dự báo của động cơ chính mình |
| `dist_airports[6]` | Khoảng cách signed tới 6 sân bay phụ |
| `dist_destination` | Khoảng cách tới đích |

### 3.2. Quan sát Toàn cục (Shared - thấy cả đội)
| Biến | Ý nghĩa |
|---|---|
| `other_positions[N-1]` | Vị trí hiện tại của các máy bay khác |
| `other_rul[N-1]` | RUL hiện tại của các máy bay bạn |
| `slot_occupied[6]` | Trạng thái bận/rỗi của từng sân bay phụ (0/1) |

> **Ghi chú về Communication:** Việc chia sẻ `other_rul` và `slot_occupied` chính là cơ chế giao tiếp ngầm (Implicit Communication). Không cần thiết kế channel giao tiếp riêng - Agent học cách diễn giải tín hiệu này.

---

## 4. Không gian Hành động (Action Space)
Giống hệt môi trường đơn Agent:
- `0: CRUISE` - Bay bằng
- `1: DESCEND` - Hạ độ cao (kích hoạt hạ cánh nếu đang qua sân bay)
- `2: CLIMB` - Leo lên

---

## 5. Cơ chế Xung đột Slot (Conflict Resolution)

Mỗi bước, môi trường xử lý xung đột theo thứ tự:
1. Tập hợp danh sách tất cả máy bay đang thực hiện `DESCEND` và đang trong phạm vi `LANDING_THRESHOLD` của một sân bay.
2. Với mỗi sân bay bị tranh chấp, **ưu tiên máy bay có RUL thấp nhất**.
3. Máy bay thắng: Đáp xuống, slot bị khóa trong N bước (thời gian bảo trì).
4. Máy bay thua: Nhận tín hiệu `REJECTED` trong observation. Bay tiếp, slot vẫn bị đánh dấu bận.
5. Nếu máy bay thua không tìm được sân bay tiếp theo kịp thời → RUL = 0 → Crash.

---

## 6. Hàm Phần thưởng (Reward Function)

### Phần thưởng Hợp tác (Shared)
| Sự kiện | Phần thưởng |
|---|---|
| Toàn đội về đích (tất cả ARRIVED hoặc MAINTAINED) | `+200` (chia đều) |
| Mỗi bước bay tiến lên | `+0.04` |
| Bất kỳ chiếc nào bị Crash | `-100` (toàn đội chịu phạt) |

### Phần thưởng Cá nhân (Individual)
| Sự kiện | Phần thưởng |
|---|---|
| Hạ cánh bảo trì thành công (cận kề giới hạn) | `10 + (150 - RUL) * 0.1` |
| Tự nguyện nhường slot cho người khó khăn hơn | `+5` (Social reward - khuyến khích chia sẻ) |
| Hạ cánh ngoài sân bay (Field Crash) | `-20` |
| Bị từ chối slot (không phạt, nhưng lãng phí bước) | `0` |

> **Social Reward:** Đây là yếu tố phân biệt bài toán hợp tác và cạnh tranh. Agent B tự nhường slot cho A sẽ được thưởng nhỏ (+5). Điều này dạy Agent học được hành vi "vị tha" - hy sinh lợi ích ngắn hạn của mình vì lợi ích chung của đội.

---

## 7. Lộ trình Triển khai (Roadmap)

### Phase 1: Xây dựng Multi-Agent Environment
- Viết lớp `FleetEnv` kế thừa từ `ParallelEnv` của thư viện **PettingZoo**.
- Tích hợp logic Conflict Resolution cho slot bảo trì.
- Cập nhật Observation Space với thông tin toàn cục.

### Phase 2: Nâng cấp Digital Twin
- Mỗi Agent trong môi trường có một instance `AircraftDigitalTwin` riêng.
- Quản lý trạng thái bảo trì: Sau khi đáp xuống, động cơ được "sửa" và RUL được reset về giá trị của một động cơ mới (bốc ngẫu nhiên từ dataset).

### Phase 3: Huấn luyện với IPPO / MAPPO
- **IPPO (Independent PPO):** Mỗi Agent chạy PPO riêng, nhưng nhìn thấy observation toàn cục. Đơn giản, dễ cài, là baseline tốt.
- **MAPPO (Multi-Agent PPO):** Dùng một Centralized Critic (bộ phê bình chung) để đánh giá chính sách của toàn đội. Phức tạp hơn nhưng hội tụ tốt hơn cho bài toán hợp tác.
- **Thư viện tham khảo:** `RLlib` (Ray), `MAPPO` (on-policy-baselines).

### Phase 4: Thử nghiệm & Phân tích
- So sánh chiến lược: IPPO vs. MAPPO vs. Heuristic (máy bay nào RUL thấp nhất thì ưu tiên).
- Phân tích hành vi: Chiếu bản đồ hành trình của cả đội, xem Agent có tự nhiên học được hành vi nhường slot không.

---

## 8. Kỳ vọng kết quả
Sau khi huấn luyện thành công, Agent sẽ biểu hiện các hành vi hợp tác nổi bật như:
- **Động cơ khỏe bay nhanh hơn (Cruise ở 12000m)** để giải phóng đường băng cho đồng đội.
- **Động cơ yếu chủ động Descend sớm** và không cạnh tranh slot với những chiếc có RUL thấp hơn.
- **Toàn đội tự phân phối** đều nhau để không 2 chiếc nào cùng đến 1 sân bay trong cùng thời điểm.
