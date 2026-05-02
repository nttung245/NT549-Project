# -*- coding: utf-8 -*-
"""
Aircraft Environment Module (Gymnasium).
3-action PPO environment: Cruise, Descend, or Climb for altitude control.

The agent decides at each cycle:
  Action 0 (CRUISE):  Maintain altitude, move at cruise speed
  Action 1 (DESCEND): Lower altitude to approach/land at a sub-airport
  Action 2 (CLIMB):   Increase altitude (costs more fuel, same base reward)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from scripts.digital_twin import AircraftDigitalTwin
from scripts.data_processor import FEATURES, KEY_SENSORS, SEQUENCE_LENGTH


class AircraftEnv(gym.Env):
    """
    Gymnasium environment for aircraft predictive maintenance using PPO.

    Observation Space (10-dim):
        [Altitude, Velocity, Fuel, Current_RUL,
         SignedDist_Airport_1..5 (sorted by position, negative=behind),
         Dist_to_Destination]

        Airport distances are SIGNED relative to current position:
            Positive  → airport is still ahead
            Negative  → airport has been passed

    Action Space (Discrete 3):
        0 (CRUISE):  Maintain altitude, advance at cruise speed
        1 (DESCEND): Lower altitude toward a sub-airport or destination
        2 (CLIMB):   Increase altitude (higher fuel cost, same base reward)
    """

    metadata = {"render_modes": ["human"]}

    # ── Physical Constants ──────────────────────────────────────
    V_CRUISE = 25.0           # Tốc độ ngang khi bay bằng (m/s ~ đơn vị/cycle)
    V_DESCEND = 15.0          # Tốc độ ngang khi rà xuống (m/cycle)
    V_CLIMB = 20.0            # Tốc độ ngang khi leo cao (m/cycle)
    
    DESCEND_RATE = 500.0      # Tốc độ giảm độ cao (m/cycle) — giảm từ 1000 để hạ cánh từ từ hơn
    CLIMB_RATE = 1000.0       # Tốc độ tăng độ cao (m/cycle)
    MAX_ALTITUDE = 12000.0    # Độ cao tối đa cho phép
    
    LANDING_THRESHOLD = 500.0 # Tăng từ 300 → 500 để cửa sổ hạ cánh rộng hơn
    APPROACH_DISTANCE = 1500.0 # Khoảng cách để bắt đầu approach (dense reward zone)
    MAX_STEPS = 2000          # Tăng từ 1000 → 2000 (vì DESCEND_RATE giảm cần nhiều step hơn)
    FUEL_RATE = 0.5           # Nhiên liệu tiêu hao cơ bản mỗi cycle
    TOTAL_DISTANCE = 20000.0  # Tổng quãng đường bay (đơn vị)
    NUM_SUB_AIRPORTS = 6      # Số sân bay phụ trên đường bay
    FUEL_CAPACITY = 250.0     # Dung tích nhiên liệu tối đa

    def __init__(self, fleet_data: pd.DataFrame, model_path: str, scaler,
                 sensor_list: list = None, features_list: list = None):
        """
        Args:
            fleet_data: DataFrame huấn luyện (đã rolling + có cột RUL).
            model_path: Path to trained Keras LSTM/GRU model file.
            scaler: Fitted StandardScaler.
            sensor_list: List sensor columns cho LSTM.
            features_list: List tất cả feature columns cho scaler.
        """
        super(AircraftEnv, self).__init__()

        self.fleet_data = fleet_data
        self.model_path = model_path
        self.scaler = scaler
        self.sensor_list = sensor_list or KEY_SENSORS
        self.features_list = features_list or FEATURES

        # Tự động load model nội bộ để hỗ trợ chạy đa luồng (SubprocVecEnv)
        # Việc load ở đây đảm bảo mỗi subprocess có một bản sao riêng, không bị lỗi pickle trên Windows
        import tensorflow as tf
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            # Khai báo tf.function MỘT LẦN DUY NHẤT để tránh lỗi Retracing và tăng tốc x10
            @tf.function(reduce_retracing=True)
            def fast_predict(x):
                return self.model(x, training=False)
            self._fast_predict_fn = fast_predict
            
        except Exception as e:
            print(f"⚠️ Error loading model in environment: {e}")
            self.model = None
            self._fast_predict_fn = None

        # ── Spaces ──────────────────────────────────────────────
        # Action 0: CRUISE (Maintain altitude)
        # Action 1: DESCEND (Altitude -500/cycle)
        # Action 2: CLIMB (Altitude +1000/cycle)
        self.action_space = spaces.Discrete(3)

        # Observation Space 11-dim:
        # [Altitude, Fuel, RUL,
        #  SignedDist_AP1..AP6 (sorted by position; âm=đã qua, dương=phía trước),
        #  Dist_to_Destination,
        #  In_Approach_Zone (1 nếu đang trong APPROACH_DISTANCE của sân bay tiếp theo)]
        #
        # Ghi chú: Velocity được loại bỏ vì nó luôn cố định theo action (CRUISE/DESCEND/CLIMB).
        # Agent có thể suy ra Velocity từ action nó đã chọn.
        #
        # Với 6 signed distances, agent có đầy đủ thông tin để:
        # - Biết khoảng cách đến từng sân bay phía trước
        # - So sánh nhiên liệu hiện có vs khoảng cách để quyết định bỏ qua sân bay này hay không
        n_ap = self.NUM_SUB_AIRPORTS  # 6
        low = np.array(
            [0.0, 0.0, 0.0] +                         # Altitude, Fuel, RUL
            [-self.TOTAL_DISTANCE] * n_ap +            # SignedDist_AP1..6 (có thể âm)
            [0.0, 0.0],                               # Dist_Destination, In_Approach_Zone
            dtype=np.float32
        )
        high = np.array(
            [self.MAX_ALTITUDE, self.FUEL_CAPACITY, 300.0] +
            [self.TOTAL_DISTANCE] * n_ap +
            [self.TOTAL_DISTANCE, 1.0],               # Dist_Destination, In_Approach_Zone
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Internal state ──────────────────────────────────────
        self.twin = None
        # Chỉ lấy động cơ số 1 để huấn luyện giai đoạn đầu (đảm bảo đủ RUL để hoàn thành lộ trình)
        self.engine_units = [1]
        
        # Tối ưu hóa: Chuyển DataFrame sang Dictionary của NumPy arrays để lookup O(1)
        self.unit_data_map = {
            unit_id: group.sort_values('time_cycles')[self.features_list].values.astype(np.float32)
            for unit_id, group in fleet_data.groupby('unit_nr')
        }

        self.current_unit_idx = 0
        self.current_cycle_idx = 0
        self.current_engine_data = None
        self.distance_to_destination = self.TOTAL_DISTANCE
        self.sub_airports = []    # Vị trí các sân bay phụ (distance from origin)
        self.flight_phase = "CRUISING"  # Trạng thái bay: "CRUISING" hoặc "DESCENDING"

    # ================================================================
    # Reset
    # ================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Chọn ngẫu nhiên một engine unit từ dataset (Sử dụng lookup O(1))
        unit_id = self.np_random.choice(self.engine_units)
        self.current_engine_data = self.unit_data_map[unit_id]

        # Tạo Digital Twin
        self.twin = AircraftDigitalTwin(
            engine_id=unit_id, model=self.model, scaler=self.scaler,
            seq_length=SEQUENCE_LENGTH, features_list=self.features_list,
            sensor_list=self.sensor_list, fuel_capacity=self.FUEL_CAPACITY,
            predict_fn=self._fast_predict_fn
        )

        # Nạp các cycle đầu vào buffer trong twin (Sử dụng NumPy slicing - Cực nhanh)
        init_cycles = min(SEQUENCE_LENGTH, len(self.current_engine_data))
        if init_cycles > 0:
            # Gán trực tiếp mảng dữ liệu thay vì chạy vòng lặp
            self.twin.buffer[-init_cycles:] = self.current_engine_data[:init_cycles]
            self.twin.buffer_filled = init_cycles
        
        self.current_cycle_idx = init_cycles

        # Khởi tạo hành trình
        self.distance_to_destination = self.TOTAL_DISTANCE
        self.twin.fuel = self.FUEL_CAPACITY
        self.twin.altitude = 2000.0  # Bắt đầu trên không (tránh giai đoạn học cất cánh)
        self.twin.velocity = self.V_CRUISE
        self.flight_phase = "CRUISING"  # Bắt đầu đã ở trạng thái CRUISING
        self.current_step = 0
        self.dist_since_last_maintenance = 0.0  # Khoảng cách đã bay kể từ lần bảo trì gần nhất

        # Chia lộ trình thành các phân đoạn, đặt ngẫu nhiên sân bay (giới hạn độ lệch để tránh khoảng cách quá lớn)
        base_points = [2857.0, 5714.0, 8571.0, 11428.0, 14285.0, 17142.0]
        self.sub_airports = []
        for base in base_points:
            # Noise ngẫu nhiên từ -200 đến +200 xung quanh mốc
            noise = self.np_random.uniform(-200.0, 200.0)
            self.sub_airports.append(base + noise)
        
        # Sort theo vị trí để observation index nhất quán
        self.sub_airports.sort()

        obs = self._get_obs()
        return obs, {}

    # ================================================================
    # Step
    # ================================================================
    def step(self, action):
        self.current_step += 1
        done = False
        truncated = False
        reward = 0.0
        info = {}

        # 1. Cập nhật dữ liệu cảm biến từ NASA Dataset (tăng 1 cycle)
        new_row = self._get_next_sensor_row()
        if new_row is not None:
            self.twin.update_sensor_data(new_row)

        # 2. Dự báo RUL từ Digital Twin
        current_rul = self.twin.predict_status()
        if current_rul is None:
            current_rul = 150.0  # Default khi chưa đủ dữ liệu

        # Kiểm tra xem máy bay có đang ở trên không không
        was_in_air = self.twin.altitude > 0

        # 3. Xử lý Hành động
        if action == 0:
        # ── CRUISE: Giữ độ cao, Tốc độ cao ──
            self.flight_phase = "CRUISING"
            self.twin.velocity = self.V_CRUISE
            distance_covered = self.V_CRUISE
            fuel_spent = self.FUEL_RATE

        elif action == 1:
        # ── DESCEND: Hạ độ cao, Tốc độ thấp ──
            self.flight_phase = "DESCENDING"
            self.twin.altitude -= self.DESCEND_RATE
            self.twin.velocity = self.V_DESCEND
            distance_covered = self.V_DESCEND
            fuel_spent = self.FUEL_RATE

        elif action == 2:
        # ── CLIMB: Tăng độ cao, Tốc độ trung bình, Tốn xăng hơn ──
            self.flight_phase = "CLIMBING"
            self.twin.altitude += self.CLIMB_RATE
            self.twin.velocity = self.V_CLIMB
            distance_covered = self.V_CLIMB
            fuel_spent = self.FUEL_RATE * 1.5

        # Base step reward khuyến khích tiến lên (Reward Shaping)
        reward += (distance_covered / 1000.0)
        reward -= 0.01  # Time penalty

        # 4. Cập nhật trạng thái vật lý
        self.distance_to_destination -= distance_covered
        self.dist_since_last_maintenance += distance_covered  # Cộng dồn quãng đường bay
        self.twin.fuel -= fuel_spent
        self.twin.fuel = max(0.0, self.twin.fuel)
        self.twin.altitude = np.clip(self.twin.altitude, 0, self.MAX_ALTITUDE)

        # Vị trí hiện tại SAU khi di chuyển
        current_pos = self.TOTAL_DISTANCE - self.distance_to_destination

        # Tìm sân bay phụ tiếp theo (phía trước) và khoảng cách tuyệt đối đến sân bay gần nhất
        dist_nearest_abs = float('inf')
        dist_to_next_target = self.distance_to_destination  # Default: hướng về đích
        if self.sub_airports:
            dist_nearest_abs = min(abs(ap - current_pos) for ap in self.sub_airports)
            # Tìm sân bay phụ gần nhất phía trước
            ahead_airports = [ap for ap in self.sub_airports if ap > current_pos]
            if ahead_airports:
                next_ap = min(ahead_airports)
                dist_to_next_target = next_ap - current_pos

        # ── Dense Approach Reward: Thưởng khi tiếp cận và hạ cánh đúng cách ──
        # Khi trong vùng tiếp cận (APPROACH_DISTANCE), thưởng DESCEND, phạt CLIMB
        in_approach_zone = dist_to_next_target <= self.APPROACH_DISTANCE
        if was_in_air and in_approach_zone:
            if action == 1:  # DESCEND trong approach zone → thưởng mạnh
                approach_reward = 0.5 * (1.0 - dist_to_next_target / self.APPROACH_DISTANCE)
                reward += approach_reward
            elif action == 2:  # CLIMB khi đang gần sân bay → phạt nhẹ
                reward -= 0.15

        # 6. KIỂM TRA ĐÁP ĐẤT (Chỉ kiểm tra nếu vừa đáp từ trên không xuống)
        if was_in_air and self.twin.altitude <= 0:
            at_airport = dist_nearest_abs <= self.LANDING_THRESHOLD
            at_destination = self.distance_to_destination <= self.LANDING_THRESHOLD

            if at_airport or at_destination:
                if at_destination:
                    # Đến đích thành công!
                    reward += 100.0
                    done = True
                    info['event'] = "ARRIVED"
                else:
                    # Hạ cánh bảo trì tại sân bay phụ
                    # Thưởng tỷ lệ với quãng đường đã bay từ lần bảo trì trước
                    # → Ngăn Agent "farm" bằng cách climb/descend tại chỗ (chỉ ~35m)
                    # → Khuyến khích bay đủ xa rồi mới bảo trì
                    progress_bonus = min(self.dist_since_last_maintenance / 100.0, 30.0)  # 0 đến +30
                    rul_bonus = max(0.0, (80.0 - current_rul) * 0.3)         # 0 đến +24 (khi RUL < 80)
                    fuel_bonus = max(0.0, (1.0 - self.twin.fuel / self.FUEL_CAPACITY) * 15.0)  # 0 đến +15
                    maintenance_reward = progress_bonus + rul_bonus + fuel_bonus  # Luôn >= 0
                    reward += maintenance_reward

                    self.dist_since_last_maintenance = 0.0  # Reset khoảng cách sau mỗi lần bảo trì
                    self._reset_to_new_engine()
                    # Snap vị trí về sân bay phụ đã hạ cánh
                    landed_ap = min(self.sub_airports, key=lambda p: abs(p - current_pos))
                    self.distance_to_destination = self.TOTAL_DISTANCE - landed_ap

                    info['event'] = "MAINTAINED"
                    info['rul_at_landing'] = current_rul
            else:
                # Rớt máy bay (Hạ cánh giữa đường)
                reward -= 40.0
                done = True
                info['event'] = "FIELD_CRASH"

        # 7. KIỂM TRA TỬ VONG (Hết Nhiên Liệu hoặc RUL = 0)
        # Chỉ kiểm tra nếu chưa hoàn thành (ví dụ đã hạ cánh thành công)
        if not done:
            if self.twin.fuel <= 0:
                reward -= 40.0
                done = True
                info['event'] = "FUEL_EMPTY"

            elif current_rul <= 0:
                reward -= 40.0
                done = True
                info['event'] = "CRASHED"

        # Kiểm tra timeout (vượt quá MAX_STEPS) — phạt nặng để ngăn agent "lười"
        if self.current_step >= self.MAX_STEPS:
            reward -= 20.0
            truncated = True
            info['event'] = "TIMEOUT"

        # Trả về Observation mới
        obs = self._get_obs()
        return obs, reward, done, truncated, info

    # ================================================================
    # Helpers
    # ================================================================
    def _get_obs(self) -> np.ndarray:
        """
        Trả về observation vector 11 chiều:
        [Altitude, Fuel, RUL,
         SignedDist_AP1..AP6 (sorted, âm=đã qua, dương=phía trước),
         Dist_to_Destination,
         In_Approach_Zone (1.0 nếu đang gần sân bay tiếp theo)]

        Với 6 signed distances, agent có thể thấy TẤT CẢ các sân bay đang ở đâu so với nó,
        từ đó suy luận về nhiên liệu để quyết định bỏ qua hay dừng bảo trì.
        """
        current_rul = self.twin.current_rul if self.twin.current_rul is not None else 150.0
        current_pos = self.TOTAL_DISTANCE - self.distance_to_destination

        # Signed distances: dương = sân bay phía trước, âm = đã bay qua
        airport_dists = [ap - current_pos for ap in self.sub_airports]

        # Tính dist_to_next_target để xác định approach zone
        dist_to_next_target = max(0.0, self.distance_to_destination)
        ahead_airports = [ap for ap in self.sub_airports if ap > current_pos]
        if ahead_airports:
            dist_to_next_target = min(ahead_airports) - current_pos

        in_approach_zone = 1.0 if dist_to_next_target <= self.APPROACH_DISTANCE else 0.0

        return np.array([
            self.twin.altitude,
            self.twin.fuel,
            current_rul,
            *airport_dists,                      # 6 giá trị signed
            max(0.0, self.distance_to_destination),
            in_approach_zone,
        ], dtype=np.float32)

    def _dist_to_nearest_airport(self) -> float:
        """Khoảng cách tuyệt đối đến sân bay gần nhất (cả phía trước và phía sau)."""
        current_pos = self.TOTAL_DISTANCE - self.distance_to_destination
        if not self.sub_airports:
            return max(0.0, self.distance_to_destination)
        return min(abs(ap - current_pos) for ap in self.sub_airports)

    def _get_next_sensor_row(self):
        """Lấy dòng cảm biến tiếp theo từ dataset hiện tại (Đã tối ưu NumPy)."""
        if self.current_cycle_idx < len(self.current_engine_data):
            row = self.current_engine_data[self.current_cycle_idx]
            self.current_cycle_idx += 1
            return row
        return None  # Hết dữ liệu cho engine hiện tại

    def _reset_to_new_engine(self):
        """
        Giả lập bảo trì: chọn engine mới từ dataset (Lookup O(1)),
        nạp lại các cycle đầu vào buffer bằng NumPy slicing.
        """
        # Chọn engine mới từ dataset
        unit_id = self.np_random.choice(self.engine_units)
        self.current_engine_data = self.unit_data_map[unit_id]

        # Reset twin buffer
        self.twin.buffer.fill(0)
        self.twin.buffer_filled = 0
        self.twin.engine_id = unit_id
        self.twin.current_rul = None
        self.twin.status = "HEALTHY"
        self.twin.fuel = self.FUEL_CAPACITY
        self.twin.altitude = 2000.0  # Sau bảo trì, khởi động lại ở độ cao 2000m (giống reset())
        self.twin.velocity = self.V_CRUISE
        self.flight_phase = "CRUISING"

        # Nạp cycle đầu (engine mới = khỏe mạnh) bằng NumPy slicing
        init_cycles = min(SEQUENCE_LENGTH, len(self.current_engine_data))
        if init_cycles > 0:
            self.twin.buffer[-init_cycles:] = self.current_engine_data[:init_cycles]
            self.twin.buffer_filled = init_cycles
        self.current_cycle_idx = init_cycles
