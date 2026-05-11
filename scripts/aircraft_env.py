# -*- coding: utf-8 -*-
"""
Aircraft Environment Module (Gymnasium).
3-action PPO environment: Cruise, Descend, or Climb for altitude control.

The agent decides at each cycle:
  Action 0 (CRUISE):  Maintain altitude, move at cruise speed
  Action 1 (DESCEND): Lower altitude to approach/land at a sub-airport
  Action 2 (CLIMB):   Increase altitude (costs more fuel, same base reward)
"""

from typing import Any, Callable, cast

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
    
    DESCEND_RATE = 500.0      # Tốc độ giảm độ cao (m/cycle)
    CLIMB_RATE = 500.0        # Giảm từ 1000 để agent không nhảy altitude quá cao
    MAX_ALTITUDE = 5000.0     # Curriculum dễ hơn: tránh agent leo quá cao và khó hạ cánh
    
    LANDING_THRESHOLD = 400.0 # Đủ rộng cho quãng đường lướt ~150m khi hạ từ 5000m
    APPROACH_DISTANCE = 800.0 # Vùng nhận approach reward tương ứng với cửa sổ hạ cánh
    MAX_STEPS = 2000          # Tăng từ 1000 → 2000 (vì DESCEND_RATE giảm cần nhiều step hơn)
    FUEL_RATE = 0.5           # Nhiên liệu tiêu hao cơ bản mỗi cycle
    TOTAL_DISTANCE = 20000.0  # Tổng quãng đường bay (đơn vị)
    NUM_SUB_AIRPORTS = 6      # Số sân bay phụ trên đường bay
    FUEL_CAPACITY = 250.0     # Dung tích nhiên liệu tối đa

    def __init__(self, fleet_data: pd.DataFrame, model_path: str, scaler,
                 sensor_list: list[str] | None = None,
                 features_list: list[str] | None = None,
                 eligible_units: list[int] | None = None,
                 min_initial_rul: float = 120.0,
                 maintenance_resets_health: bool = True):
        """
        Args:
            fleet_data: DataFrame huấn luyện/đánh giá (đã rolling + có cột RUL).
            model_path: Path to trained Keras LSTM/GRU model file.
            scaler: Fitted StandardScaler.
            sensor_list: List sensor columns cho LSTM.
            features_list: List tất cả feature columns cho scaler.
            eligible_units: Optional explicit engine IDs to sample from.
            min_initial_rul: Minimum RUL at the initial LSTM window. Units below
                this threshold are excluded to avoid impossible starting episodes.
            maintenance_resets_health: If True, a successful intermediate
                maintenance landing refuels and swaps/resets the engine health.
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
            self.model: Any | None = tf.keras.models.load_model(model_path)
            model = cast(Callable[..., Any], self.model)
            
            # Khai báo tf.function MỘT LẦN DUY NHẤT để tránh lỗi Retracing và tăng tốc x10
            @tf.function(reduce_retracing=True)
            def fast_predict(x):
                return model(x, training=False)
            self._fast_predict_fn: Callable[..., Any] | None = fast_predict
            
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
        self.min_initial_rul = float(min_initial_rul)
        self.maintenance_resets_health = bool(maintenance_resets_health)
        self.unit_rul_map: dict[int, np.ndarray] = {}
        self.unit_initial_rul_map: dict[int, float] = {}
        self.unit_data_map: dict[int, np.ndarray] = {}
        
        # Tối ưu hóa: Chuyển DataFrame sang Dictionary của NumPy arrays để lookup O(1)
        for raw_unit_id, group in fleet_data.groupby('unit_nr'):
            unit_id = int(cast(Any, raw_unit_id))
            sorted_group = group.sort_values('time_cycles')
            self.unit_data_map[unit_id] = np.asarray(
                sorted_group[self.features_list].values,
                dtype=np.float32,
            )
            if 'RUL' in sorted_group.columns:
                self.unit_rul_map[unit_id] = np.asarray(
                    sorted_group['RUL'].values,
                    dtype=np.float32,
                )

        all_units = sorted(self.unit_data_map.keys())
        candidate_units = [int(unit_id) for unit_id in eligible_units] if eligible_units is not None else all_units
        self.engine_units = self._filter_engine_units(candidate_units)
        if not self.engine_units:
            raise ValueError(
                f"No eligible engine units after filtering with min_initial_rul={self.min_initial_rul}. "
                "Lower min_initial_rul or pass explicit eligible_units."
            )
      
        self.current_unit_idx = 0
        self.current_cycle_idx = 0
        self.current_engine_data = None
        self.distance_to_destination = self.TOTAL_DISTANCE
        self.sub_airports = []    # Vị trí các sân bay phụ (distance from origin)
        self.flight_phase = "CRUISING"  # Trạng thái bay: "CRUISING" hoặc "DESCENDING"
        self.pending_reset = False  # Flag để defer reset sau khi landing

    # ================================================================
    # Reset
    # ================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Chọn ngẫu nhiên một engine unit đủ RUL từ dataset (Sử dụng lookup O(1))
        unit_id = self.engine_units[int(self.np_random.integers(len(self.engine_units)))]
        twin = self._build_twin_for_unit(unit_id)
        self.twin = twin

        # Khởi tạo hành trình
        self.distance_to_destination = self.TOTAL_DISTANCE
        twin.fuel = self.FUEL_CAPACITY
        twin.altitude = 2000.0  # Bắt đầu trên không (tránh giai đoạn học cất cánh)
        twin.velocity = self.V_CRUISE
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
        action_int = int(action)
        self.current_step += 1
        done = False
        truncated = False
        reward = 0.0
        reward_components = {}
        info: dict[str, Any] = {"action": action_int}
   
        if self.twin is None:
            raise RuntimeError("Environment must be reset before calling step().")
   
        twin = self.twin

        # 1. Cập nhật dữ liệu cảm biến từ NASA Dataset (tăng 1 cycle)
        new_row = self._get_next_sensor_row()
        if new_row is not None:
            twin.update_sensor_data(new_row)

        # 2. Dự báo RUL từ Digital Twin
        current_rul = twin.predict_status()
        if current_rul is None:
            current_rul = 150.0  # Default khi chưa đủ dữ liệu

        # Kiểm tra xem máy bay có đang ở trên không không
        was_in_air = twin.altitude > 0
        is_grounded = not was_in_air

        # Tính toán hiệu suất bay dựa trên độ cao hiện tại
        # Càng cao bay càng nhanh và càng tiết kiệm nhiên liệu
        altitude_ratio = twin.altitude / self.MAX_ALTITUDE
        current_v_cruise = 25.0 + altitude_ratio * 15.0  # 25.0 -> 40.0
        current_fuel_rate = 0.5 - altitude_ratio * 0.2   # 0.5 -> 0.3

        # 3. Xử lý Hành động
        if is_grounded and action_int in (0, 1):
            # Sau khi bảo trì/hạ cánh, CRUISE/DESCEND trên mặt đất là hành động không hợp lệ.
            # Không cho máy bay trượt dọc đường băng vô hạn; chỉ CLIMB mới tiếp tục hành trình.
            self.flight_phase = "GROUNDED"
            self.twin.velocity = 0.0
            distance_covered = 0.0
            fuel_spent = 0.0
            invalid_ground_action_penalty = -0.25 if action_int == 1 else -0.10
            reward += invalid_ground_action_penalty
            reward_components["invalid_ground_action"] = invalid_ground_action_penalty

        elif action_int == 0:
        # ── CRUISE: Giữ độ cao, Tốc độ tỷ lệ với độ cao ──
            self.flight_phase = "CRUISING"
            self.twin.velocity = current_v_cruise
            distance_covered = current_v_cruise
            fuel_spent = current_fuel_rate

        elif action_int == 1:
        # ── DESCEND: Hạ độ cao, Tốc độ thấp ──
            self.flight_phase = "DESCENDING"
            self.twin.altitude -= self.DESCEND_RATE
            self.twin.velocity = self.V_DESCEND
            distance_covered = self.V_DESCEND
            fuel_spent = current_fuel_rate

        elif action_int == 2:
        # ── CLIMB: Tăng độ cao, Tốc độ trung bình, Tốn xăng hơn ──
            self.flight_phase = "CLIMBING"
            self.twin.altitude += self.CLIMB_RATE
            self.twin.velocity = self.V_CLIMB
            distance_covered = self.V_CLIMB
            fuel_spent = current_fuel_rate * 1.5

        else:
            raise ValueError(f"Invalid action: {action_int}")

        # Base step reward khuyến khích tiến lên (Reward Shaping).
        # Không còn thưởng cruise theo altitude: reward này từng khuyến khích agent
        # leo cao quá mức, làm timing DESCEND khó học dù threshold sân bay khá rộng.
        progress_reward = distance_covered / 1500.0
        altitude_efficiency_reward = 0.0
        time_penalty = -0.01
        reward += progress_reward + altitude_efficiency_reward + time_penalty
        reward_components["progress"] = progress_reward
        reward_components["altitude_efficiency"] = altitude_efficiency_reward
        reward_components["time"] = time_penalty

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
        altitude_after_action = float(twin.altitude)
        descent_steps_remaining = int(np.ceil(altitude_after_action / self.DESCEND_RATE)) if altitude_after_action > 0 else 0
        descent_distance_needed = descent_steps_remaining * self.V_DESCEND
        landing_feasible_now = dist_to_next_target <= (descent_distance_needed + self.LANDING_THRESHOLD)

        if was_in_air and in_approach_zone:
            if action_int == 1 and landing_feasible_now:
                # Reward only feasible descent timing, not blind descent inside the zone.
                # Tăng reward để tín hiệu "DESCEND đúng thời điểm" thắng incentive cruise/progress.
                approach_reward = 0.75 * (1.0 - dist_to_next_target / self.APPROACH_DISTANCE)
                reward += approach_reward
                reward_components["approach"] = approach_reward
            elif action_int == 1 and not landing_feasible_now:
                early_descent_penalty = -0.08
                reward += early_descent_penalty
                reward_components["early_descent"] = early_descent_penalty
            elif action_int == 2:
                late_climb_penalty = -0.05
                reward += late_climb_penalty
                reward_components["late_climb"] = late_climb_penalty

        # 6. KIỂM TRA ĐÁP ĐẤT (Chỉ kiểm tra nếu vừa đáp từ trên không xuống)
        if was_in_air and self.twin.altitude <= 0:
            # Chỉ cho phép đáp nếu đã bay đủ xa (tránh farm điểm tại chỗ)
            valid_flight = self.dist_since_last_maintenance >= 1000.0
            at_airport = (dist_nearest_abs <= self.LANDING_THRESHOLD) and valid_flight
            at_destination = self.distance_to_destination <= self.LANDING_THRESHOLD

            if at_airport or at_destination:
                if at_destination:
                    # Đến đích thành công!
                    reward += 150.0
                    reward_components["arrived"] = 150.0
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
                    landing_accuracy_bonus = max(0.0, (1.0 - dist_nearest_abs / self.LANDING_THRESHOLD) * 10.0)
                    maintenance_reward = progress_bonus + rul_bonus + fuel_bonus + landing_accuracy_bonus  # Luôn >= 0
                    reward += maintenance_reward
                    reward_components["maintenance"] = maintenance_reward
            
                    # Snap vị trí về sân bay phụ đã hạ cánh
                    landed_ap = min(self.sub_airports, key=lambda p: abs(p - current_pos))
                    self.distance_to_destination = self.TOTAL_DISTANCE - landed_ap
            
                    info['event'] = "MAINTAINED"
                    info['rul_at_landing'] = current_rul
            
                    # Intermediate maintenance is a checkpoint: do not terminate.
                    # Refuel and, by default, swap/reset to a healthy engine so RUL
                    # becomes sufficient for the next route segment.
                    if self.maintenance_resets_health:
                        replacement_unit = self.engine_units[int(self.np_random.integers(len(self.engine_units)))]
                        twin = self._build_twin_for_unit(replacement_unit)
                        twin.altitude = 0.0
                        twin.velocity = 0.0
                        self.twin = twin
                        info['maintenance_replacement_unit'] = replacement_unit
                        current_rul = float(twin.current_rul if twin.current_rul is not None else 150.0)
                    else:
                        twin.altitude = 0.0
                        twin.velocity = 0.0
                        twin.fuel = self.FUEL_CAPACITY
                    self.flight_phase = "GROUNDED"
                    done = False
                    
                    # Reset maintenance distance counter
                    self.dist_since_last_maintenance = 0.0
            else:
                # Rớt máy bay (Hạ cánh giữa đường)
                field_crash_penalty = -60.0
                reward += field_crash_penalty
                reward_components["field_crash"] = field_crash_penalty
                done = True
                info['event'] = "FIELD_CRASH"

        # 7. KIỂM TRA TỬ VONG (Hết Nhiên Liệu hoặc RUL = 0)
        # Chỉ kiểm tra nếu chưa hoàn thành (ví dụ đã hạ cánh thành công).
        # RUL chỉ làm crash khi máy bay đang bay; sau MAINTAINED máy bay có thể chờ CLIMB ở sân bay.
        if not done:
            if self.twin.fuel <= 0:
                fuel_empty_penalty = -60.0
                reward += fuel_empty_penalty
                reward_components["fuel_empty"] = fuel_empty_penalty
                done = True
                info['event'] = "FUEL_EMPTY"

            elif current_rul <= 0 and self.flight_phase != "GROUNDED":
                crashed_penalty = -60.0
                reward += crashed_penalty
                reward_components["crashed"] = crashed_penalty
                done = True
                info['event'] = "CRASHED"

        # Kiểm tra timeout (vượt quá MAX_STEPS) — phạt nặng để ngăn agent "lười"
        if self.current_step >= self.MAX_STEPS:
            timeout_penalty = -30.0
            reward += timeout_penalty
            reward_components["timeout"] = timeout_penalty
            truncated = True
            info['event'] = "TIMEOUT"

        info.update({
            "reward_components": reward_components,
            "altitude": float(twin.altitude),
            "fuel": float(twin.fuel),
            "rul": float(current_rul),
            "current_pos": float(current_pos),
            "distance_to_destination": float(self.distance_to_destination),
            "dist_to_next_target": float(dist_to_next_target),
            "dist_nearest_airport": float(dist_nearest_abs),
            "in_approach_zone": bool(in_approach_zone),
            "landing_feasible_now": bool(landing_feasible_now),
            "descent_steps_remaining": int(descent_steps_remaining),
            "descent_distance_needed": float(descent_distance_needed),
            "flight_phase": self.flight_phase,
        })

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
        if self.twin is None:
            raise RuntimeError("Environment must be reset before observations are requested.")

        twin = self.twin
        current_rul = twin.current_rul if twin.current_rul is not None else 150.0
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
            twin.altitude,
            twin.fuel,
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
        current_engine_data = self.current_engine_data
        if current_engine_data is None:
            return None

        if self.current_cycle_idx < len(current_engine_data):
            row = current_engine_data[self.current_cycle_idx]
            self.current_cycle_idx += 1
            return row
        return None  # Hết dữ liệu cho engine hiện tại

    def _filter_engine_units(self, candidate_units: list[int]) -> list[int]:
        """Return units whose initial window has enough true RUL for a fair start."""
        eligible: list[int] = []
        for unit_id in candidate_units:
            if unit_id not in self.unit_data_map:
                continue
            rul_values = self.unit_rul_map.get(unit_id)
            if rul_values is None or len(rul_values) == 0:
                eligible.append(unit_id)
                self.unit_initial_rul_map[unit_id] = float('nan')
                continue
            init_idx = min(SEQUENCE_LENGTH, len(rul_values)) - 1
            initial_rul = float(rul_values[max(init_idx, 0)])
            self.unit_initial_rul_map[unit_id] = initial_rul
            if initial_rul >= self.min_initial_rul:
                eligible.append(unit_id)
        return eligible

    def _build_twin_for_unit(self, unit_id: int) -> AircraftDigitalTwin:
        """Create/reset the digital twin and preload the initial healthy window."""
        current_engine_data = self.unit_data_map[unit_id]
        self.current_engine_data = current_engine_data

        twin = AircraftDigitalTwin(
            engine_id=unit_id, model=self.model, scaler=self.scaler,
            seq_length=SEQUENCE_LENGTH, features_list=self.features_list,
            sensor_list=self.sensor_list, fuel_capacity=self.FUEL_CAPACITY,
            predict_fn=self._fast_predict_fn
        )

        init_cycles = min(SEQUENCE_LENGTH, len(current_engine_data))
        if init_cycles > 0:
            twin.buffer[-init_cycles:] = current_engine_data[:init_cycles]
            twin.buffer_filled = init_cycles
        self.current_cycle_idx = init_cycles
        twin.current_rul = self.unit_initial_rul_map.get(unit_id)
        twin.status = "HEALTHY"
        twin.fuel = self.FUEL_CAPACITY
        twin.altitude = 2000.0
        twin.velocity = self.V_CRUISE
        return twin

    def _reset_to_new_engine(self):
        """
        Giả lập bảo trì: chọn engine mới từ dataset (Lookup O(1)),
        nạp lại các cycle đầu vào buffer bằng NumPy slicing.

        Lưu ý: helper này không được gọi cho hạ cánh bảo trì trung gian trong
        step(), vì reset engine ở đó sẽ làm RUL nhảy lại như episode mới và
        phá tính liên tục của hành trình. Nó được giữ lại cho các thử nghiệm
        muốn mô phỏng thay động cơ riêng biệt.
        """
        if self.twin is None:
            raise RuntimeError("Environment must be reset before resetting to a new engine.")

        twin = self.twin

        # Chọn engine mới từ dataset
        unit_id = self.engine_units[int(self.np_random.integers(len(self.engine_units)))]
        new_twin = self._build_twin_for_unit(unit_id)
        twin.buffer = new_twin.buffer
        twin.buffer_filled = new_twin.buffer_filled
        twin.engine_id = new_twin.engine_id
        twin.current_rul = new_twin.current_rul
        twin.status = new_twin.status
        twin.fuel = new_twin.fuel
        twin.altitude = new_twin.altitude
        twin.velocity = new_twin.velocity
        self.flight_phase = "CRUISING"
