# -*- coding: utf-8 -*-
"""
Aircraft Environment Module (Gymnasium).
2-action PPO environment: Fly or Land for Maintenance.

The agent decides at each cycle:
  Action 0 (Fly):  Continue toward destination
  Action 1 (Land): Land at nearest sub-airport for maintenance
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

    Observation Space (6-dim):
        [Altitude, Velocity, Fuel, Current_RUL, Dist_to_Next_Airport, Dist_to_Destination]

    Action Space (Discrete 2):
        0: Continue flying toward destination
        1: Land at nearest sub-airport for maintenance
    """

    metadata = {"render_modes": ["human"]}

    # ── Physical Constants ──────────────────────────────────────
    V_CRUISE = 250.0          # Vận tốc bay đường trường (m/s ~ đơn vị/cycle)
    FUEL_RATE = 0.1           # Nhiên liệu tiêu hao mỗi cycle
    SAFE_RUL = 30             # Ngưỡng RUL an toàn tâm lý
    TOTAL_DISTANCE = 5000.0   # Tổng quãng đường bay (đơn vị)
    NUM_SUB_AIRPORTS = 3      # Số sân bay phụ trên đường bay
    FUEL_CAPACITY = 100.0     # Dung tích nhiên liệu tối đa

    def __init__(self, fleet_data: pd.DataFrame, model, scaler,
                 sensor_list: list = None, features_list: list = None):
        """
        Args:
            fleet_data: DataFrame huấn luyện (đã rolling + có cột RUL).
            model: Trained Keras LSTM/GRU model.
            scaler: Fitted StandardScaler.
            sensor_list: List sensor columns cho LSTM.
            features_list: List tất cả feature columns cho scaler.
        """
        super(AircraftEnv, self).__init__()

        self.fleet_data = fleet_data
        self.model = model
        self.scaler = scaler
        self.sensor_list = sensor_list or KEY_SENSORS
        self.features_list = features_list or FEATURES

        # ── Spaces ──────────────────────────────────────────────
        self.action_space = spaces.Discrete(2)  # 0: Fly, 1: Land

        # [Altitude, Velocity, Fuel, RUL, Dist_Next_Airport, Dist_Destination]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([50000.0, 1000.0, self.FUEL_CAPACITY,
                         300.0, self.TOTAL_DISTANCE, self.TOTAL_DISTANCE],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Internal state ──────────────────────────────────────
        self.twin = None
        self.engine_units = fleet_data['unit_nr'].unique()
        self.current_unit_idx = 0
        self.current_cycle_idx = 0
        self.current_engine_data = None
        self.distance_to_destination = self.TOTAL_DISTANCE
        self.sub_airports = []    # Vị trí các sân bay phụ (distance from origin)

    # ================================================================
    # Reset
    # ================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Chọn ngẫu nhiên một engine unit từ dataset
        unit_id = self.np_random.choice(self.engine_units)
        self.current_engine_data = (
            self.fleet_data[self.fleet_data['unit_nr'] == unit_id]
            .sort_values('time_cycles')
            .reset_index(drop=True)
        )

        # Tạo Digital Twin
        self.twin = AircraftDigitalTwin(
            engine_id=unit_id, model=self.model, scaler=self.scaler,
            seq_length=SEQUENCE_LENGTH, features_list=self.features_list,
            sensor_list=self.sensor_list, fuel_capacity=self.FUEL_CAPACITY
        )

        # Nạp 30 cycle đầu vào buffer trong twin
        init_cycles = min(SEQUENCE_LENGTH, len(self.current_engine_data))
        for i in range(init_cycles):
            row = self.current_engine_data.iloc[i]
            self.twin.update_sensor_data(row[self.features_list].values.tolist())
        self.current_cycle_idx = init_cycles

        # Khởi tạo hành trình
        self.distance_to_destination = self.TOTAL_DISTANCE
        self.twin.fuel = self.FUEL_CAPACITY

        # Sân bay phụ đặt đều trên lộ trình
        spacing = self.TOTAL_DISTANCE / (self.NUM_SUB_AIRPORTS + 1)
        self.sub_airports = [spacing * (i + 1) for i in range(self.NUM_SUB_AIRPORTS)]
        # sub_airports chứa khoảng cách từ điểm xuất phát

        obs = self._get_obs()
        return obs, {}

    # ================================================================
    # Step
    # ================================================================
    def step(self, action):
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

        # 3. Xử lý Hành động
        if action == 1:
            # ── CHỌN HẠ CÁNH BẢO TRÌ ──
            reward = -5.0  # Phí bảo trì và chậm trễ
            self.twin.fuel = self.FUEL_CAPACITY  # Bơm đầy nhiên liệu

            # Reset sang engine mới từ dataset (giả lập thay engine)
            self._reset_to_new_engine()
            info['event'] = "MAINTAINED"
            info['rul_at_landing'] = current_rul

        else:
            # ── CHỌN TIẾP TỤC BAY ──
            distance_covered = self.V_CRUISE * 1  # Quãng đường mỗi cycle
            self.distance_to_destination -= distance_covered
            self.twin.fuel -= self.FUEL_RATE
            reward = 10.0  # Thưởng vì đang tiến về đích

            # Kiểm tra hết nhiên liệu
            if self.twin.fuel <= 0:
                reward = -1000.0
                done = True
                info['event'] = "FUEL_EMPTY"

            # KIỂM TRA THẤT BẠI (Nổ máy trên không - RUL <= 0)
            if current_rul <= 0:
                reward = -2000.0
                done = True
                info['event'] = "CRASHED"

            # KIỂM TRA HOÀN THÀNH
            if self.distance_to_destination <= 0:
                reward = 5000.0
                done = True
                info['event'] = "ARRIVED"

        # 4. Trả về Observation mới
        obs = self._get_obs()
        return obs, reward, done, truncated, info

    # ================================================================
    # Helpers
    # ================================================================
    def _get_obs(self) -> np.ndarray:
        """Trả về observation vector 6 chiều."""
        current_rul = self.twin.current_rul if self.twin.current_rul is not None else 150.0
        dist_next_airport = self._dist_to_nearest_airport()

        return np.array([
            self.twin.altitude,
            self.twin.velocity,
            self.twin.fuel,
            current_rul,
            dist_next_airport,
            max(0.0, self.distance_to_destination)
        ], dtype=np.float32)

    def _dist_to_nearest_airport(self) -> float:
        """Tính khoảng cách đến sân bay phụ gần nhất phía trước."""
        current_pos = self.TOTAL_DISTANCE - self.distance_to_destination
        ahead = [ap - current_pos for ap in self.sub_airports if ap > current_pos]
        if ahead:
            return min(ahead)
        # Nếu không còn sân bay phụ phía trước, trả về quãng đường đến đích
        return max(0.0, self.distance_to_destination)

    def _get_next_sensor_row(self):
        """Lấy dòng cảm biến tiếp theo từ dataset hiện tại."""
        if self.current_cycle_idx < len(self.current_engine_data):
            row = self.current_engine_data.iloc[self.current_cycle_idx]
            self.current_cycle_idx += 1
            return row[self.features_list].values.tolist()
        return None  # Hết dữ liệu cho engine hiện tại

    def _reset_to_new_engine(self):
        """
        Giả lập bảo trì: chọn engine mới từ dataset,
        nạp lại 30 cycle đầu vào buffer.
        """
        unit_id = self.np_random.choice(self.engine_units)
        self.current_engine_data = (
            self.fleet_data[self.fleet_data['unit_nr'] == unit_id]
            .sort_values('time_cycles')
            .reset_index(drop=True)
        )

        # Reset twin buffer
        self.twin.buffer = []
        self.twin.engine_id = unit_id
        self.twin.current_rul = None
        self.twin.status = "HEALTHY"
        self.twin.fuel = self.FUEL_CAPACITY

        # Nạp 30 cycle đầu (engine mới = khỏe mạnh)
        init_cycles = min(SEQUENCE_LENGTH, len(self.current_engine_data))
        for i in range(init_cycles):
            row = self.current_engine_data.iloc[i]
            self.twin.update_sensor_data(row[self.features_list].values.tolist())
        self.current_cycle_idx = init_cycles
