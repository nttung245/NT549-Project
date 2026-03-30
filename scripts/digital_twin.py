# -*- coding: utf-8 -*-
"""
Aircraft Digital Twin Module.
Manages individual engine state and LSTM-based RUL predictions.
Refactored from RocketDigitalTwin → AircraftDigitalTwin.
"""

import numpy as np
import pandas as pd

from scripts.data_processor import FEATURES, KEY_SENSORS, SEQUENCE_LENGTH


class AircraftDigitalTwin:
    """
    Digital Twin for an aircraft engine.
    Maintains a sliding window buffer of sensor data and uses a trained
    LSTM/GRU model to predict Remaining Useful Life (RUL).
    """

    def __init__(self, engine_id, model, scaler, seq_length=SEQUENCE_LENGTH,
                 features_list=None, sensor_list=None, fuel_capacity=100.0):
        self.engine_id = engine_id
        self.model = model
        self.scaler = scaler
        self.seq_length = seq_length
        self.features_list = features_list or FEATURES
        self.sensor_list = sensor_list or KEY_SENSORS

        # Thông số vật lý
        self.altitude = 10000.0   # Độ cao bay đường trường (m)
        self.velocity = 250.0     # Vận tốc (m/s)
        self.fuel = fuel_capacity

        # AI buffer & status
        self.buffer = []
        self.status = "HEALTHY"
        self.current_rul = None

    def update_sensor_data(self, new_data_row):
        """
        Nạp dữ liệu cảm biến mới vào sliding window buffer.

        Args:
            new_data_row: dict, list, hoặc pd.Series chứa giá trị cảm biến.
        """
        self.buffer.append(new_data_row)
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)

    def predict_status(self) -> float | None:
        """
        Dự đoán RUL dựa trên dữ liệu chuỗi thời gian trong buffer.

        Returns:
            Predicted RUL (float) hoặc None nếu chưa đủ dữ liệu.
        """
        if len(self.buffer) < self.seq_length:
            self.status = "COLLECTING_DATA"
            return None

        # Tiền xử lý: buffer → DataFrame → Scale → lọc sensors
        df_temp = pd.DataFrame(self.buffer, columns=self.features_list)
        scaled_data = self.scaler.transform(df_temp[self.features_list])

        # Lấy subset sensor cho LSTM
        sensor_indices = [self.features_list.index(s) for s in self.sensor_list]
        lstm_input = scaled_data[:, sensor_indices]
        lstm_input_reshaped = lstm_input.reshape(1, self.seq_length, -1)

        # Dự đoán RUL
        prediction = self.model.predict(lstm_input_reshaped, verbose=0)
        self.current_rul = float(prediction.flatten()[0])

        # Cập nhật trạng thái
        if self.current_rul < 20:
            self.status = "CRITICAL"
        elif self.current_rul < 50:
            self.status = "WARNING"
        else:
            self.status = "HEALTHY"

        return self.current_rul

    def reset_state(self, fuel_capacity=100.0):
        """Reset trạng thái vật lý (sau khi bảo trì)."""
        self.fuel = fuel_capacity
        self.altitude = 10000.0
        self.velocity = 250.0
        self.buffer = []
        self.status = "HEALTHY"
        self.current_rul = None
