# -*- coding: utf-8 -*-
"""
Aircraft Digital Twin Module.
Manages individual engine state and LSTM-based RUL predictions.
Refactored from RocketDigitalTwin → AircraftDigitalTwin.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# Tối ưu hóa Threading cho CPU: Tránh tranh chấp tài nguyên khi chạy nhiều Env song song
# Điều này cực kỳ quan trọng khi chạy 4-8 Env trên CPU 4 nhân / 16 luồng
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
        # Sử dụng NumPy array cố định kích thước thay vì list để tối ưu CPU
        self.n_features = len(self.features_list)
        self.buffer = np.zeros((self.seq_length, self.n_features), dtype=np.float32)
        self.buffer_filled = 0  # Đếm số dòng đã nạp

        self.status = "HEALTHY"
        self.current_rul = None

    def update_sensor_data(self, new_data_row):
        """
        Nạp dữ liệu cảm biến mới vào sliding window buffer (Sử dụng NumPy).
        """
        # Đẩy dữ liệu cũ lên và chèn dòng mới vào cuối
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = new_data_row

        if self.buffer_filled < self.seq_length:
            self.buffer_filled += 1

    def predict_status(self) -> float | None:
        """
        Dự đoán RUL dựa trên dữ liệu chuỗi thời gian trong buffer (Đã tối ưu CPU).
        """
        if self.buffer_filled < self.seq_length:
            self.status = "COLLECTING_DATA"
            return None

        # Tiền xử lý: Scale trực tiếp từ NumPy array (Bỏ qua pd.DataFrame để tăng tốc)
        # scaler.transform nhận (n_samples, n_features)
        scaled_data = self.scaler.transform(self.buffer)

        # Lấy subset sensor cho LSTM
        sensor_indices = [self.features_list.index(s) for s in self.sensor_list]
        lstm_input = scaled_data[:, sensor_indices]
        
        # Reshape cho Keras LSTM (batch_size, seq_len, features)
        lstm_input_reshaped = lstm_input.reshape(1, self.seq_length, -1).astype(np.float32)

        # Dự đoán RUL: Gọi trực tiếp __call__ thay vì predict() để bỏ qua setup overhead
        # training=False giúp tăng tốc và đảm bảo không kích hoạt Dropout/BatchNorm
        prediction = self.model(lstm_input_reshaped, training=False)
        self.current_rul = float(prediction.numpy().flatten()[0])

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
        self.buffer.fill(0)
        self.buffer_filled = 0
        self.status = "HEALTHY"
        self.current_rul = None
