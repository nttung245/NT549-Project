# -*- coding: utf-8 -*-
"""
LSTM/GRU Model Module.
Defines the deep learning model architecture and training utilities for RUL prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scripts.data_processor import FEATURES, KEY_SENSORS, SEQUENCE_LENGTH


# ============================================================
# Model Architecture
# ============================================================
def create_lstm_model(input_shape: tuple) -> Sequential:
    """
    Create a GRU-based model for RUL prediction.

    Args:
        input_shape: (sequence_length, num_features)

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        GRU(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1, activation='relu')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# ============================================================
# Sequence Creation
# ============================================================
def create_sequences(data: pd.DataFrame, scaler, seq_length: int = SEQUENCE_LENGTH,
                     all_features: list = None, sensor_only_list: list = None,
                     is_test: bool = None):
    """
    Create sliding window sequences for LSTM.
    Handles 'Final Sequence only' (for test set) and 'All sequences' (for training).
    """
    if all_features is None:
        all_features = FEATURES
    if sensor_only_list is None:
        sensor_only_list = KEY_SENSORS

    X, y = [], []
    has_rul = 'RUL' in data.columns
    
    # If is_test is not explicitly provided, infer from presence of RUL
    if is_test is None:
        is_test_set = not has_rul
    else:
        is_test_set = is_test

    for unit_nr in data['unit_nr'].unique():
        unit_data = data[data['unit_nr'] == unit_nr]

        # Scale dữ liệu theo đúng danh sách features đã fit
        unit_data_scaled = scaler.transform(unit_data[all_features])
        scaled_df = pd.DataFrame(unit_data_scaled, columns=all_features)
        unit_sensors = scaled_df[sensor_only_list].values

        if is_test_set:
            # CHẾ ĐỘ TEST: Lấy 30 chu kỳ cuối (có Padding nếu thiếu)
            if len(unit_sensors) >= seq_length:
                X.append(unit_sensors[-seq_length:])
            else:
                padding = np.zeros((seq_length - len(unit_sensors), len(sensor_only_list)))
                X.append(np.vstack((padding, unit_sensors)))
        else:
            # CHẾ ĐỘ TRAIN: Lấy toàn bộ sliding windows
            if len(unit_sensors) < seq_length:
                continue
            for i in range(len(unit_sensors) - seq_length + 1):
                X.append(unit_sensors[i:i + seq_length])
                if has_rul:
                    y.append(unit_data['RUL'].iloc[i + seq_length - 1])

    X = np.array(X)
    y = np.array(y) if has_rul else None
    return X, y


# ============================================================
# Training
# ============================================================
def train_model(X_train, y_train, epochs: int = 54, batch_size: int = 256,
                validation_split: float = 0.2):
    """
    Train a GRU model on the prepared sequence data.

    Args:
        X_train: Training sequences (num_samples, seq_length, num_features).
        y_train: Target RUL values.
        epochs: Max training epochs.
        batch_size: Batch size.
        validation_split: Fraction of data for validation.

    Returns:
        Trained Keras model, training history.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    print("🚀 Training GRU model for RUL prediction...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )
    print("✅ Training complete!")
    return model, history
