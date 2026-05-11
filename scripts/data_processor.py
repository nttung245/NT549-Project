# -*- coding: utf-8 -*-
"""
Data Processing Module for NASA CMAPSS Dataset.
Handles loading, RUL calculation, rolling features, and scaling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================
# Constants
# ============================================================
INDEX_NAMES = ['unit_nr', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES = ['s_{}'.format(i) for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# Cảm biến có độ lệch chuẩn ≈ 0 (không mang thông tin)
DROP_SENSORS = ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']

# Cảm biến quan trọng để huấn luyện
KEY_SENSORS = [col for col in SENSOR_NAMES if col not in DROP_SENSORS]

# Đặc trưng đầu vào (cảm biến quan trọng + time_cycles) cho scaler
FEATURES = KEY_SENSORS + ['time_cycles']

WINDOW_SIZE = 20
SEQUENCE_LENGTH = 30


# ============================================================
# Data Loading
# ============================================================
def load_cmapss_data(data_dir: str):
    """
    Load train, test, and RUL ground truth from CMAPSS dataset directory.

    Args:
        data_dir: Path to directory containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt

    Returns:
        train_data, test_data, true_rul (DataFrames)
    """
    train_path = os.path.join(data_dir, 'train_FD001.txt')
    test_path = os.path.join(data_dir, 'test_FD001.txt')
    rul_path = os.path.join(data_dir, 'RUL_FD001.txt')

    train_data = pd.read_csv(train_path, sep=r'\s+', header=None, names=COL_NAMES)
    test_data = pd.read_csv(test_path, sep=r'\s+', header=None, names=COL_NAMES)
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL_ground_truth'])
    true_rul['unit_nr'] = true_rul.index + 1

    print(f"Loaded CMAPSS data: {len(train_data)} train rows, {len(test_data)} test rows, {len(true_rul)} engines")
    return train_data, test_data, true_rul


# ============================================================
# Feature Engineering
# ============================================================
def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for training units (until failure)."""
    max_cycle = df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    df = df.merge(max_cycle, on=['unit_nr'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df = df.drop('max_cycle', axis=1)
    return df


def add_test_rul(test_df: pd.DataFrame, true_rul_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL for test units using Ground Truth file."""
    # Lấy chu kỳ tối đa hiện có của mỗi unit trong tập test
    max_cycle = test_df.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle_test']

    # Merge với ground truth (RUL tại chu kỳ cuối cùng)
    df = test_df.merge(max_cycle, on='unit_nr', how='left')
    df = df.merge(true_rul_df, on='unit_nr', how='left')

    # RUL tại chu kỳ c = (max_cycle_test + rul_ground_truth) - current_cycle
    df['RUL'] = (df['max_cycle_test'] + df['RUL_ground_truth']) - df['time_cycles']

    return df.drop(['max_cycle_test', 'RUL_ground_truth'], axis=1)


def add_rolling_features(df: pd.DataFrame, sensors: list[str] | None = None, window: int = WINDOW_SIZE) -> pd.DataFrame:
    """Áp dụng Rolling Mean cho các cảm biến để giảm nhiễu."""
    if sensors is None:
        sensors = KEY_SENSORS
    df_out = df.copy()
    for sensor in sensors:
        df_out[sensor] = df_out.groupby('unit_nr')[sensor].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    return df_out


def prepare_data(data_dir: str):
    """
    Full data pipeline: load → add RUL → rolling features → scale.

    Returns:
        train_rolling, test_rolling, true_rul, scaler
    """
    train_data, test_data, true_rul = load_cmapss_data(data_dir)

    # Tính RUL cho cả tập train và tập test
    train_data = add_remaining_useful_life(train_data)
    test_data = add_test_rul(test_data, true_rul)

    # Áp dụng Rolling Mean
    train_rolling = add_rolling_features(train_data)
    test_rolling = add_rolling_features(test_data)

    # Chuẩn hóa (fit trên tập train)
    scaler = StandardScaler()
    scaler.fit(train_rolling[FEATURES])
    print(f"Scaler fitted on {len(FEATURES)} features")

    return train_rolling, test_rolling, true_rul, scaler
