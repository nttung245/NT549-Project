import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scripts.lstm_model import create_sequences
from scripts.data_processor import FEATURES, SEQUENCE_LENGTH

def evaluate_lstm_performance(model, test_rolling, true_rul, scaler):
    """
    Calculate R2, RMSE, MAE and plot residual distribution.
    """
    print("📊 Calculating metrics on the test set...")
    X_test, _ = create_sequences(test_rolling, scaler, is_test=True)
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_true = true_rul['RUL_ground_truth'].values

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("=" * 30)
    print(f"✅ R2 Score:  {r2:.4f}")
    print(f"✅ RMSE:      {rmse:.2f} cycles")
    print(f"✅ MAE:       {mae:.2f} cycles")
    print("=" * 30)

    # Plot Comparison (All 100 engines)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Thực tế', color='blue', linewidth=2)
    plt.plot(y_pred, label=f'LSTM (R2={r2:.2f})', color='green', linewidth=2)
    plt.title('SO SÁNH HIỆU NĂNG LSTM TRÊN TẬP TEST NASA CMAPSS')
    plt.xlabel('Chỉ số Động cơ (Engine Index)')
    plt.ylabel('Số chu kỳ còn lại (RUL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot Residuals
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 4))
    plt.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Error Distribution (Residuals: Predicted - Actual)')
    plt.xlabel('Error (Cycles)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return y_true, y_pred

def plot_engine_degradation(model, test_rolling, engine_id, scaler, seq_len=SEQUENCE_LENGTH):
    """
    Plot the entire RUL trajectory for a specific engine.
    """
    print(f"📈 Extracting trajectory for Engine #{engine_id}...")
    engine_data = test_rolling[test_rolling['unit_nr'] == engine_id]
    
    # Scale data
    scaled_data = scaler.transform(engine_data[FEATURES])
    # IMPORTANT: Assumes FEATURES matches model expectations (e.g., sensor only)
    # The current model in scripts/lstm_model.py uses input_shape=(None, len(sensor_only_list))
    # We need to extract just the sensors.
    # In data_processor.py, FEATURES = KEY_SENSORS + ['time_cycles'] (15 features)
    # The create_sequences logic uses sensor_only_list.
    
    from scripts.data_processor import KEY_SENSORS
    sensor_indices = [FEATURES.index(s) for s in KEY_SENSORS]
    sensors = scaled_data[:, sensor_indices]
    
    X_engine = []
    y_engine_true = []
    
    for i in range(len(sensors) - seq_len + 1):
        X_engine.append(sensors[i:i + seq_len])
        y_engine_true.append(engine_data['RUL'].iloc[i + seq_len - 1])
    
    X_engine = np.array(X_engine)
    y_engine_pred = model.predict(X_engine, verbose=0).flatten()
    
    plt.figure(figsize=(12, 5))
    plt.plot(y_engine_true, label='Actual RUL', color='blue', linewidth=2)
    plt.plot(y_engine_pred, label='Digital Twin Prediction', color='orange', linestyle='--')
    plt.title(f'Degradation Tracking: Engine #{engine_id}')
    plt.xlabel('Time Cycles (window sliding)')
    plt.ylabel('Remaining Useful Life (RUL)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
