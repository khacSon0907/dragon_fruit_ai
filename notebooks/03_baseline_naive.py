import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# LOAD DATA
# =========================
def load_data():
    """
    Load dữ liệu thanh long
    """
    csv_path = Path(__file__).parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting (2).csv"

    df = pd.read_csv(csv_path)

    # Convert date
    df['date'] = pd.to_datetime(df['ngay'])

    # Sort theo thời gian (rất quan trọng cho time series)
    df = df.sort_values('date').reset_index(drop=True)

    return df


# =========================
# NAIVE FORECAST
# =========================
def naive_forecast(last_train_value, steps):
    """
    Naive forecast:
    Giá trị dự đoán = giá trị cuối cùng của train
    """
    return np.repeat(last_train_value, steps)


# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_naive(df, column_name, label):

    # =========================
    # Split 70% train - 30% test
    # =========================
    split_idx = int(len(df) * 0.7)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    y_true = test[column_name].values

    # Giá trị cuối train
    last_value = train[column_name].iloc[-1]

    # Dự đoán naive
    y_pred = naive_forecast(last_value, len(test))

    # =========================
    # Metrics
    # =========================
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n===== {label} - Naive Baseline (70/30) =====")
    print(f"Train size: {len(train)}")
    print(f"Test size : {len(test)}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(14, 6))

    plt.plot(test['date'], y_true,
             label='Thực tế', color='blue', linewidth=2)

    plt.plot(test['date'], y_pred,
             label='Naive Forecast',
             color='red', linestyle='--', linewidth=2)

    plt.title(f'{label} - Naive Baseline (70/30 Split)',
              fontsize=14, fontweight='bold')

    plt.xlabel('Thời gian')
    plt.ylabel('Giá (VNĐ/kg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return mae, rmse


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()

    # Ruột Trắng
    mae_white, rmse_white = evaluate_naive(
        df,
        'gia_ruot_trang_vnd_kg',
        'Ruột Trắng'
    )

    # Ruột Đỏ
    mae_red, rmse_red = evaluate_naive(
        df,
        'gia_ruot_do_vnd_kg',
        'Ruột Đỏ'
    )

    # =========================
    # Tổng hợp kết quả
    # =========================
    print("\n===== KẾT QUẢ TỔNG HỢP =====")
    print(f"Ruột Trắng → MAE={mae_white:.2f}, RMSE={rmse_white:.2f}")
    print(f"Ruột Đỏ    → MAE={mae_red:.2f}, RMSE={rmse_red:.2f}")