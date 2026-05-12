import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ==============================
# Configuration
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting (2).csv"


# ==============================
# Load Data
# ==============================

def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df["ngay"] = pd.to_datetime(df["ngay"])
    df = df.sort_values("ngay").reset_index(drop=True)

    return df


# ==============================
# Train-Test Split (70-30)
# ==============================

def train_test_split_time_series(df: pd.DataFrame, train_ratio: float = 0.7):
    split_index = int(len(df) * train_ratio)

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    return train, test


# ==============================
# Naive Forecast (NO LEAKAGE)
# ==============================

def naive_forecast(train: pd.DataFrame, test: pd.DataFrame, target_column: str):
    """
    Naive model:
    y_hat(t) = last value from training set
    Multi-step forecast (flat line)
    """

    last_train_value = train[target_column].iloc[-1]

    # Dự đoán toàn bộ test bằng giá cuối train
    predictions = np.full(shape=len(test), fill_value=last_train_value)

    return predictions


# ==============================
# Evaluation
# ==============================

def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mae, rmse


# ==============================
# Plot Function
# ==============================

def plot_results(train, test, y_pred, target_column, title):
    plt.figure(figsize=(12, 6))

    # Train 70%
    plt.plot(train["ngay"], train[target_column],
             color="deepskyblue", label="Train (70%)")

    # Test 30% thật
    plt.plot(test["ngay"], test[target_column],
             color="green", label="Test Actual (30%)")

    # Dự đoán 30%
    plt.plot(test["ngay"], y_pred,
             color="red", linestyle="--", label="Naive Prediction")

    plt.title(title)
    plt.xlabel("Ngày")
    plt.ylabel("Giá (VND/kg)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ==============================
# Main
# ==============================

def main():
    df = load_data(DATA_PATH)

    train, test = train_test_split_time_series(df, 0.7)

    # =====================
    # Ruột Trắng
    # =====================

    y_pred_trang = naive_forecast(train, test, "gia_ruot_trang_vnd_kg")

    mae_trang, rmse_trang = evaluate(
        test["gia_ruot_trang_vnd_kg"].values,
        y_pred_trang
    )

    print("Ruột Trắng")
    print("MAE :", round(mae_trang, 2))
    print("RMSE:", round(rmse_trang, 2))

    plot_results(
        train,
        test,
        y_pred_trang,
        "gia_ruot_trang_vnd_kg",
        "Naive Model - Ruột Trắng"
    )

    # =====================
    # Ruột Đỏ
    # =====================

    y_pred_do = naive_forecast(train, test, "gia_ruot_do_vnd_kg")

    mae_do, rmse_do = evaluate(
        test["gia_ruot_do_vnd_kg"].values,
        y_pred_do
    )

    print("\nRuột Đỏ")
    print("MAE :", round(mae_do, 2))
    print("RMSE:", round(rmse_do, 2))

    plot_results(
        train,
        test,
        y_pred_do,
        "gia_ruot_do_vnd_kg",
        "Naive Model - Ruột Đỏ"
    )


if __name__ == "__main__":
    main()