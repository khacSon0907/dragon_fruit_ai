import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==============================
# CONFIG
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting (2).csv"


# ==============================
# LOAD DATA
# ==============================

def load_data(file_path: Path):
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    df["ngay"] = pd.to_datetime(df["ngay"])

    df = df.sort_values("ngay")
    df = df.set_index("ngay")

    return df


# ==============================
# PREPROCESS
# ==============================

def preprocess_data(df):
    df = df.sort_index()
    df = df.asfreq("D")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if "xu_huong" in df.columns:
        df["xu_huong"] = df["xu_huong"].ffill().bfill()

    return df


# ==============================
# PLOT SERIES
# ==============================

def plot_series(series, title):
    plt.figure(figsize=(14, 6))
    plt.plot(series, label="Price", linewidth=1.5)

    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("VND/kg")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ==============================
# ADF TEST
# ==============================

def adf_test(series, name="Series"):
    print(f"\n===== ADF TEST: {name} =====")

    result = adfuller(series.dropna())

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    if result[1] < 0.05:
        print("=> Stationary")
    else:
        print("=> Not Stationary")


# ==============================
# TRAIN TEST SPLIT
# ==============================

def train_test_split_series(series, test_size=0.2):
    split_idx = int(len(series) * (1 - test_size))

    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    return train, test


# ==============================
# FOURIER FEATURES
# dùng để mô phỏng mùa vụ 365 ngày
# nhẹ hơn seasonal_order=(..., 365)
# ==============================

def create_fourier_features(index, period=365, order=3):
    t = np.arange(len(index))
    features = pd.DataFrame(index=index)

    for k in range(1, order + 1):
        features[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    return features


# ==============================
# TRAIN SARIMAX
# ==============================

def train_sarimax(train, exog_train=None):
    model = SARIMAX(
        train,
        exog=exog_train,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)
    return result


# ==============================
# EVALUATE
# ==============================

def evaluate_forecast(test, forecast, name):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f"\n===== EVALUATION: {name} =====")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAPE:", round(mape, 2), "%")


# ==============================
# PLOT FORECAST
# ==============================

def plot_forecast(train, test, forecast, title):
    plt.figure(figsize=(14, 6))

    plt.plot(train, label="Train", linewidth=1.2)
    plt.plot(test, label="Actual", linewidth=1.5)
    plt.plot(forecast, label="Forecast", linewidth=1.5)

    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("VND/kg")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ==============================
# FORECAST FUTURE
# ==============================

def forecast_future(result, full_index, steps=30, period=365, order=3):
    last_date = full_index[-1]

    future_index = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )

    combined_index = full_index.append(future_index)

    fourier_all = create_fourier_features(
        combined_index,
        period=period,
        order=order
    )

    exog_future = fourier_all.loc[future_index]

    future_forecast = result.forecast(
        steps=steps,
        exog=exog_future
    )

    future_forecast = pd.Series(
        future_forecast,
        index=future_index
    )

    return future_forecast


def plot_future_forecast(series, future_forecast, title):
    plt.figure(figsize=(14, 6))

    plt.plot(series, label="Historical Price", linewidth=1.3)
    plt.plot(future_forecast, label="Future Forecast", linewidth=1.8)

    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("VND/kg")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ==============================
# FULL PIPELINE
# ==============================

def run_forecasting_pipeline(series, name, future_steps=30):
    print("\n" + "=" * 50)
    print(f"FORECASTING: {name}")
    print("=" * 50)

    series = series.dropna()

    train, test = train_test_split_series(series, test_size=0.2)

    print("Train length:", len(train))
    print("Test length:", len(test))

    period = 365
    fourier_order = 3

    fourier = create_fourier_features(
        series.index,
        period=period,
        order=fourier_order
    )

    exog_train = fourier.loc[train.index]
    exog_test = fourier.loc[test.index]

    result = train_sarimax(train, exog_train)

    forecast = result.forecast(
        steps=len(test),
        exog=exog_test
    )

    forecast = pd.Series(
        forecast,
        index=test.index
    )

    evaluate_forecast(test, forecast, name)

    plot_forecast(
        train,
        test,
        forecast,
        f"SARIMAX + Yearly Seasonality Forecast - {name}"
    )

    future_forecast = forecast_future(
        result=result,
        full_index=series.index,
        steps=future_steps,
        period=period,
        order=fourier_order
    )

    print(f"\n===== FUTURE FORECAST: {name} =====")
    print(future_forecast.head(10))

    plot_future_forecast(
        series,
        future_forecast,
        f"Next {future_steps} Days Forecast - {name}"
    )

    return result, forecast, future_forecast


# ==============================
# MAIN
# ==============================

def main():
    df = load_data(DATA_PATH)

    print("Data loaded successfully!")
    print(df.head())

    df = preprocess_data(df)

    print("\nAfter preprocessing shape:", df.shape)
    print(df.head())

    series_trang = df["gia_ruot_trang_vnd_kg"]
    series_do = df["gia_ruot_do_vnd_kg"]

    print("\nRuot Trang length:", len(series_trang))
    print("Ruot Do length:", len(series_do))

    plot_series(series_trang, "Dragon Fruit Price - Ruot Trang")
    plot_series(series_do, "Dragon Fruit Price - Ruot Do")

    adf_test(series_trang, "Ruot Trang")
    adf_test(series_do, "Ruot Do")

    result_trang, forecast_trang, future_trang = run_forecasting_pipeline(
        series_trang,
        "Ruot Trang",
        future_steps=30
    )

    result_do, forecast_do, future_do = run_forecasting_pipeline(
        series_do,
        "Ruot Do",
        future_steps=30
    )


if __name__ == "__main__":
    main()