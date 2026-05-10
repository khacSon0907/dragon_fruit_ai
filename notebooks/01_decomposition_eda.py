"""
=============================================================
01_decomposition_eda.py
Phân tích Decomposition — Thanh Long Trắng & Đỏ
Mục tiêu: Xác định Additive hay Multiplicative phù hợp hơn
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ============================================================
# CONFIG
# ============================================================
PERIOD       = 12   # chu kỳ 12 tháng
FIGSIZE_DECOMP = (14, 10)
FIGSIZE_COMPARE = (14, 6)

CSV_PATH = (
    Path(__file__).parent.parent
    / "data" / "processed"
    / "dragon_fruit_price_forecasting.csv"
)

COLUMNS = {
    "price_white_vnd": "Thanh Long Trắng",
    "price_red_vnd"  : "Thanh Long Đỏ",
}

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["month"])
    df = df.set_index("date").sort_index()
    print("✅ Load data thành công!")
    print(f"   Shape  : {df.shape}")
    print(f"   Period : {df.index.min()} → {df.index.max()}")
    print(f"   Columns: {list(df.columns)}\n")
    return df


# ============================================================
# 2. DECOMPOSE — CẢ HAI KIỂU
# ============================================================
def run_decomposition(series: pd.Series, period: int = PERIOD):
    """Chạy cả additive và multiplicative, trả về cả hai kết quả."""
    add_result = seasonal_decompose(series.dropna(), model="additive",       period=period)
    mul_result = seasonal_decompose(series.dropna(), model="multiplicative",  period=period)
    return add_result, mul_result


# ============================================================
# 3. SO SÁNH — CHỌN MÔ HÌNH PHÙ HỢP
# ============================================================
def compare_models(add_result, mul_result, label: str) -> str:
    """
    So sánh Additive vs Multiplicative dựa trên:
    - Variance của Residual (nhỏ hơn = tốt hơn)
    - STD của Residual
    - CV của Seasonal component
    """
    add_resid = add_result.resid.dropna()
    mul_resid = mul_result.resid.dropna()

    add_var = add_resid.var()
    mul_var = mul_resid.var()

    add_std = add_resid.std()
    mul_std = mul_resid.std()

    # CV: std/mean — đo độ phân tán tương đối của seasonal
    add_seasonal_cv = add_result.seasonal.std() / abs(add_result.seasonal.mean())
    mul_seasonal_cv = mul_result.seasonal.std() / abs(mul_result.seasonal.mean())

    winner = "additive" if add_var < mul_var else "multiplicative"

    print(f"{'='*55}")
    print(f"  📊 So sánh — {label}")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Additive':>10} {'Multiplicative':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Residual Variance':<30} {add_var:>10.4f} {mul_var:>15.4f}")
    print(f"  {'Residual STD':<30} {add_std:>10.4f} {mul_std:>15.4f}")
    print(f"  {'Seasonal CV':<30} {add_seasonal_cv:>10.4f} {mul_seasonal_cv:>15.4f}")
    print(f"{'='*55}")
    print(f"  ✅ Kết luận: {label} → dùng [{winner.upper()}]\n")

    return winner


# ============================================================
# 4. VISUALIZE DECOMPOSITION
# ============================================================
def plot_decomposition(add_result, mul_result, label: str, winner: str):
    """Vẽ decomposition của cả 2 kiểu cạnh nhau để so sánh trực quan."""

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Decomposition — {label}\n✅ Phù hợp hơn: {winner.upper()}",
        fontsize=14, fontweight="bold", y=1.01
    )

    components = ["observed", "trend", "seasonal", "resid"]
    comp_labels = ["Observed", "Trend", "Seasonal", "Residual"]
    colors_add = ["steelblue", "orange", "green", "red"]
    colors_mul = ["steelblue", "darkorange", "darkgreen", "darkred"]

    for i, (comp, comp_label) in enumerate(zip(components, comp_labels)):
        # Additive
        ax1 = fig.add_subplot(4, 2, i * 2 + 1)
        getattr(add_result, comp).plot(ax=ax1, color=colors_add[i])
        ax1.set_title(f"Additive — {comp_label}", fontsize=10)
        ax1.set_ylabel(comp_label)
        ax1.grid(True, alpha=0.3)
        if winner == "additive":
            ax1.set_facecolor("#f0fff0")  # highlight xanh nhạt

        # Multiplicative
        ax2 = fig.add_subplot(4, 2, i * 2 + 2)
        getattr(mul_result, comp).plot(ax=ax2, color=colors_mul[i])
        ax2.set_title(f"Multiplicative — {comp_label}", fontsize=10)
        ax2.set_ylabel(comp_label)
        ax2.grid(True, alpha=0.3)
        if winner == "multiplicative":
            ax2.set_facecolor("#f0fff0")  # highlight xanh nhạt

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. PLOT SEASONAL PATTERN (theo tháng)
# ============================================================
def plot_seasonal_pattern(add_result, mul_result, label: str, winner: str):
    """Vẽ seasonal pattern theo tháng trong năm."""

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_COMPARE)
    fig.suptitle(f"Seasonal Pattern Theo Tháng — {label}", fontsize=13, fontweight="bold")

    for ax, result, model_type in zip(axes, [add_result, mul_result], ["Additive", "Multiplicative"]):
        seasonal = result.seasonal
        monthly_avg = seasonal.groupby(seasonal.index.month).mean()

        month_names = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"]
        bars = ax.bar(month_names, monthly_avg.values, color="steelblue", alpha=0.7, edgecolor="white")

        # Highlight tháng cao nhất / thấp nhất
        max_idx = monthly_avg.values.argmax()
        min_idx = monthly_avg.values.argmin()
        bars[max_idx].set_color("green")
        bars[min_idx].set_color("red")

        ax.axhline(y=monthly_avg.mean(), color="orange", linestyle="--", linewidth=1.5, label="Mean")
        ax.set_title(f"{model_type} {'✅' if model_type.lower() == winner else ''}", fontsize=11)
        ax.set_xlabel("Tháng")
        ax.set_ylabel("Seasonal Component")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        print(f"  [{model_type}] {label}: Tháng cao nhất = T{max_idx+1} | Thấp nhất = T{min_idx+1}")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. ADF TEST — KIỂM TRA STATIONARITY
# ============================================================
def adf_test(series: pd.Series, label: str):
    """Kiểm tra data có stationary không (p < 0.05 = stationary)."""
    adf_stat, p_value, _, _, critical_values, _ = adfuller(series.dropna())

    print(f"\n📋 ADF Test — {label}")
    print(f"   ADF Statistic : {adf_stat:.4f}")
    print(f"   p-value       : {p_value:.4f}")
    for key, val in critical_values.items():
        print(f"   Critical ({key}) : {val:.4f}")

    if p_value < 0.05:
        print(f"   ✅ Stationary (p={p_value:.4f} < 0.05) → có thể dùng ARIMA")
    else:
        print(f"   ❌ Non-stationary (p={p_value:.4f} > 0.05) → cần differencing hoặc SARIMA")


# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Load
    df = load_data()

    results_summary = {}

    for col, label in COLUMNS.items():
        print(f"\n{'#'*60}")
        print(f"  🔍 Đang phân tích: {label}")
        print(f"{'#'*60}\n")

        series = df[col]

        # 2. Decompose
        add_result, mul_result = run_decomposition(series)

        # 3. So sánh → chọn winner
        winner = compare_models(add_result, mul_result, label)
        results_summary[label] = winner

        # 4. Vẽ decomposition
        plot_decomposition(add_result, mul_result, label, winner)

        # 5. Vẽ seasonal pattern theo tháng
        plot_seasonal_pattern(add_result, mul_result, label, winner)

        # 6. ADF test
        adf_test(series, label)

    # ── Tổng kết ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  📌 TỔNG KẾT DECOMPOSITION")
    print(f"{'='*55}")
    for label, winner in results_summary.items():
        print(f"  {label:<25} → [{winner.upper()}]")
    print(f"{'='*55}")
    print("\n  👉 Bước tiếp theo: Dùng kết quả trên để chọn mô hình")
    print("     Additive       → SARIMA / Prophet (additive mode)")
    print("     Multiplicative → SARIMA với log transform / Prophet (multiplicative mode)")


if __name__ == "__main__":
    main()