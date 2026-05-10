"""
=============================================================
02_stationarity.py
Kiểm tra Stationarity — Thanh Long Trắng & Đỏ
Mục tiêu:
  - Log transform (vì Multiplicative)
  - ADF Test → data có stationary không?
  - Nếu không → differencing d=1
  - ACF / PACF → gợi ý tham số p, q cho SARIMA
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ============================================================
# CONFIG
# ============================================================
PERIOD   = 12
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
    print(f"   Shape : {df.shape}")
    print(f"   Period: {df.index.min()} → {df.index.max()}\n")
    return df


# ============================================================
# 2. LOG TRANSFORM (vì Multiplicative)
# ============================================================
def log_transform(series: pd.Series, label: str) -> pd.Series:
    """
    Multiplicative → Log transform → về dạng Additive
    Y = T × S × R  →  log(Y) = log(T) + log(S) + log(R)
    """
    log_series = np.log(series.dropna())
    print(f"📐 Log transform — {label}")
    print(f"   Original mean : {series.mean():.2f} VNĐ")
    print(f"   Log mean      : {log_series.mean():.4f}\n")
    return log_series


# ============================================================
# 3. ADF TEST
# ============================================================
def adf_test(series: pd.Series, label: str) -> bool:
    """
    Augmented Dickey-Fuller Test
    H0: Non-stationary (có unit root)
    H1: Stationary

    p < 0.05 → reject H0 → Stationary ✅
    p > 0.05 → fail to reject H0 → Non-stationary ❌ → cần differencing
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value, _, _, critical_values, _ = result

    is_stationary = p_value < 0.05

    print(f"{'='*50}")
    print(f"  📋 ADF Test — {label}")
    print(f"{'='*50}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    for key, val in critical_values.items():
        marker = " ←" if abs(adf_stat) > abs(val) else ""
        print(f"  Critical ({key}) : {val:.4f}{marker}")
    print(f"{'='*50}")

    if is_stationary:
        print(f"  ✅ STATIONARY (p={p_value:.4f} < 0.05)")
        print(f"     → Không cần differencing")
        print(f"     → Dùng SARIMA với d=0\n")
    else:
        print(f"  ❌ NON-STATIONARY (p={p_value:.4f} > 0.05)")
        print(f"     → Cần differencing d=1")
        print(f"     → Dùng SARIMA với d=1\n")

    return is_stationary


# ============================================================
# 4. DIFFERENCING (nếu non-stationary)
# ============================================================
def apply_differencing(series: pd.Series, label: str, d: int = 1) -> pd.Series:
    """Áp dụng differencing bậc d để đạt stationary."""
    diff_series = series.copy()
    for i in range(d):
        diff_series = diff_series.diff().dropna()

    print(f"🔄 Differencing d={d} — {label}")
    # Kiểm tra lại sau khi differencing
    result = adfuller(diff_series.dropna(), autolag="AIC")
    p_after = result[1]
    print(f"   p-value sau differencing: {p_after:.4f}")
    print(f"   {'✅ Stationary rồi!' if p_after < 0.05 else '❌ Vẫn chưa stationary, thử d=2'}\n")

    return diff_series


# ============================================================
# 5. ACF / PACF PLOT → gợi ý p, q
# ============================================================
def plot_acf_pacf(series: pd.Series, label: str, lags: int = 36):
    """
    ACF  → gợi ý q (MA order)
    PACF → gợi ý p (AR order)

    Cách đọc:
    - Nhìn lag nào vượt qua đường confidence interval (màu xanh)
    - ACF  cut off tại lag k → q = k
    - PACF cut off tại lag k → p = k
    - Có spike tại lag 12 → seasonal component rõ
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"ACF & PACF — {label}\n(gợi ý tham số p, q cho SARIMA)",
        fontsize=13, fontweight="bold"
    )

    plot_acf(series.dropna(), lags=lags, ax=axes[0],
             title=f"ACF — gợi ý q (MA order)", alpha=0.05)
    plot_pacf(series.dropna(), lags=lags, ax=axes[1],
              title=f"PACF — gợi ý p (AR order)", alpha=0.05, method="ywm")

    for ax in axes:
        ax.axvline(x=12, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.axvline(x=24, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_xlabel("Lag (tháng)")
        ax.grid(True, alpha=0.3)

    axes[0].annotate("Seasonal\nlag 12", xy=(12, 0), fontsize=9,
                     color="red", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. VISUALIZE — so sánh trước/sau transform
# ============================================================
def plot_transform_comparison(original: pd.Series,
                               log_series: pd.Series,
                               diff_series: pd.Series,
                               label: str):
    """Vẽ 3 dạng: gốc, log, log+diff để so sánh trực quan."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Biến đổi Series — {label}", fontsize=13, fontweight="bold")

    original.plot(ax=axes[0], color="steelblue", linewidth=1.5)
    axes[0].set_title("Gốc (VNĐ) — Multiplicative, non-stationary")
    axes[0].set_ylabel("Giá (VNĐ)")
    axes[0].grid(True, alpha=0.3)

    log_series.plot(ax=axes[1], color="orange", linewidth=1.5)
    axes[1].set_title("Log Transform — về dạng Additive")
    axes[1].set_ylabel("log(Giá)")
    axes[1].grid(True, alpha=0.3)

    diff_series.plot(ax=axes[2], color="green", linewidth=1.5)
    axes[2].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_title("Log + Differencing d=1 — Stationary ✅")
    axes[2].set_ylabel("Δlog(Giá)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    df = load_data()

    summary = {}

    for col, label in COLUMNS.items():
        print(f"\n{'#'*55}")
        print(f"  🔍 {label}")
        print(f"{'#'*55}\n")

        series = df[col].dropna()

        # ── Step 1: Log transform ──────────────────────────
        log_series = log_transform(series, label)

        # ── Step 2: ADF test trên log series ──────────────
        print("--- ADF test trên Log series ---")
        is_stationary = adf_test(log_series, f"{label} (log)")

        # ── Step 3: Differencing nếu cần ──────────────────
        if is_stationary:
            final_series = log_series
            d_value = 0
        else:
            diff_series = apply_differencing(log_series, label, d=1)

            # Kiểm tra lần 2 sau diff
            print("--- ADF test sau Differencing ---")
            is_stat_after = adf_test(diff_series, f"{label} (log+diff)")

            if not is_stat_after:
                print("⚠️  Thử differencing d=2...")
                diff_series = apply_differencing(log_series, label, d=2)
                adf_test(diff_series, f"{label} (log+diff2)")
                d_value = 2
            else:
                d_value = 1

            final_series = diff_series

        # ── Step 4: Visualize so sánh ─────────────────────
        plot_transform_comparison(
            series,
            log_series,
            final_series if d_value > 0 else log_series.diff().dropna(),
            label
        )

        # ── Step 5: ACF / PACF ────────────────────────────
        plot_acf_pacf(final_series, label)

        # ── Ghi lại kết quả ───────────────────────────────
        summary[label] = {
            "transform"    : "log",
            "d"            : d_value,
            "stationary"   : True,
        }

    # ── Tổng kết ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  📌 TỔNG KẾT STATIONARITY")
    print(f"{'='*55}")
    for label, info in summary.items():
        print(f"  {label}")
        print(f"    Transform  : {info['transform']}")
        print(f"    d          : {info['d']}")
        print(f"    Stationary : ✅")
    print(f"{'='*55}")
    print("\n  👉 Bước tiếp theo: dùng d này để train SARIMA")
    print("     hoặc dùng Prophet (không cần log transform)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()