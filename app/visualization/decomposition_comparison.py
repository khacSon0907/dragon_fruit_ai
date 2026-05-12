"""
Decomposition Comparison: Additive vs Multiplicative
=====================================================
So sánh residual của 2 loại decomposition để chọn model phù hợp.
Tiêu chí: residual nhỏ hơn → model phù hợp hơn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# 1. TẠO DỮ LIỆU MẪU (thay bằng data thực của bạn)
# ============================================================
np.random.seed(42)
periods = 120  # 120 tháng = 10 năm

time = np.arange(periods)
trend = 50 + 0.5 * time                          # trend tăng dần
seasonal = 10 * np.sin(2 * np.pi * time / 12)   # chu kỳ 12 tháng
noise = np.random.normal(0, 3, periods)

# Data multiplicative: biên độ phình theo trend
data_values = trend * (1 + 0.2 * np.sin(2 * np.pi * time / 12)) + noise

date_range = pd.date_range(start="2015-01-01", periods=periods, freq="MS")
series = pd.Series(data_values, index=date_range, name="value")

# ============================================================
# 2. DECOMPOSITION
# ============================================================
period = 12  # chu kỳ mùa vụ (12 = monthly seasonality, 7 = weekly, 4 = quarterly)

add_result  = seasonal_decompose(series, model="additive",       period=period, extrapolate_trend="freq")
mul_result  = seasonal_decompose(series, model="multiplicative",  period=period, extrapolate_trend="freq")

# ============================================================
# 3. TÍNH METRICS SO SÁNH RESIDUAL
# ============================================================
add_resid = add_result.resid.dropna()
mul_resid = mul_result.resid.dropna()

# Align index
common_idx = add_resid.index.intersection(mul_resid.index)
add_resid  = add_resid[common_idx]
mul_resid  = mul_resid[common_idx]

zeros = np.zeros(len(add_resid))

metrics = {
    "MSE":  (mean_squared_error(zeros, add_resid),
             mean_squared_error(zeros, mul_resid)),
    "RMSE": (np.sqrt(mean_squared_error(zeros, add_resid)),
             np.sqrt(mean_squared_error(zeros, mul_resid))),
    "MAE":  (mean_absolute_error(zeros, add_resid),
             mean_absolute_error(zeros, mul_resid)),
    "Std Residual": (add_resid.std(),
                     mul_resid.std()),
    "Variance Residual": (add_resid.var(),
                          mul_resid.var()),
}

# ============================================================
# 4. IN KẾT QUẢ SO SÁNH
# ============================================================
print("=" * 60)
print("     SO SÁNH: ADDITIVE vs MULTIPLICATIVE DECOMPOSITION")
print("=" * 60)
print(f"{'Metric':<22} {'Additive':>14} {'Multiplicative':>16}  {'Winner'}")
print("-" * 60)

scores = {"Additive": 0, "Multiplicative": 0}
for metric, (add_val, mul_val) in metrics.items():
    winner = "Additive ✓" if add_val < mul_val else "Multiplicative ✓"
    if add_val < mul_val:
        scores["Additive"] += 1
    else:
        scores["Multiplicative"] += 1
    print(f"{metric:<22} {add_val:>14.4f} {mul_val:>16.4f}  {winner}")

print("-" * 60)
print(f"\n📊 TỔNG ĐIỂM:")
print(f"   Additive       : {scores['Additive']}/{len(metrics)} metrics thắng")
print(f"   Multiplicative : {scores['Multiplicative']}/{len(metrics)} metrics thắng")

recommended = "Additive" if scores["Additive"] >= scores["Multiplicative"] else "Multiplicative"
print(f"\n✅ KHUYẾN NGHỊ  : Dùng model  [{recommended.upper()}]")
print("=" * 60)

# Lý do bổ sung
add_std  = add_resid.std()
mul_std  = mul_resid.std()
if add_std < mul_std:
    ratio = mul_std / add_std
    print(f"\n💡 Residual Additive nhỏ hơn {ratio:.2f}x → seasonality KHÔNG phình theo trend")
    print("   → Data phù hợp cấu trúc ADDITIVE hơn.")
else:
    ratio = add_std / mul_std
    print(f"\n💡 Residual Multiplicative nhỏ hơn {ratio:.2f}x → seasonality PHÌNH theo trend")
    print("   → Data phù hợp cấu trúc MULTIPLICATIVE hơn.")

# ============================================================
# 5. VISUALIZE: 3 PANELS (Original + Additive + Multiplicative)
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

COLORS = {
    "data":     "#e8e8e8",
    "trend":    "#4cc9f0",
    "seasonal": "#f72585",
    "resid":    "#f8961e",
    "add":      "#06d6a0",
    "mul":      "#ffbe0b",
    "grid":     "#2a2d3a",
    "text":     "#c9d1d9",
}

def style_ax(ax, title=""):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors=COLORS["text"], labelsize=8)
    ax.spines[:].set_color(COLORS["grid"])
    ax.grid(color=COLORS["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=COLORS["text"], fontsize=9, fontweight="bold", pad=6)

# --- Row 0: Original data (full width) ---
ax_orig = fig.add_subplot(gs[0, :])
ax_orig.plot(series.index, series.values, color=COLORS["data"], linewidth=1.2, label="Original")
ax_orig.set_title("📈 Original Data", color=COLORS["text"], fontsize=11, fontweight="bold")
style_ax(ax_orig)
ax_orig.set_facecolor("#161b22")

# --- Rows 1-3: Additive (left) vs Multiplicative (right), Residual comparison (center) ---
components = ["trend", "seasonal", "resid"]
labels_add = [add_result.trend, add_result.seasonal, add_result.resid]
labels_mul = [mul_result.trend, mul_result.seasonal, mul_result.resid]
comp_colors = [COLORS["trend"], COLORS["seasonal"], COLORS["resid"]]
comp_names  = ["Trend", "Seasonal", "Residual"]

for i, (comp, add_comp, mul_comp, color, name) in enumerate(
    zip(components, labels_add, labels_mul, comp_colors, comp_names)
):
    row = i + 1

    # Additive column (left)
    ax_a = fig.add_subplot(gs[row, 0])
    ax_a.plot(series.index, add_comp, color=COLORS["add"], linewidth=1.1)
    style_ax(ax_a, f"[Additive] {name}")

    # Multiplicative column (right)
    ax_m = fig.add_subplot(gs[row, 2])
    ax_m.plot(series.index, mul_comp, color=COLORS["mul"], linewidth=1.1)
    style_ax(ax_m, f"[Multiplicative] {name}")

    # Center column: overlay comparison (only residual has meaningful comparison)
    ax_c = fig.add_subplot(gs[row, 1])
    if comp == "resid":
        ax_c.plot(series.index, add_resid.reindex(series.index),
                  color=COLORS["add"], linewidth=1.0, label=f"Add (std={add_std:.2f})", alpha=0.85)
        ax_c.plot(series.index, mul_resid.reindex(series.index),
                  color=COLORS["mul"], linewidth=1.0, label=f"Mul (std={mul_std:.2f})", alpha=0.85)
        ax_c.axhline(0, color="white", linewidth=0.5, linestyle="--")
        ax_c.legend(fontsize=7, facecolor="#161b22", labelcolor=COLORS["text"])
        style_ax(ax_c, "Residual Overlay (nhỏ hơn = tốt hơn)")
    else:
        ax_c.plot(series.index, add_comp,  color=COLORS["add"], linewidth=1.0, label="Additive",       alpha=0.85)
        ax_c.plot(series.index, mul_comp,  color=COLORS["mul"], linewidth=1.0, label="Multiplicative", alpha=0.85)
        ax_c.legend(fontsize=7, facecolor="#161b22", labelcolor=COLORS["text"])
        style_ax(ax_c, f"{name} Overlay")

# --- Header labels ---
for col, label, color in zip([0, 1, 2],
                              ["ADDITIVE", "COMPARISON", "MULTIPLICATIVE"],
                              [COLORS["add"], COLORS["data"], COLORS["mul"]]):
    fig.text(
        0.18 + col * 0.32, 0.97, label,
        ha="center", va="top",
        color=color, fontsize=12, fontweight="bold",
        transform=fig.transFigure
    )

# Winner badge
badge_color = COLORS["add"] if recommended == "Additive" else COLORS["mul"]
fig.text(
    0.5, 0.002,
    f"✅  Recommended: {recommended.upper()}  |  Add residual std={add_std:.3f}  |  Mul residual std={mul_std:.3f}",
    ha="center", va="bottom",
    color=badge_color, fontsize=10, fontweight="bold",
    transform=fig.transFigure,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2530", edgecolor=badge_color, linewidth=1.5)
)

plt.savefig("/mnt/user-data/outputs/decomposition_comparison.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\n📁 Plot đã lưu: decomposition_comparison.png")