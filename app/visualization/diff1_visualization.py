import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_dragon_fruit_data():
    """Tải dữ liệu thanh long từ CSV"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting (2).csv"
    df = pd.read_csv(csv_path)

    # =========================
    # FIX COLUMN MỚI
    # =========================
    # ngay = date
    df['date'] = pd.to_datetime(df['ngay'])

    return df


# =========================
# RUỘT TRẮNG - DIFF(1)
# =========================
def plot_white_diff1():
    df = load_dragon_fruit_data()

    # sai phân bậc 1
    df['white_diff'] = df['gia_ruot_trang_vnd_kg'].diff()
    df = df.dropna()

    # Thêm smoothing với moving average 7 ngày
    df['white_diff_smooth'] = df['white_diff'].rolling(window=7).mean()

    plt.figure(figsize=(16, 8))

    # Vẽ đường gốc
    plt.plot(
        df['date'],
        df['white_diff'],
        linewidth=1,
        color='#bdc3c7',
        alpha=0.7,
        label='Ruột Trắng - Diff(1) (Gốc)'
    )

    # Vẽ đường mượt
    plt.plot(
        df['date'],
        df['white_diff_smooth'],
        linewidth=3,
        color='#2ecc71',
        label='Ruột Trắng - Diff(1) (Mượt - MA7)'
    )

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.title('Sai Phân Bậc 1 - Thanh Long Ruột Trắng (Với Smoothing)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Δ Giá (VNĐ/kg)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =========================
# RUỘT ĐỎ - DIFF(1)
# =========================
def plot_red_diff1():
    df = load_dragon_fruit_data()

    # sai phân bậc 1
    df['red_diff'] = df['gia_ruot_do_vnd_kg'].diff()
    df = df.dropna()

    plt.figure(figsize=(14, 5))

    plt.plot(
        df['date'],
        df['red_diff'],
        linewidth=2,
        color='#e74c3c',
        label='Ruột Đỏ - Diff(1)'
    )

    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.title('Sai Phân Bậc 1 - Thanh Long Ruột Đỏ',
              fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian')
    plt.ylabel('Δ Giá (VNĐ/kg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================
# RUN MAIN
# =========================
if __name__ == "__main__":
    plot_white_diff1()
    plot_red_diff1()