import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# LOAD DATA
# =========================
def load_dragon_fruit_data():
    """Tải dữ liệu thanh long từ CSV"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting (2).csv"
    df = pd.read_csv(csv_path)

    df['date'] = pd.to_datetime(df['ngay'])
    return df


# =========================
# SMOOTHING FUNCTION
# =========================
def plot_smoothing_white():
    df = load_dragon_fruit_data()
    df = df.sort_values('date')

    # 👉 smoothing (trung bình trượt 7 ngày)
    df['white_smooth'] = df['gia_ruot_trang_vnd_kg'].rolling(window=7).mean()

    plt.figure(figsize=(14, 6))

    # giá gốc (nhẹ thôi)
    plt.plot(df['date'], df['gia_ruot_trang_vnd_kg'],
             color='gray', alpha=0.3, label='Raw Data')

    # đường mượt
    plt.plot(df['date'], df['white_smooth'],
             color='#2ecc71', linewidth=2.5,
             label='White Smoothed (MA 7)')

    plt.title('Thanh Long Ruột Trắng - Smoothing Visualization',
              fontsize=14, fontweight='bold')

    plt.xlabel('Thời gian')
    plt.ylabel('Giá (VNĐ/kg)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
def plot_smoothing_red():
    df = load_dragon_fruit_data()
    df = df.sort_values('date')

    # smoothing
    df['red_smooth'] = df['gia_ruot_do_vnd_kg'].rolling(window=7).mean()

    plt.figure(figsize=(14, 6))

    # raw data
    plt.plot(df['date'], df['gia_ruot_do_vnd_kg'],
             color='gray', alpha=0.3, label='Raw Data')

    # smoothed line
    plt.plot(df['date'], df['red_smooth'],
             color='#e74c3c', linewidth=2.5,
             label='Red Smoothed (MA 7)')

    plt.title('Thanh Long Ruột Đỏ - Smoothing Visualization',
              fontsize=14, fontweight='bold')

    plt.xlabel('Thời gian')
    plt.ylabel('Giá (VNĐ/kg)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    plot_smoothing_white()
    plot_smoothing_red()