import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_dragon_fruit_data():
    """Tải dữ liệu thanh long từ CSV"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting.csv"
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['month'])
    return df


def plot_white_diff1():
    df = load_dragon_fruit_data()
    df['white_diff'] = df['price_white_vnd'].diff()
    df = df.dropna()

    plt.figure(figsize=(14, 5))
    plt.plot(df['date'], df['white_diff'],
             linewidth=2, color='#2ecc71', label='Trắng Diff (d=1)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Sai Phân Bậc 1 — Thanh Long Trắng', fontsize=13, fontweight='bold')
    plt.xlabel('Thời gian')
    plt.ylabel('Δ Giá (VNĐ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_red_diff1():
    df = load_dragon_fruit_data()
    df['red_diff'] = df['price_red_vnd'].diff()
    df = df.dropna()

    plt.figure(figsize=(14, 5))
    plt.plot(df['date'], df['red_diff'],
             linewidth=2, color='#e74c3c', label='Đỏ Diff (d=1)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('Sai Phân Bậc 1 — Thanh Long Đỏ', fontsize=13, fontweight='bold')
    plt.xlabel('Thời gian')
    plt.ylabel('Δ Giá (VNĐ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_white_diff1()
    plot_red_diff1()