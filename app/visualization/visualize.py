import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_dragon_fruit_data():
    """Tải dữ liệu thanh long từ CSV"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting (2).csv"
    df = pd.read_csv(csv_path)

    # Chuyển ngày sang datetime
    df['date'] = pd.to_datetime(df['ngay'])

    # Sắp xếp theo thời gian
    df = df.sort_values('date')

    return df


def plot_both_prices():
    """Vẽ biểu đồ giá ruột trắng và ruột đỏ trên cùng một biểu đồ"""

    df = load_dragon_fruit_data()

    plt.figure(figsize=(16, 8))

    # Vẽ đường giá ruột trắng
    plt.plot(
        df['date'],
        df['gia_ruot_trang_vnd_kg'],
        marker='o',
        linewidth=2,
        color='#2ecc71',
        label='Ruột Trắng (VNĐ/kg)',
        markersize=4
    )

    # Vẽ đường giá ruột đỏ
    plt.plot(
        df['date'],
        df['gia_ruot_do_vnd_kg'],
        marker='s',
        linewidth=2,
        color='#e74c3c',
        label='Ruột Đỏ (VNĐ/kg)',
        markersize=4
    )

    plt.title('Giá Thanh Long Ruột Trắng và Ruột Đỏ Theo Thời Gian',
              fontsize=16, fontweight='bold')

    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Giá (VNĐ/kg)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_both_prices()
