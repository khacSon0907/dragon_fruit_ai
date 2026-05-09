import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_dragon_fruit_data():
    """Tải dữ liệu thanh long từ CSV"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "processed" / "dragon_fruit_price_forecasting.csv"
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['month'])
    return df

def plot_price_chart():
    """Vẽ biểu đồ giá thanh long trắng và đỏ theo thời gian"""
    df = load_dragon_fruit_data()
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(df['date'], df['price_white_vnd'], marker='o', linewidth=2, label='Thanh Long Trắng (VNĐ)', color='#2ecc71')
    plt.plot(df['date'], df['price_red_vnd'], marker='s', linewidth=2, label='Thanh Long Đỏ (VNĐ)', color='#e74c3c')
    
    plt.xlabel('Thời gian', fontsize=12, fontweight='bold')
    plt.ylabel('Giá (VNĐ)', fontsize=12, fontweight='bold')
    plt.title('Biểu Đồ Giá Thanh Long Trắng và Đỏ Theo Thời Gian', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_price_chart()
