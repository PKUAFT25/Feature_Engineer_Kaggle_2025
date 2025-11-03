"""
高级使用示例 - Optiver Feature Engineering

演示如何自定义特征生成流程
"""

import pandas as pd
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Factor import Factor
from operators import Operators


def main():
    print("=" * 80)
    print("Optiver Feature Engineering - 高级使用示例")
    print("=" * 80)
    
    # 1. 读取数据
    print("\n[1/6] 读取数据...")
    df = pd.read_csv('../train.csv', nrows=10000)
    print(f"原始数据: {df.shape}")
    
    # 2. 处理缺失值
    print("\n[2/6] 处理缺失值...")
    df['near_price'] = df.groupby('stock_id')['near_price'].fillna(method='ffill')
    df['far_price'] = df.groupby('stock_id')['far_price'].fillna(method='ffill')
    
    # 3. 生成拍卖特征（使用 Factor 类）
    print("\n[3/6] 生成拍卖特征...")
    
    # 价格压力
    print("  - 价格压力特征")
    df = Factor.auction_price_pressure(df, group_col='stock_id')
    
    # 不平衡强度
    print("  - 不平衡强度特征")
    df = Factor.auction_imbalance_intensity(df, group_col='stock_id')
    
    # 参考价格质量
    print("  - 参考价格质量特征")
    df = Factor.auction_reference_price_quality(df, group_col='stock_id')
    
    # 流动性
    print("  - 流动性特征")
    df = Factor.auction_liquidity_profile(df, group_col='stock_id')
    
    # 价格发现
    print("  - 价格发现特征")
    df = Factor.auction_price_discovery(df, group_col='stock_id')
    
    # 订单簿压力
    print("  - 订单簿压力特征")
    df = Factor.auction_order_book_pressure(df, group_col='stock_id')
    
    # 时间动态
    print("  - 时间动态特征")
    for n in [1, 3, 5]:
        df = Factor.auction_time_dynamics(df, group_col='stock_id', n=n)
    
    # 不平衡动量
    print("  - 不平衡动量特征")
    df = Factor.auction_imbalance_momentum(df, group_col='stock_id', windows=[3, 5, 10])
    
    # 价格波动
    print("  - 价格波动特征")
    df = Factor.auction_price_volatility(df, group_col='stock_id', windows=[5, 10, 20])
    
    # 交叉特征
    print("  - 交叉特征")
    df = Factor.auction_cross_features(df, group_col='stock_id')
    
    print(f"拍卖特征生成后: {df.shape}")
    
    # 4. 生成时序特征（使用 Operators 类）
    print("\n[4/6] 生成时序特征...")
    
    # 定义关键列
    price_cols = ['wap', 'bid_price', 'ask_price', 'reference_price', 'near_price', 'far_price']
    size_cols = ['bid_size', 'ask_size', 'imbalance_size', 'matched_size']
    
    # 移动平均
    print("  - 移动平均")
    df = Operators.ts_mean(df, price_cols, windows=[5, 10, 20], group_col='stock_id')
    df = Operators.ts_mean(df, size_cols, windows=[5, 10], group_col='stock_id')
    
    # 标准差
    print("  - 标准差")
    df = Operators.ts_std(df, price_cols, windows=[5, 10, 20], group_col='stock_id')
    
    # 百分比变化
    print("  - 百分比变化")
    df = Operators.ts_pct_change(df, price_cols, periods=[1, 3, 5], group_col='stock_id')
    
    # 动量
    print("  - 动量")
    df = Operators.ts_momentum(df, price_cols, periods=[3, 5, 10], group_col='stock_id')
    
    # EMA
    print("  - EMA")
    df = Operators.ts_ema(df, ['wap', 'reference_price'], spans=[5, 10, 20], group_col='stock_id')
    
    print(f"时序特征生成后: {df.shape}")
    
    # 5. 生成截面特征（使用 Operators 类）
    print("\n[5/6] 生成截面特征...")
    
    key_cols = ['wap', 'reference_price', 'imbalance_size', 'matched_size', 
                'bid_size', 'ask_size', 'auction_price_pressure']
    
    # 截面排名
    print("  - 截面排名")
    df = Operators.cs_rank(df, key_cols, group_col='time_id')
    
    # 截面标准化
    print("  - 截面标准化")
    df = Operators.cs_zscore(df, key_cols, group_col='time_id')
    
    # 偏离度
    print("  - 偏离度")
    df = Operators.cs_deviation_from_mean(df, ['wap', 'imbalance_size'], group_col='time_id')
    
    print(f"截面特征生成后: {df.shape}")
    
    # 6. 生成技术指标
    print("\n[6/6] 生成技术指标...")
    
    # RSI
    print("  - RSI")
    df = Operators.ts_rsi(df, 'wap', period=14, group_col='stock_id')
    df = Operators.ts_rsi(df, 'reference_price', period=14, group_col='stock_id')
    
    print(f"最终数据: {df.shape}")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("特征生成总结")
    print("=" * 80)
    print(f"原始列数: 17")
    print(f"最终列数: {len(df.columns)}")
    print(f"新增列数: {len(df.columns) - 17}")
    
    # 8. 查看部分新特征
    new_cols = [col for col in df.columns if col not in [
        'stock_id', 'date_id', 'seconds_in_bucket', 'imbalance_size',
        'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
        'far_price', 'near_price', 'bid_price', 'bid_size', 'ask_price',
        'ask_size', 'wap', 'target', 'time_id', 'row_id'
    ]]
    
    print(f"\n新增特征示例（前 30 个）:")
    for i, col in enumerate(new_cols[:30], 1):
        print(f"  {i:2d}. {col}")
    
    # 9. 保存
    save_option = input("\n是否保存特征到 CSV？(y/n): ")
    if save_option.lower() == 'y':
        output_file = '../train_with_features_advanced.csv'
        df.to_csv(output_file, index=False)
        print(f"特征已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = main()

