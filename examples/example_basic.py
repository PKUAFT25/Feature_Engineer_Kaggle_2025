"""
基础使用示例 - Optiver Feature Engineering

演示如何使用 OptiverFeatureGenerator 快速生成特征
"""

import pandas as pd
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optiver_feature_generator import OptiverFeatureGenerator


def main():
    print("=" * 80)
    print("Optiver Feature Engineering - 基础使用示例")
    print("=" * 80)
    
    # 1. 读取数据（使用小样本测试）
    print("\n[1/5] 读取数据...")
    df = pd.read_csv('../train.csv', nrows=10000)
    print(f"原始数据形状: {df.shape}")
    print(f"原始列数: {len(df.columns)}")
    
    # 2. 查看数据
    print("\n[2/5] 数据预览:")
    print(df.head())
    print(f"\n数据信息:")
    print(df.info())
    
    # 3. 处理缺失值（可选）
    print("\n[3/5] 处理缺失值...")
    print(f"Near price 缺失值: {df['near_price'].isna().sum()}")
    print(f"Far price 缺失值: {df['far_price'].isna().sum()}")
    
    # 前向填充
    df['near_price'] = df.groupby('stock_id')['near_price'].fillna(method='ffill')
    df['far_price'] = df.groupby('stock_id')['far_price'].fillna(method='ffill')
    
    print(f"处理后 Near price 缺失值: {df['near_price'].isna().sum()}")
    print(f"处理后 Far price 缺失值: {df['far_price'].isna().sum()}")
    
    # 4. 生成特征
    print("\n[4/5] 生成特征...")
    fg = OptiverFeatureGenerator(df)
    fg.generate_all_features(group_col='stock_id', windows=[5, 10, 20])
    
    # 5. 获取结果
    print("\n[5/5] 获取结果...")
    df_final = fg.get_features()
    new_features = fg.get_feature_names()
    
    print(f"\n最终数据形状: {df_final.shape}")
    print(f"原始列数: {len(df.columns)}")
    print(f"最终列数: {len(df_final.columns)}")
    print(f"新增列数: {len(new_features)}")
    
    # 6. 查看新特征
    print(f"\n前 30 个新特征:")
    for i, feat in enumerate(new_features[:30], 1):
        print(f"  {i:2d}. {feat}")
    
    # 7. 保存（可选）
    save_option = input("\n是否保存特征到 CSV？(y/n): ")
    if save_option.lower() == 'y':
        output_file = '../train_with_features_basic.csv'
        fg.save_features(output_file)
        print(f"特征已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
    
    return df_final, new_features


if __name__ == "__main__":
    df_final, new_features = main()

