"""
特征选择示例 - Optiver Feature Engineering

演示如何使用 LightGBM 进行特征重要性分析和特征选择
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optiver_feature_generator import OptiverFeatureGenerator


def main():
    print("=" * 80)
    print("Optiver Feature Engineering - 特征选择示例")
    print("=" * 80)
    
    # 1. 读取数据
    print("\n[1/6] 读取数据...")
    df = pd.read_csv('../train.csv', nrows=50000)  # 使用更多数据以获得更好的特征重要性
    print(f"原始数据: {df.shape}")
    
    # 2. 处理缺失值
    print("\n[2/6] 处理缺失值...")
    df['near_price'] = df.groupby('stock_id')['near_price'].fillna(method='ffill')
    df['far_price'] = df.groupby('stock_id')['far_price'].fillna(method='ffill')
    
    # 3. 生成特征
    print("\n[3/6] 生成特征...")
    fg = OptiverFeatureGenerator(df)
    fg.generate_all_features(group_col='stock_id', windows=[5, 10, 20])
    
    df_final = fg.get_features()
    new_features = fg.get_feature_names()
    
    print(f"生成了 {len(new_features)} 个新特征")
    
    # 4. 准备训练数据
    print("\n[4/6] 准备训练数据...")
    
    # 删除不需要的列
    exclude_cols = ['target', 'row_id', 'stock_id', 'date_id', 'time_id']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    # 删除有缺失值的行
    df_clean = df_final.dropna()
    print(f"删除缺失值后: {df_clean.shape}")
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    
    print(f"特征数: {len(feature_cols)}")
    print(f"样本数: {len(X)}")
    
    # 5. 训练模型并分析特征重要性
    print("\n[5/6] 训练模型并分析特征重要性...")
    
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")
        
        # 训练模型
        print("\n训练 LightGBM 模型...")
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            # early_stopping_rounds=10,
            verbose=False
        )
        
        # 获取特征重要性
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 6. 展示结果
        print("\n[6/6] 特征重要性分析结果")
        print("=" * 80)
        
        print(f"\nTop 30 最重要的特征:")
        print("-" * 80)
        for i, row in importance_df.head(30).iterrows():
            print(f"  {i+1:2d}. {row['feature']:50s} {row['importance']:10.2f}")
        
        print(f"\n\nBottom 20 最不重要的特征:")
        print("-" * 80)
        for i, row in importance_df.tail(20).iterrows():
            print(f"  {row['feature']:50s} {row['importance']:10.2f}")
        
        # 统计分析
        print("\n\n特征重要性统计:")
        print("-" * 80)
        print(f"总特征数: {len(importance_df)}")
        print(f"平均重要性: {importance_df['importance'].mean():.2f}")
        print(f"中位数重要性: {importance_df['importance'].median():.2f}")
        print(f"最大重要性: {importance_df['importance'].max():.2f}")
        print(f"最小重要性: {importance_df['importance'].min():.2f}")
        
        # 特征选择建议
        print("\n\n特征选择建议:")
        print("-" * 80)
        
        # Top 50 特征
        top_50_features = importance_df.head(50)['feature'].tolist()
        print(f"\nTop 50 特征（推荐使用）:")
        print(f"  特征数: {len(top_50_features)}")
        print(f"  累计重要性: {importance_df.head(50)['importance'].sum():.2f}")
        print(f"  占比: {importance_df.head(50)['importance'].sum() / importance_df['importance'].sum() * 100:.1f}%")
        
        # Top 100 特征
        top_100_features = importance_df.head(100)['feature'].tolist()
        print(f"\nTop 100 特征:")
        print(f"  特征数: {len(top_100_features)}")
        print(f"  累计重要性: {importance_df.head(100)['importance'].sum():.2f}")
        print(f"  占比: {importance_df.head(100)['importance'].sum() / importance_df['importance'].sum() * 100:.1f}%")
        
        # 保存特征重要性
        save_option = input("\n是否保存特征重要性到 CSV？(y/n): ")
        if save_option.lower() == 'y':
            importance_file = '../feature_importance.csv'
            importance_df.to_csv(importance_file, index=False)
            print(f"特征重要性已保存到: {importance_file}")
            
            # 保存 Top 50 特征列表
            top_features_file = '../top_50_features.txt'
            with open(top_features_file, 'w') as f:
                for feat in top_50_features:
                    f.write(f"{feat}\n")
            print(f"Top 50 特征列表已保存到: {top_features_file}")
        
    except ImportError:
        print("\n错误: 未安装 lightgbm")
        print("请运行: pip install lightgbm")
        return None
    
    print("\n" + "=" * 80)
    print("示例完成！")
    print("=" * 80)
    
    return importance_df


if __name__ == "__main__":
    importance_df = main()

