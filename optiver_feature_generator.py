"""
Optiver 2023 Trading at Close 专用特征生成器

基于 Auction Order Book 的特性生成针对性特征
"""

import pandas as pd
import numpy as np
from Factor import Factor
from operators import Operators


class OptiverFeatureGenerator:
    """
    Optiver 竞赛专用特征生成器
    
    针对 Trading at Close 竞赛的 Auction Order Book 特性
    生成高度相关的特征
    """
    
    def __init__(self, df):
        """
        初始化
        
        Args:
            df: 包含 train.csv 数据的 DataFrame
        """
        self.df = df.copy()
        self.original_cols = df.columns.tolist()
        
    def generate_all_features(self, group_col='stock_id', windows=[3, 5, 10]):
        """
        一键生成所有 Optiver 特征
        
        Args:
            group_col: 分组列，默认 'stock_id'
            windows: 时间窗口列表
        
        Returns:
            self
        """
        print("=" * 80)
        print("开始生成 Optiver 2023 专用特征")
        print("=" * 80)
        
        # 1. 拍卖特征（使用 Factor 类的新方法）
        print("\n[1/5] 生成拍卖相关特征...")
        self.df = Factor.auction_all_features(self.df, group_col, windows)
        
        # 2. 时序特征（使用 Operators 类）
        print("\n[2/5] 生成时序特征...")
        self._generate_time_series_features(group_col, windows)
        
        # 3. 截面特征（使用 Operators 类）
        print("\n[3/5] 生成截面特征...")
        self._generate_cross_sectional_features()
        
        # 4. 技术指标
        print("\n[4/5] 生成技术指标...")
        self._generate_technical_indicators(group_col)
        
        # 5. 高级组合特征
        print("\n[5/5] 生成高级组合特征...")
        self._generate_advanced_features(group_col, windows)
        
        print("\n" + "=" * 80)
        print(f"特征生成完成！")
        print(f"原始特征数: {len(self.original_cols)}")
        print(f"最终特征数: {len(self.df.columns)}")
        print(f"新增特征数: {len(self.df.columns) - len(self.original_cols)}")
        print("=" * 80)
        
        return self
    
    def _generate_time_series_features(self, group_col, windows):
        """生成时序特征"""
        # 关键价格列
        price_cols = ['wap', 'bid_price', 'ask_price', 'reference_price', 'near_price', 'far_price']
        
        # 关键量列
        size_cols = ['bid_size', 'ask_size', 'imbalance_size', 'matched_size']
        
        # 移动平均
        self.df = Operators.ts_mean(self.df, price_cols, windows=windows, group_col=group_col)
        self.df = Operators.ts_mean(self.df, size_cols, windows=windows, group_col=group_col)
        
        # 标准差（波动率）
        self.df = Operators.ts_std(self.df, price_cols, windows=windows, group_col=group_col)
        self.df = Operators.ts_std(self.df, size_cols, windows=windows, group_col=group_col)
        
        # 百分比变化
        self.df = Operators.ts_pct_change(self.df, price_cols, periods=[1, 3, 5], group_col=group_col)
        
        # 动量
        self.df = Operators.ts_momentum(self.df, price_cols, periods=[3, 5, 10], group_col=group_col)
        
        # EMA
        self.df = Operators.ts_ema(self.df, ['wap', 'reference_price'], spans=[5, 10, 20], group_col=group_col)
        
    def _generate_cross_sectional_features(self):
        """生成截面特征"""
        # 关键列
        key_cols = ['wap', 'reference_price', 'imbalance_size', 'matched_size', 
                    'bid_size', 'ask_size', 'auction_price_pressure']
        
        # 截面排名
        self.df = Operators.cs_rank(self.df, key_cols, group_col='time_id')
        
        # 截面标准化
        self.df = Operators.cs_zscore(self.df, key_cols, group_col='time_id')
        
        # 偏离度
        self.df = Operators.cs_deviation_from_mean(self.df, ['wap', 'imbalance_size'], group_col='time_id')
        
    def _generate_technical_indicators(self, group_col):
        """生成技术指标"""
        # RSI
        self.df = Operators.ts_rsi(self.df, 'wap', period=14, group_col=group_col)
        self.df = Operators.ts_rsi(self.df, 'reference_price', period=14, group_col=group_col)
        
        # MACD (可选，较慢)
        # self.df = Operators.ts_macd(self.df, 'wap', group_col=group_col)
        
    def _generate_advanced_features(self, group_col, windows):
        """生成高级组合特征"""
        # 1. 价格一致性指标
        self.df['price_alignment'] = (
            (self.df['near_price'] - self.df['far_price']).abs() / self.df['wap']
        )
        
        # 2. 订单簿深度
        self.df['order_book_depth'] = self.df['bid_size'] + self.df['ask_size']
        self.df['order_book_depth_ratio'] = (
            self.df['order_book_depth'] / self.df['total_liquidity'].replace(0, np.nan)
        )
        
        # 3. 价格效率
        self.df['price_efficiency'] = (
            (self.df['wap'] - self.df['reference_price']).abs() / self.df['bid_ask_spread'].replace(0, np.nan)
        )
        
        # 4. 不平衡加速度（二阶导数）
        for window in windows:
            self.df[f'imbalance_acceleration_{window}'] = (
                self.df.groupby(group_col)[f'imbalance_size_change_{window}'].diff(1)
            )
        
        # 5. 相对强度
        self.df['buy_pressure'] = self.df['bid_size'] / (self.df['bid_size'] + self.df['ask_size']).replace(0, np.nan)
        self.df['sell_pressure'] = self.df['ask_size'] / (self.df['bid_size'] + self.df['ask_size']).replace(0, np.nan)
        
        # 6. 时间加权特征
        time_weight = 1 - (self.df['seconds_in_bucket'] / 600)  # 越接近收盘权重越大
        self.df['time_weighted_imbalance'] = self.df['signed_imbalance'] * time_weight
        self.df['time_weighted_pressure'] = self.df['auction_price_pressure'] * time_weight
        
    def get_features(self):
        """获取生成特征后的 DataFrame"""
        return self.df
    
    def get_feature_names(self):
        """获取新生成的特征名称列表"""
        return [col for col in self.df.columns if col not in self.original_cols]
    
    def save_features(self, filepath='train_with_optiver_features.csv'):
        """保存特征到 CSV"""
        self.df.to_csv(filepath, index=False)
        print(f"特征已保存到: {filepath}")


def generate_optiver_features_simple(df, group_col='stock_id', windows=[5, 10, 20]):
    """
    简化版：快速生成 Optiver 特征
    
    Args:
        df: 原始 DataFrame
        group_col: 分组列
        windows: 时间窗口
    
    Returns:
        df: 添加了特征的 DataFrame
    """
    print("快速生成 Optiver 特征...")
    
    # 使用 Factor 类的一键生成方法
    df = Factor.auction_all_features(df, group_col, windows)
    
    # 添加一些关键的时序特征
    price_cols = ['wap', 'reference_price', 'near_price']
    df = Operators.ts_mean(df, price_cols, windows=windows, group_col=group_col)
    df = Operators.ts_std(df, price_cols, windows=windows, group_col=group_col)
    
    # 添加截面特征
    df = Operators.cs_rank(df, ['wap', 'imbalance_size'], group_col='time_id')
    
    print(f"完成！共 {len(df.columns)} 列")
    return df


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例 1: 使用 OptiverFeatureGenerator 类
    print("示例 1: 使用 OptiverFeatureGenerator 类")
    print("-" * 80)
    
    # 读取数据（这里用小样本测试）
    df = pd.read_csv('train.csv', nrows=10000)
    print(f"原始数据: {df.shape}")
    
    # 生成特征
    fg = OptiverFeatureGenerator(df)
    fg.generate_all_features(group_col='stock_id', windows=[5, 10, 20])
    
    # 获取结果
    df_final = fg.get_features()
    new_features = fg.get_feature_names()
    
    print(f"\n新增特征数: {len(new_features)}")
    print(f"新增特征示例: {new_features[:20]}")
    
    # 保存
    # fg.save_features('train_with_features.csv')
    
    print("\n" + "=" * 80)
    print("示例 2: 使用简化版函数")
    print("-" * 80)
    
    # 读取数据
    df2 = pd.read_csv('train.csv', nrows=10000)
    
    # 快速生成
    df2 = generate_optiver_features_simple(df2, group_col='stock_id', windows=[5, 10])
    
    print(f"最终数据: {df2.shape}")

