"""
整合的 Operators 类 - 用于快速生成大量特征
整合自: operators_11_1.py, operators_ly.py, operators_yyh.py, operator_jym.py, operator_wtq.ipynb
Authors: Zhifan, Yang Lan, Yiming Jiang, YYH, WTQ
Date: 2024/11/3
"""

import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from scipy.stats import kurtosis, skew, rankdata, mstats
from sklearn.linear_model import LinearRegression


class Operators:
    """
    统一的算子类，包含截面算子(CS)和时序算子(TS)
    
    使用方法:
        df = Operators.cs_add(df, ['col1', 'col2'])
        df = Operators.ts_mean(df, ['col1'], windows=[5, 10, 20])
    """
    
    # ==================== 截面算子 (Cross Section Operators) ====================
    
    @staticmethod
    def cs_add(df, cols):
        """截面加法 - 对列的两两组合进行加法"""
        for c in itertools.combinations(cols, 2):
            df[f'cs_add_{c[0]}_{c[1]}'] = df[c[0]] + df[c[1]]
        return df
    
    @staticmethod
    def cs_subtract(df, cols):
        """截面减法 - 对列的两两组合进行减法"""
        for c in itertools.combinations(cols, 2):
            df[f'cs_subtract_{c[0]}_{c[1]}'] = df[c[0]] - df[c[1]]
        return df
    
    @staticmethod
    def cs_multiply(df, cols):
        """截面乘法 - 对列的两两组合进行乘法"""
        for c in itertools.combinations(cols, 2):
            df[f'cs_multiply_{c[0]}_{c[1]}'] = df[c[0]] * df[c[1]]
        return df
    
    @staticmethod
    def cs_divide(df, cols):
        """截面除法 - 对列的两两组合进行除法"""
        for c in itertools.combinations(cols, 2):
            df[f'cs_divide_{c[0]}_{c[1]}'] = df[c[0]] / df[c[1]].replace(0, np.nan)
        return df
    
    @staticmethod
    def cs_zscore_add(df, cols, group_col='time_id'):
        """先对每列进行截面标准化，再进行两两加法"""
        for col in cols:
            df[f"cs_zscore_{col}"] = (df[col] - df.groupby([group_col])[col].transform('mean')) / df.groupby([group_col])[col].transform('std')
        for c in itertools.combinations(cols, 2):
            df[f'cs_zscore_add_{c[0]}_{c[1]}'] = df[f'cs_zscore_{c[0]}'] + df[f'cs_zscore_{c[1]}']
        return df
    
    @staticmethod
    def cs_rank_add(df, cols, group_col='time_id'):
        """先对每列进行截面排名，再进行两两加法"""
        for col in cols:
            df[f"cs_rank_{col}"] = df.groupby([group_col])[col].rank(pct=True)
        for c in itertools.combinations(cols, 2):
            df[f'cs_rank_add_{c[0]}_{c[1]}'] = df[f'cs_rank_{c[0]}'] + df[f'cs_rank_{c[1]}']
        return df
    
    @staticmethod
    def cs_max(df, cols):
        """截面最大值 - 每行取多列的最大值"""
        df['cs_max'] = df[cols].max(axis=1)
        return df
    
    @staticmethod
    def cs_min(df, cols):
        """截面最小值 - 每行取多列的最小值"""
        df['cs_min'] = df[cols].min(axis=1)
        return df
    
    @staticmethod
    def cs_range(df, cols):
        """截面范围 - 每行的最大值减最小值"""
        df['cs_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)
        return df
    
    @staticmethod
    def cs_mean(df, cols):
        """截面均值 - 每行取多列的均值"""
        df['cs_mean'] = df[cols].mean(axis=1)
        return df
    
    @staticmethod
    def cs_std(df, cols):
        """截面标准差 - 每行取多列的标准差"""
        df['cs_std'] = df[cols].std(axis=1)
        return df
    
    @staticmethod
    def cs_rank(df, cols, group_col='time_id'):
        """截面排名 - 在每个时间点内对指定列进行排名"""
        for col in cols:
            df[f'cs_rank_{col}'] = df.groupby([group_col])[col].rank(pct=True)
        return df
    
    @staticmethod
    def cs_zscore(df, cols, group_col='time_id'):
        """截面标准化 - 在每个时间点内进行标准化"""
        for col in cols:
            df[f'cs_zscore_{col}'] = (df[col] - df.groupby([group_col])[col].transform('mean')) / df.groupby([group_col])[col].transform('std')
        return df
    
    @staticmethod
    def cs_deviation_from_mean(df, cols, group_col='time_id'):
        """偏离均值 - 计算每个值与组内均值的偏离"""
        for col in cols:
            df[f'cs_dev_mean_{col}'] = df[col] - df.groupby([group_col])[col].transform('mean')
        return df
    
    @staticmethod
    def cs_deviation_from_max(df, cols, group_col='time_id'):
        """偏离最大值 - 计算每个值与组内最大值的偏离"""
        for col in cols:
            df[f'cs_dev_max_{col}'] = df[col] - df.groupby([group_col])[col].transform('max')
        return df
    
    @staticmethod
    def cs_deviation_from_min(df, cols, group_col='time_id'):
        """偏离最小值 - 计算每个值与组内最小值的偏离"""
        for col in cols:
            df[f'cs_dev_min_{col}'] = df[col] - df.groupby([group_col])[col].transform('min')
        return df
    
    @staticmethod
    def cs_cross_feature(df, cols):
        """交叉特征 - 生成列之间的交叉乘积"""
        for c in itertools.combinations(cols, 2):
            df[f'cs_cross_{c[0]}_{c[1]}'] = df[c[0]] * df[c[1]]
        return df
    
    @staticmethod
    def cs_weighted_sum(df, cols, weights):
        """加权求和 - 对多列进行加权求和"""
        df['cs_weighted_sum'] = sum(df[col] * weight for col, weight in zip(cols, weights))
        return df
    
    # ==================== 时序算子 (Time Series Operators) ====================
    
    @staticmethod
    def ts_mean(df, cols, windows=[], group_col='stock_id'):
        """时序均值 - 计算滚动窗口均值"""
        for col in cols:
            for window in windows:
                df[f"ts_mean_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        return df
    
    @staticmethod
    def ts_std(df, cols, windows=[], group_col='stock_id'):
        """时序标准差 - 计算滚动窗口标准差"""
        for col in cols:
            for window in windows:
                df[f"ts_std_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        return df
    
    @staticmethod
    def ts_sum(df, cols, windows=[], group_col='stock_id'):
        """时序求和 - 计算滚动窗口求和"""
        for col in cols:
            for window in windows:
                df[f"ts_sum_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum()
                )
        return df
    
    @staticmethod
    def ts_max(df, cols, windows=[], group_col='stock_id'):
        """时序最大值 - 计算滚动窗口最大值"""
        for col in cols:
            for window in windows:
                df[f"ts_max_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
        return df
    
    @staticmethod
    def ts_min(df, cols, windows=[], group_col='stock_id'):
        """时序最小值 - 计算滚动窗口最小值"""
        for col in cols:
            for window in windows:
                df[f"ts_min_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
        return df
    
    @staticmethod
    def ts_diff(df, cols, periods=[], group_col='stock_id'):
        """时序差分 - 计算与前N期的差值"""
        for col in cols:
            for period in periods:
                df[f"ts_diff_{col}_{period}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.diff(periods=period)
                )
        return df
    
    @staticmethod
    def ts_pct_change(df, cols, periods=[], group_col='stock_id'):
        """时序百分比变化 - 计算收益率"""
        for col in cols:
            for period in periods:
                df[f"ts_pct_change_{col}_{period}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.pct_change(periods=period)
                )
        return df
    
    @staticmethod
    def ts_log_return(df, cols, periods=[], group_col='stock_id'):
        """时序对数收益率"""
        for col in cols:
            for period in periods:
                df[f'ts_log_return_{col}_{period}'] = df.groupby([group_col])[col].transform(
                    lambda x: np.log(x / x.shift(period))
                )
        return df
    
    @staticmethod
    def ts_zscore(df, cols, windows=[], group_col='stock_id'):
        """时序标准化 - 滚动窗口内标准化"""
        for col in cols:
            for window in windows:
                df[f"ts_zscore_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: (x - x.rolling(window=window, min_periods=1).mean()) / x.rolling(window=window, min_periods=1).std()
                )
        return df
    
    @staticmethod
    def ts_rank(df, cols, windows=[], group_col='stock_id'):
        """时序排名 - 滚动窗口内排名"""
        for col in cols:
            for window in windows:
                df[f"ts_rank_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).rank(pct=True)
                )
        return df

    @staticmethod
    def ts_corr(df, col_pairs, windows=[], group_col='stock_id'):
        """时序相关性 - 计算两列的滚动相关系数"""
        for col1, col2 in col_pairs:
            for window in windows:
                df[f"ts_corr_{col1}_{col2}_{window}"] = df.groupby([group_col]).apply(
                    lambda x: x[col1].rolling(window=window, min_periods=1).corr(x[col2])
                ).reset_index(level=0, drop=True)
        return df

    @staticmethod
    def ts_slope(df, cols, windows=[], group_col='stock_id'):
        """时序斜率 - 计算线性回归斜率"""
        for col in cols:
            for window in windows:
                df[f"ts_slope_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else np.nan
                    )
                )
        return df

    @staticmethod
    def ts_skew(df, cols, windows=[], group_col='stock_id'):
        """时序偏度 - 计算滚动窗口偏度"""
        for col in cols:
            for window in windows:
                df[f"ts_skew_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=3).skew()
                )
        return df

    @staticmethod
    def ts_kurt(df, cols, windows=[], group_col='stock_id'):
        """时序峰度 - 计算滚动窗口峰度"""
        for col in cols:
            for window in windows:
                df[f"ts_kurt_{col}_{window}"] = df.groupby([group_col])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=4).kurt()
                )
        return df

    @staticmethod
    def ts_ema(df, cols, spans=[], group_col='stock_id'):
        """时序指数移动平均 (EMA)"""
        for col in cols:
            for span in spans:
                df[f'ts_ema_{col}_{span}'] = df.groupby([group_col])[col].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )
        return df

    @staticmethod
    def ts_lag(df, cols, periods=[], group_col='stock_id'):
        """时序滞后特征 - 创建滞后N期的特征"""
        for col in cols:
            for period in periods:
                df[f'ts_lag_{col}_{period}'] = df.groupby([group_col])[col].shift(period)
        return df

    @staticmethod
    def ts_momentum(df, cols, periods=[], group_col='stock_id'):
        """时序动量 - 计算过去N期的变化"""
        for col in cols:
            for period in periods:
                df[f'ts_momentum_{col}_{period}'] = df.groupby([group_col])[col].transform(
                    lambda x: x.diff(periods=period)
                )
        return df

    @staticmethod
    def ts_rsi(df, price_col, period=14, group_col='stock_id'):
        """相对强弱指数 (RSI)"""
        def calc_rsi(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df[f'ts_rsi_{price_col}_{period}'] = df.groupby([group_col])[price_col].transform(calc_rsi)
        return df

    @staticmethod
    def ts_macd(df, price_col, short_window=12, long_window=26, signal_window=9, group_col='stock_id'):
        """移动平均收敛散布指标 (MACD)"""
        def calc_macd(x):
            short_ema = x.ewm(span=short_window, adjust=False).mean()
            long_ema = x.ewm(span=long_window, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=signal_window, adjust=False).mean()
            histogram = macd - signal
            return pd.DataFrame({
                'macd': macd,
                'signal': signal,
                'histogram': histogram
            })

        grouped = df.groupby([group_col])[price_col].apply(calc_macd)
        df[f'ts_macd_{price_col}'] = grouped['macd'].values
        df[f'ts_macd_signal_{price_col}'] = grouped['signal'].values
        df[f'ts_macd_hist_{price_col}'] = grouped['histogram'].values
        return df

    @staticmethod
    def ts_bollinger_bands(df, price_col, window=20, num_sd=2, group_col='stock_id'):
        """布林带 (Bollinger Bands)"""
        def calc_bb(x):
            ma = x.rolling(window=window).mean()
            std = x.rolling(window=window).std()
            upper = ma + (std * num_sd)
            lower = ma - (std * num_sd)
            return pd.DataFrame({
                'ma': ma,
                'upper': upper,
                'lower': lower,
                'width': upper - lower,
                'pct_b': (x - lower) / (upper - lower)
            })

        grouped = df.groupby([group_col])[price_col].apply(calc_bb)
        df[f'ts_bb_ma_{price_col}_{window}'] = grouped['ma'].values
        df[f'ts_bb_upper_{price_col}_{window}'] = grouped['upper'].values
        df[f'ts_bb_lower_{price_col}_{window}'] = grouped['lower'].values
        df[f'ts_bb_width_{price_col}_{window}'] = grouped['width'].values
        df[f'ts_bb_pct_{price_col}_{window}'] = grouped['pct_b'].values
        return df

    # ==================== 原子操作算子 (Atomic Operators) ====================

    @staticmethod
    def at_abs(df, cols):
        """绝对值"""
        for col in cols:
            df[f'at_abs_{col}'] = df[col].abs()
        return df

    @staticmethod
    def at_square(df, cols):
        """平方"""
        for col in cols:
            df[f'at_square_{col}'] = df[col] ** 2
        return df

    @staticmethod
    def at_cube(df, cols):
        """立方"""
        for col in cols:
            df[f'at_cube_{col}'] = df[col] ** 3
        return df

    @staticmethod
    def at_log(df, cols):
        """对数变换 (加1防止对0取对数)"""
        for col in cols:
            df[f'at_log_{col}'] = np.log(df[col] + 1)
        return df

    @staticmethod
    def at_sign(df, cols):
        """符号函数"""
        for col in cols:
            df[f'at_sign_{col}'] = np.sign(df[col])
        return df

    @staticmethod
    def at_sigmoid(df, cols):
        """Sigmoid 函数"""
        for col in cols:
            df[f'at_sigmoid_{col}'] = 1 / (1 + np.exp(-df[col]))
        return df

    @staticmethod
    def at_signlog(df, cols):
        """带符号的对数"""
        for col in cols:
            df[f'at_signlog_{col}'] = np.sign(df[col]) * np.log(df[col].abs() + 1)
        return df

    @staticmethod
    def at_signsqrt(df, cols):
        """带符号的平方根"""
        for col in cols:
            df[f'at_signsqrt_{col}'] = np.sign(df[col]) * np.sqrt(df[col].abs())
        return df

