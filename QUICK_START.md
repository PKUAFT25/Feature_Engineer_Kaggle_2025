# 快速参考

## 一键生成特征

```python
import pandas as pd
from optiver_feature_generator import OptiverFeatureGenerator

df = pd.read_csv('train.csv')
fg = OptiverFeatureGenerator(df)
fg.generate_all_features(group_col='stock_id', windows=[5, 10, 20])
df_final = fg.get_features()
```

## Factor 类 - Optiver 专用因子

```python
from Factor import Factor

# 一键生成所有 Optiver 特征
df = Factor.auction_all_features(df, group_col='stock_id', windows=[5, 10])

# 或单独生成
df = Factor.auction_price_pressure(df)          # 价格压力
df = Factor.auction_imbalance_intensity(df)     # 不平衡强度
df = Factor.auction_liquidity_profile(df)       # 流动性
df = Factor.auction_price_discovery(df)         # 价格发现
df = Factor.auction_order_book_pressure(df)     # 订单簿压力
```

## Operators 类 - 通用算子

### 时序算子 (TS)

```python
from operators import Operators

# 移动平均
df = Operators.ts_mean(df, ['wap'], windows=[5, 10, 20], group_col='stock_id')

# 标准差
df = Operators.ts_std(df, ['wap'], windows=[5, 10, 20], group_col='stock_id')

# 百分比变化
df = Operators.ts_pct_change(df, ['wap'], periods=[1, 3, 5], group_col='stock_id')

# EMA
df = Operators.ts_ema(df, ['wap'], spans=[5, 10, 20], group_col='stock_id')

# RSI
df = Operators.ts_rsi(df, 'wap', period=14, group_col='stock_id')
```

### 截面算子 (CS)

```python
# 截面排名
df = Operators.cs_rank(df, ['wap'], group_col='time_id')

# 截面标准化
df = Operators.cs_zscore(df, ['wap'], group_col='time_id')
```

## 推荐配置

```python
# 推荐时间窗口
windows = [5, 10, 20]

# 推荐列
price_cols = ['wap', 'reference_price', 'near_price']
size_cols = ['imbalance_size', 'matched_size']
```

## 处理缺失值

```python
# Near/Far price 在 3:50-3:55 缺失，需要填充
df['near_price'] = df.groupby('stock_id')['near_price'].fillna(method='ffill')
df['far_price'] = df.groupby('stock_id')['far_price'].fillna(method='ffill')
```

## 运行示例

```bash
# 运行示例脚本
./run_example.sh

# 或直接运行
python examples/example_basic.py
python examples/example_advanced.py
python examples/example_feature_selection.py
```

