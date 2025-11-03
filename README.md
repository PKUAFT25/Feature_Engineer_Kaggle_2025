# Optiver 2023 Trading at Close - 特征工程项目

> 专为 Optiver 2023 Trading at Close Kaggle 竞赛设计的特征工程工具包

---

## 项目简介

本项目针对 **Optiver 2023 Trading at Close** Kaggle 竞赛，提供完整的特征工程解决方案。

**核心功能**:
- ✅ **Factor 类**: 70+ 个因子方法（含 11 个 Optiver 专用方法）
- ✅ **Operators 类**: 60+ 个算子（时序、截面、原子操作）
- ✅ **一键生成**: 快速生成 100+ 个特征

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用方式

#### 方式一：一键生成（推荐新手）

```python
import pandas as pd
from optiver_feature_generator import OptiverFeatureGenerator

df = pd.read_csv('train.csv')
fg = OptiverFeatureGenerator(df)
fg.generate_all_features(group_col='stock_id', windows=[5, 10, 20])
df_final = fg.get_features()
```

#### 方式二：使用 Factor 类

```python
from Factor import Factor

df = Factor.auction_price_pressure(df)
df = Factor.auction_imbalance_intensity(df)
df = Factor.auction_all_features(df)  # 一键生成所有 Optiver 特征
```

#### 方式三：使用 Operators 类

```python
from operators import Operators

# 时序特征
df = Operators.ts_mean(df, ['wap'], windows=[5, 10, 20], group_col='stock_id')
df = Operators.ts_std(df, ['wap'], windows=[5, 10, 20], group_col='stock_id')

# 截面特征
df = Operators.cs_rank(df, ['wap'], group_col='time_id')
df = Operators.cs_zscore(df, ['imbalance_size'], group_col='time_id')
```

---

## 项目结构

```
optiver_feature_engineering/
├── README.md                           # 使用说明
├── QUICK_START.md                      # 快速参考
├── requirements.txt                    # 依赖列表
├── run_example.sh                      # 运行示例脚本
│
├── Factor.py                           # 因子类（70+ 方法）
├── operators.py                        # 算子类（60+ 方法）
├── optiver_feature_generator.py       # 特征生成器
│
├── train.csv                           # 训练数据
├── multilinear_and_tree-feature.ipynb # 特征分析 Notebook
│
└── examples/                           # 示例脚本
    ├── example_basic.py               # 基础示例
    ├── example_advanced.py            # 高级示例
    └── example_feature_selection.py   # 特征选择示例
```

---

## 核心功能

### Factor 类 - Optiver 专用因子

11 个专门针对 Auction Order Book 的因子方法：

| 方法 | 功能 |
|------|------|
| `auction_price_pressure` | 拍卖价格压力 |
| `auction_imbalance_intensity` | 不平衡强度 |
| `auction_reference_price_quality` | 参考价格质量 |
| `auction_liquidity_profile` | 流动性特征 |
| `auction_price_discovery` | 价格发现 |
| `auction_order_book_pressure` | 订单簿压力 |
| `auction_time_dynamics` | 时间动态 |
| `auction_imbalance_momentum` | 不平衡动量 |
| `auction_price_volatility` | 价格波动 |
| `auction_cross_features` | 交叉特征 |
| **`auction_all_features`** | **一键生成所有特征** |

### Operators 类 - 60+ 个算子

- **截面算子 (CS)**: `cs_rank`, `cs_zscore`, `cs_add`, `cs_multiply` 等
- **时序算子 (TS)**: `ts_mean`, `ts_std`, `ts_ema`, `ts_rsi`, `ts_macd` 等
- **原子操作 (AT)**: `at_abs`, `at_log`, `at_square`, `at_sigmoid` 等

---

## 运行示例

```bash
# 运行示例脚本
./run_example.sh

# 或直接运行
python examples/example_basic.py
python examples/example_advanced.py
python examples/example_feature_selection.py
```

---

## 常见问题

### 1. Near Price 和 Far Price 有缺失值怎么办？

```python
# 前向填充
df['near_price'] = df.groupby('stock_id')['near_price'].fillna(method='ffill')
df['far_price'] = df.groupby('stock_id')['far_price'].fillna(method='ffill')
```

### 2. 如何选择时间窗口？

推荐使用 `windows=[5, 10, 20]`：
- 5: 捕捉短期变化
- 10: 捕捉中期趋势
- 20: 捕捉长期模式

### 3. 如何避免生成过多特征？

```python
# 只对关键列生成特征
core_cols = ['wap', 'reference_price', 'imbalance_size']
df = Operators.ts_mean(df, core_cols, windows=[5, 10], group_col='stock_id')
core_cols = ['wap', 'reference_price', 'near_price', 'imbalance_size', 'matched_size']
```

### 4. 如何进行特征选择？

运行 `examples/example_feature_selection.py` 查看完整示例。

---

## 项目文件说明

- **Factor.py**: 核心因子类，包含 70+ 个因子方法
- **operators.py**: 算子类，包含 60+ 个算子
- **optiver_feature_generator.py**: 一键生成特征的工具类
- **examples/**: 示例脚本，演示不同使用方式
- **QUICK_START.md**: 快速参考手册

---

祝你在 Optiver 竞赛中取得优异成绩！
