#!/bin/bash

# Optiver Feature Engineering - 运行示例脚本

echo "=========================================="
echo "Optiver Feature Engineering"
echo "=========================================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null
then
    echo "错误: 未找到 Python 3"
    echo "请先安装 Python 3.8+"
    exit 1
fi

echo "Python 版本:"
python3 --version
echo ""

# 检查依赖
echo "检查依赖..."
python3 -c "import pandas, numpy, scipy, sklearn, statsmodels" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖未安装"
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
fi

echo ""
echo "=========================================="
echo "选择要运行的示例:"
echo "=========================================="
echo "1. 基础示例 (example_basic.py)"
echo "2. 高级示例 (example_advanced.py)"
echo "3. 特征选择示例 (example_feature_selection.py)"
echo "4. 退出"
echo ""

read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "运行基础示例..."
        python3 examples/example_basic.py
        ;;
    2)
        echo ""
        echo "运行高级示例..."
        python3 examples/example_advanced.py
        ;;
    3)
        echo ""
        echo "运行特征选择示例..."
        python3 examples/example_feature_selection.py
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

