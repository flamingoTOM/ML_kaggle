import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('./train.csv')
print(train.sample(10))


age1 = train['Age'].dropna()

"""
# 直方图
sns.histplot(
    age1,
    bins=8,               # 柱子数量
    kde=True,             # 是否画密度曲线
    color='steelblue',    # 颜色
    alpha=0.7,            # 透明度（0~1），重叠时特别有用
    stat='count',         # 纵轴含义：'count'计数 / 'density'密度 / 'probability'概率 / 'percent'百分比
    element='bars',       # 图形元素：'bars'柱状 / 'step'阶梯 / 'poly'折线
    fill=True,            # 是否填充柱子（配合 element 用）
    cumulative=False,     # 是否画累积分布（True 时会变成累积曲线）
    log_scale=False       # 纵轴是否用对数刻度
)
"""

#sns.barplot(x='Pclass', y='Count', data=train, estimator=len, errorbar=None)

# 条形图
sns.barplot(
    x='Pclass',                # 横轴变量：列名
    y='Survived',                # 纵轴变量：列名
    data=train,             # 数据源：DataFrame
    hue='Sex',              # 按某列再分组：列名（如 'Sex'）
    estimator='mean',      # 聚合方式：默认 mean；可改 len、sum、median
    errorbar=('ci', 95),   # 误差棒：None 表示不画；默认 95% 置信区间
    palette=None,          # 配色：'muted'、'Set2'、'deep' 等

)
plt.show()