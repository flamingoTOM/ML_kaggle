import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.labelsize': 16,   # 横纵轴标题
    'xtick.labelsize': 14,  # 横轴刻度
    'ytick.labelsize': 14,  # 纵轴刻度
    'legend.fontsize': 12,  # 图例文字
    'legend.title_fontsize': 14,  # 图例标题（Sex）
})

train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
gender_submission = pd.read_csv('./data/gender_submission.csv')

print(train.sample(10))

age1 = train['Age'].dropna()

'''
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
plt.savefig('./pic/histplot.png', dpi=300, bbox_inches='tight')
'''


"""
# 条形图
sns.barplot(
    x='Pclass',                # 横轴变量：列名
    y='Survived',                # 纵轴变量：列名
    data=train,             # 数据源：DataFrame
    hue='Sex',              # 按某列再分组：列名（如 'Sex'）
    errorbar=('ci', 95),   # 误差棒：None 表示不画；默认 95% 置信区间
    palette={'female': '#C87965', 'male': '#527D9C'}   # 女暖男冷

)
plt.savefig('./pic/barplot.png', dpi=300, bbox_inches='tight')
"""

"""
# 计数图
# 从Cabin信息中提取出 deck:每位乘客所在的甲板编号
train['Deck'] = train['Cabin'].str[0].fillna('Unknown')
print(train['Deck'].value_counts(dropna=False))

sns.countplot(
    x='Deck',
    data=train,
    order=sorted(train['Deck'].unique()),
    hue='Sex',
    color='#527D9C',
    palette={'female': '#C87965', 'male': '#527D9C'}
)

plt.savefig('./pic/deck_sex.png', dpi=300, bbox_inches='tight')
"""

'''
#散点图
sns.stripplot(
    x='Embarked',
    y='Fare',
    data=train,
    jitter=0.3,
    alpha=0.6,
    size=4,
    color='#527D9C',
)

plt.savefig('./pic/embarked_fare.png', dpi=300, bbox_inches='tight')

'''

