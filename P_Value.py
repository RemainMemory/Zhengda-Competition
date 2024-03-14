import pandas as pd
from scipy.stats import spearmanr

# 读取数据文件
data_path = 'processed_data2.csv'
full_data = pd.read_csv(data_path)

# 选定的列名
selected_columns = [
    '2.您的年龄段：', '4.您目前的家庭收入情况为：', '3.您目前的职业是：',
    '5.您是否了解人工智能这一概念？', '11. 您认为人工智能对现在的家庭生活有着怎样的影响？',
    '14. 您是否担心人工智能会取代人类在家庭生活中的角色？', '15. 您是否认为人工智能在家庭生活中的应用需要更多的监管与规范？',
    '16. 您认为未来人工智能会成为家庭生活中的主流吗？', '17.您是否想要在未来家庭生活中尝试使用人工智能设备？（如智能家居、健康追踪设备、无人驾驶汽车、智能娱乐设备等）'
]

# 初始化存储结果的DataFrame
corr_results = pd.DataFrame(index=selected_columns, columns=selected_columns)
p_value_results = pd.DataFrame(index=selected_columns, columns=selected_columns)

# 计算相关系数和p值
for col1 in selected_columns:
    for col2 in selected_columns:
        if col1 != col2:
            # 使用scipy的spearmanr函数计算相关系数和p值
            corr, p_value = spearmanr(full_data[col1], full_data[col2], nan_policy='omit')
            corr_results.loc[col1, col2] = corr
            p_value_results.loc[col1, col2] = p_value

# 保存到CSV文件
p_value_results.to_csv('ai_p_values_spearman.csv')

print("相关系数和p值已成功保存。")
