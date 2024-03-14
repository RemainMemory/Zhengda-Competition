# 计算Spearman相关系数
from data_process2 import full_data

correlation_matrix = full_data[['2.您的年龄段：', '4.您目前的家庭收入情况为：', '3.您目前的职业是：',
                                '5.您是否了解人工智能这一概念？',
                                '11. 您认为人工智能对现在的家庭生活有着怎样的影响？',
                                '14. 您是否担心人工智能会取代人类在家庭生活中的角色？',
                                '15. 您是否认为人工智能在家庭生活中的应用需要更多的监管与规范？',
                                '16. 您认为未来人工智能会成为家庭生活中的主流吗？',
                                '17.您是否想要在未来家庭生活中尝试使用人工智能设备？（如智能家居、健康追踪设备、无人驾驶汽车、智能娱乐设备等）'
                                ]].corr(method='spearman')

# 提取与人工智能相关的部分
ai_correlation = correlation_matrix[['5.您是否了解人工智能这一概念？',
                                     '11. 您认为人工智能对现在的家庭生活有着怎样的影响？',
                                     '14. 您是否担心人工智能会取代人类在家庭生活中的角色？',
                                     '15. 您是否认为人工智能在家庭生活中的应用需要更多的监管与规范？',
                                     '16. 您认为未来人工智能会成为家庭生活中的主流吗？',
                                     '17.您是否想要在未来家庭生活中尝试使用人工智能设备？（如智能家居、健康追踪设备、无人驾驶汽车、智能娱乐设备等）'
                                    ]]

# 保存相关系数到CSV文件
output_file_path = 'ai_correlation.csv'
ai_correlation.to_csv(output_file_path)

print(f"相关系数已成功保存到 {output_file_path}")


