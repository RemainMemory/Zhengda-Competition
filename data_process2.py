import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# 重新定义有序编码的映射
ordinal_mappings = {
    '1.您的性别是：': {'男': 0, '女': 1},
    '2.您的年龄段：': {'18岁以下': 0, '18~25': 1, '26~30': 2, '31~40': 3, '41~50': 4, '51~60': 5, '60以上': 6},
    '3.您目前的职业是：': {'学生': 0, '各类专业、技术人员': 1, '国家机关、党群组织、企事业单位的工作人员': 2,
                          '商业工作人员': 3,
                          '服务性工作人员': 4, '农林牧渔劳动者': 5, '生产工作、运输工作和部分体力劳动者': 6,
                          '其他劳动者': 7},
    '4.您目前的家庭收入情况为：': {'低保家庭': 0, '工薪家庭': 1, '小康家庭': 2, '高收入家庭': 3},
    '5.您是否了解人工智能这一概念？': {'不了解': 0, '有点了解': 1, '了解一些': 2, '非常了解': 3},
    '11. 您认为人工智能对现在的家庭生活有着怎样的影响？': {'消极': 0, '好坏参半': 1, '积极': 2},
    '14. 您是否担心人工智能会取代人类在家庭生活中的角色？': {'不担心': 0, '担心': 1},
    '15. 您是否认为人工智能在家庭生活中的应用需要更多的监管与规范？': {'不需要监管': 0, '只需要自己自觉': 1,
                                                                      '只需要立法监管': 2,
                                                                      '相关机构和个人都需要共同参与监管': 3},
    '16. 您认为未来人工智能会成为家庭生活中的主流吗？': {'不会': 0, '仍在观望': 1, '会': 2},
    '17.您是否想要在未来家庭生活中尝试使用人工智能设备？（如智能家居、健康追踪设备、无人驾驶汽车、智能娱乐设备等）': {
        '不想尝试': 0, '想要尝试': 1}
}


# 重新定义处理多选题并为特征添加唯一前缀的函数
def process_multi_choice(column_name, data, prefix):
    split_data = data[column_name].apply(lambda x: x.split('┋') if pd.notnull(x) else [])
    mlb = MultiLabelBinarizer()
    split_result = mlb.fit_transform(split_data)
    feature_names = [f"{prefix}_{feature}" for feature in mlb.classes_]
    return pd.DataFrame(split_result, columns=feature_names, index=data.index)


# 定义多选题列名和前缀
multi_choice_columns = [
    ('6.您所认为的人工智能是什么?', 'Q6'),
    ('8. 如果您拥有或使用过智能家居等人工智能设备，您认为它们对家庭生活的影响如何？', 'Q8'),
    ('9. 您认为人工智能在家庭生活中的主要应用领域是什么？', 'Q9'),
    ('10. 你认为人工智能在家庭生活中可能存在的问题是什么？', 'Q10'),
    ('12. 您认为人工智能在未来家庭生活中的发展方向是什么？', 'Q12'),
    ('13. 您是否认为人工智能在家庭生活中的发展应该注重什么？', 'Q13'),
    ('18.您最想在未来家庭生活中尝试的人工智能设备是什么？', 'Q18'),
    ('19.您不想在家庭生活中使用人工智能设备的原因是：', 'Q19')
]

# 重新加载原始数据集
full_data = pd.read_excel('2.xlsx')

# 应用有序编码到每个单选题
for col, mapping in ordinal_mappings.items():
    if col in full_data.columns:
        full_data[col] = full_data[col].map(mapping)

# 处理每个多选题并加入到数据集中
for column_name, prefix in multi_choice_columns:
    if column_name in full_data.columns:
        multi_choice_data = process_multi_choice(column_name, full_data, prefix)
        full_data = full_data.join(multi_choice_data)

# 将处理后的数据导出到CSV文件
output_csv_path = 'processed_data2.csv'
full_data.to_csv(output_csv_path, index=False)
