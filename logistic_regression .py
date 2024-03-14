import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# 加载数据
# 请确保替换为你的实际文件路径
file_path = 'processed_data2.csv'
full_data = pd.read_csv(file_path)

# 定义自变量和目标变量的列名
X_columns = ['5.您是否了解人工智能这一概念？', '2.您的年龄段：', '4.您目前的家庭收入情况为：',
             '3.您目前的职业是：', '14. 您是否担心人工智能会取代人类在家庭生活中的角色？',
             '16. 您认为未来人工智能会成为家庭生活中的主流吗？']
y_column = '17.您是否想要在未来家庭生活中尝试使用人工智能设备？（如智能家居、健康追踪设备、无人驾驶汽车、智能娱乐设备等）'

# 移除目标变量y中含有缺失值的行
full_data_cleaned = full_data.dropna(subset=[y_column])

# 重新选择自变量和目标变量
X = full_data_cleaned[X_columns]
y = full_data_cleaned[y_column]

# 再次分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用SimpleImputer和LogisticRegression的管道
imputer = SimpleImputer(strategy='mean')
log_reg = LogisticRegression(max_iter=1000)
pipeline = make_pipeline(imputer, log_reg)

# 训练模型
pipeline.fit(X_train, y_train)

# 进行预测
y_pred = pipeline.predict(X_test)

# 评估模型性能
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
