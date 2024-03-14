import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# 加载数据
file_path = 'processed_data2.csv'  # 确保这个路径与您保存的文件路径匹配
data = pd.read_csv(file_path)

# 只保留数值型数据
data_numeric = data.select_dtypes(include=[np.number])

# 处理缺失值
imputer = SimpleImputer(strategy='mean')  # 使用列均值填充缺失值
data_imputed = imputer.fit_transform(data_numeric)

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# 应用PCA
pca = PCA(n_components=0.95)  # 选择足够的成分以解释95%的方差
pca.fit(data_scaled)

# 查看主成分解释的方差比例
print("Explained variance ratio:", pca.explained_variance_ratio_)

# 查看各个成分的累积方差解释比率
print("Cumulative explained variance ratio:", pca.explained_variance_ratio_.cumsum())
