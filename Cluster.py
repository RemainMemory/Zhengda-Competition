import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据
file_path = 'processed_data2.csv'  # 更改为您的文件路径
data = pd.read_csv(file_path)

# 选择用于聚类的特征列
features = data.select_dtypes(include=[float, int]).columns.tolist()

# 数据标准化前处理缺失值
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
data_imputed = imputer.fit_transform(data[features])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# 执行K-Means聚类
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)  # 假设我们分为3个群集
clusters = kmeans.fit_predict(data_scaled)

# 使用PCA进行降维以便可视化
pca = PCA(n_components=2)  # 降至2维
data_pca = pca.fit_transform(data_scaled)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Cluster Visualization with PCA')
plt.show()