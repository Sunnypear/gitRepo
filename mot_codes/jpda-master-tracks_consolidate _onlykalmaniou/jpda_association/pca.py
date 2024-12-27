import numpy as np
from sklearn.decomposition import PCA

# 输入待降维数据 (5 * 6) 矩阵，6个维度，5个样本值
# A = np.array(
#     [[-2,-2],[-2.5,-2.5],[-0.5,-0.5],[-1,-1],[-1.5,-1.5]])
A = np.array([[1282,466],[1283,466]])
B = np.array([[1282,466],[1282,467]])
# B = np.array([[0.5,0.25],[1,0.5],[2.5,1.25],[1.5,0.75],[2,1]])
# 直接使用PCA进行降维
pca_A = PCA(n_components=1)  # 降到 2 维
pca_B = PCA(n_components=1)  # 降到 2 维
pca_A.fit(A)
pca_B.fit(B)
# PCA(n_components=1)
A_1 = pca_A.transform(A)  # 降维后的结果
B_1 = pca_B.transform(B)
eigenvalues_A = pca_A.explained_variance_
eigenvectors_A = pca_A.components_
eigenvalues_B = pca_B.explained_variance_
eigenvectors_B = pca_B.components_
A_2 = eigenvectors_A@A.T
B_2 = eigenvectors_B@B.T
temp = np.dot(eigenvectors_A,eigenvectors_B.T)
e = np.arccos(np.dot(eigenvectors_A,eigenvectors_B.T)/(np.sqrt(eigenvectors_A[0,0]**2+eigenvectors_A[0,1]**2)\
                                             *np.sqrt(eigenvectors_B[0,0]**2+eigenvectors_B[0,1]**2)))
e = e*180/np.pi
f = eigenvectors_B[0,0]**2 + eigenvectors_B[0,1]**2
g = eigenvectors_A[0,0]**2 + eigenvectors_A[0,1]**2
h = np.sqrt(5)/4
C= np.tan(np.pi/6)
print(A)
# pca.explained_variance_ratio_  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
# pca.explained_variance_  # 降维后的各主成分的方差值
