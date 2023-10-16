import numpy as np
import pandas as pd
from utils import Nantozero,replace_text_with_zero
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# #读取文件
# c1=pd.read_excel('C:/Users/KDG/PycharmProjects/clurster/row_data/수질측정망데이터/전국수질측정자료_2011년_2015년_일자료.xlsx',index_col='No')
# print(c1.head())
#
# c2=pd.read_excel('C:/Users/KDG/PycharmProjects/clurster/row_data/수질측정망데이터/전국수질측정자료_2016년_2021년_일자료.xlsx')
# print(c2.head())
#
# c3=pd.read_excel('C:/Users/KDG/PycharmProjects/clurster/row_data/수질측정망데이터/전국수질측정자료_2022년_일자료.xlsx')
# print(c3.head())
#

# #数据前处理 删除列
# drop_colums=['음이온계면활성제(ABS)','안티몬(Sb)','비소(As)','바륨(Ba)','벤젠','사염화탄소','카드뮴(Cd)','클로로포름','염소이온(Cl-)','시안(CN)','색도','크롬(Cr)','6가크롬(Cr6+)','구리(Cu)',
#              '1,2,-다이클로로에탄','다이클로로메탄','다이에틸헥실프탈레이트(DEHP)','1,4-다이옥세인','용해성 철(Fe)','불소(F)','헥사클로로벤젠','포름알데히드','수은(Hg)','용해성 망간(Mn)','노말헥산추출물질',
#              '니켈(Ni)','유기인','납(Pb)','폴리클로리네이티드비페닐(PCB)','테트라클로로에틸렌(PCE)','페놀류(phenols)','셀레늄(Se)','트리클로로에틸렌(TCE)','투명도','아연(Zn)']
# #共删除35列
# print(len(drop_colums))
#
# #处理之后为27列（原先有62列）
# df1 = c1.drop(columns=drop_colums)
# df2 = c2.drop(columns=drop_colums)
# df3 = c3.drop(columns=drop_colums)
#
#
# #将处理后文件进行储存
# df1.to_excel('file1.xlsx', index=False)
# df2.to_excel('file2.xlsx', index=False)
# df3.to_excel('file3.xlsx', index=False)

##读取处理后的文件
# f1=pd.read_excel('C:/Users/KDG/PycharmProjects/clurster/file1.xlsx')
# # f2=pd.read_excel('C:/Users/KDG/PycharmProjects/clurster/file1.xlsx')
# # f3=pd.read_excel('C:/Users/KDG/PycharmProjects/clurster/file1.xlsx')
#
# print(f1.head(10))

#将数据的空缺值，用0处理
file1=Nantozero('file1.xlsx')
# print(file1.iloc[:, 7:])

# 去除dataframe里的字符串，使用 applymap 方法应用函数到整个 DataFrame
df1 = file1.iloc[:, 7:].applymap(replace_text_with_zero)



#使用kmeans方法，因为生物标准有5个等级，所以类别为5
kmeans = KMeans(n_clusters=5)
# 使用数据拟合 KMeans 模型
kmeans.fit(df1)
# 获取每个样本所属的簇
labels = kmeans.labels_
# 获取簇的中心点
centroids = kmeans.cluster_centers_


# 使用 PCA 进行降维,方便画图
pca = PCA(n_components=2)
data_2d = pca.fit_transform(df1)
# centroids_2d = pca.transform(centroids)

# 初始化标准化器
scaler = StandardScaler()
# 使用标准化器拟合并转换数据
data_standardized = scaler.fit_transform(data_2d)


#打印聚类后的结果
# 绘制 K-Means 聚类结果
plt.figure()
# 获取不同簇的唯一标签
unique_labels = set(labels)
# 使用 'tab10' 颜色映射
cmap = plt.get_cmap('tab10')
# 绘制每个簇的数据点
for label in unique_labels:
    cluster_data = data_2d [labels == label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cmap(label), label=f'Cluster {label}')

# # 绘制簇中心点
# plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

# 添加图例
plt.legend()
# 显示图形
plt.show()






