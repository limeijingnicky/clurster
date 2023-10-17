import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def Nantozero(filename):

    #读取文件
    file_frame_index=filename.find('.')+1
    file_frame=filename[file_frame_index:]

    if file_frame == 'xlsx' or file_frame == 'excel':
        f = pd.read_excel(filename)

    #将nan变为0
    ff = f.fillna(0, inplace=False)
    return ff

def inftozero(df):
    # 将无穷值(空白值)替换为 0
    dff = df.replace([np.inf, -np.inf], np.nan)
    dfff = dff.fillna(0)
    return dfff


# 定义一个函数，用于将文字替换为数字0
def replace_text_with_zero(value):
    if isinstance(value, str):  # 如果是字符串
        return 0
    return value  # 如果不是字符串，保持不变




class Km_pca_show:
    def __init__(self,df,n_clusters,n_components=2,if_pca=True,if_scale=True):
        self.df=df
        self.if_pca=if_pca
        self.if_scale=if_scale
        self.n_clusters=n_clusters
        self.n_components=n_components
        self.labels = None
        self.centroids = None
        self.data_2d = None
        self.data_standardized = None
        self.principal_components=None
        self.explained_variance=None

        #使用kmeans方法，因为生物标准有5个等级，所以类别为5
        kmeans = KMeans(self.n_clusters)
        # 使用数据拟合 KMeans 模型
        kmeans.fit(self.df)
        # 获取每个样本所属的簇
        self.labels = kmeans.labels_
        # 获取簇的中心点
        self.centroids = kmeans.cluster_centers_

        if self.if_pca is True:
            # 使用 PCA 进行降维,方便画图
            pca = PCA(self.n_components)
            self.data_2d = pca.fit_transform(self.df)
            # self.centroids_2d = pca.transform(self.centroids)
            self.principal_components = pca.components_
            self.explained_variance = pca.explained_variance_ratio_

        if self.if_scale is True:
            # 初始化标准化器
            scaler = StandardScaler()
            # 使用标准化器拟合并转换数据
            self.data_2d = scaler.fit_transform(self.data_2d)


    def plot_cluster(self):
        #打印聚类后的结果
        # 绘制 K-Means 聚类结果
        plt.figure()
        # 获取不同簇的唯一标签
        unique_labels = set(self.labels)
        # 使用 'tab10' 颜色映射
        cmap = plt.get_cmap('tab10')
        # 绘制每个簇的数据点
        for label in unique_labels:
            cluster_data = self.data_2d [self.labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cmap(label), label=f'Cluster {label}')

        # # 绘制簇中心点
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')

        # 添加图例
        plt.legend()
        # 显示图形
        plt.show()


def corrdic(correlation_matrix,category,corr_value=1,corr_re_value=-1,):

    # 正相关关系
    correlation_dic= {}
    #负相关关系
    correlation_re_dic={}

    #提取第一行
    row=correlation_matrix.iloc[0]
    #读出几列
    row_len=len(row)

    for i in range(row_len):
        #提取每一列
        col=correlation_matrix.iloc[:,i]
        li=[]
        for x in col[col == corr_value].index.to_list():
            if x != i:
                li.append(x)
                correlation_dic[i] = li

        for x in col[col == corr_re_value].index.to_list():
            if x != i:
                li.append(x)
                correlation_re_dic[i] = li


    # 去掉字典里相互重复的对象
    correlation_list = []
    for i in range(category):
        if i in correlation_dic.keys():
            value = correlation_dic[i]

            for j in range(len(value)):
                correlation_l = []
                pp = [i, value[j]]
                correlation_l.append(pp)
                correlation_list.append(correlation_l)

    #将数值按照小大排序，从而删除相同的元素/使list里的数值必须大于key {2：【1,4】}
    corr = []
    for i in range(len(correlation_list)):
        value = correlation_list[i][0]
        value = sorted(value)
        if value not in corr:
            corr.append(value)

    correlation_re_list = []
    for i in range(20):
        if i in correlation_re_dic.keys():
            value = correlation_re_dic[i]

            for j in range(len(value)):
                correlation_l = []
                pp = [i, value[j]]
                correlation_l.append(pp)
                correlation_re_list.append(correlation_l)

    corr_re = []
    for i in range(len(correlation_re_list)):
        value = correlation_list[i][0]
        value = sorted(value)
        if value not in corr_re:
            corr_re.append(value)

    return corr,corr_re
