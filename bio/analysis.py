import numpy as np
import pandas as pd
from utils import Nantozero,replace_text_with_zero,Km_pca_show,corrdic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import os

'''read data from excel'''
file_path=r'C:\Users\KDG\PycharmProjects\clurster\bio\bar_plot_breast_cancer.xlsx'
df=pd.read_excel(file_path)
# print(df_data.head())
# print(df_data.columns)
#37 columns
columns=['Domain', 'Phylum', 'Class', 'Order', 'Family', 'speicies',
       'A01_read', 'A03_read', 'A04_read', 'A05_read', 'A06_read', 'A07_read',
       'A09_read', 'B01_read', 'B03_read', 'B04_read', 'B06_read', 'B07_read',
       'B09_read', 'B10_read', 'A01_abundance', 'A03_abundance',
       'A04_abundance', 'A05_abundance', 'A06_abundance', 'A07_abundance',
       'A09_abundance', 'B01_abundance', 'B03_abundance', 'B04_abundance',
       'B06_abundance', 'B07_abundance', 'B09_abundance', 'B10_abundance',
       'Normal_comparison', 'Cancer_comparison', 'Total_comparison']

'''statistics analysis with Phylum'''
phylum_list=df['Phylum'].to_list()
# print(len(phylum_list))
# print(phylum_list)
phylum_dict={}
phylum_key=[]
for i in range(len(phylum_list)):
    phylum_index = []
    if phylum_list[i] not in phylum_key:
        phylum_index.append(i)
        phylum_dict[str(phylum_list[i])]=phylum_index
    else:
        phylum_dict[str(phylum_list[i])].append(i)
    phylum_key.append(phylum_list[i])
# print(phylum_dict)
# print(phylum_dict.keys())
phylums=['Actinobacteriota', 'Firmicutes', 'Bacteroidota', 'Verrucomicrobiota', 'Cyanobacteria', 'Fusobacteriota', 'Patescibacteria', 'Proteobacteria', 'Synergistota', 'Desulfobacterota', 'Euryarchaeota']


'''statistics analysis with each sample '''
# sample_id = ['A01_abundance', 'A03_abundance','A04_abundance', 'A05_abundance', 'A06_abundance', 'A07_abundance',
#        'A09_abundance', 'B01_abundance', 'B03_abundance', 'B04_abundance',
#        'B06_abundance', 'B07_abundance', 'B09_abundance', 'B10_abundance']
# #
samples_id = ['A01', 'A03','A04', 'A05', 'A06', 'A07',
       'A09', 'B01', 'B03', 'B04',
       'B06', 'B07', 'B09', 'B10']


sample_id = ['A01_read', 'A03_read','A04_read', 'A05_read', 'A06_read', 'A07_read',
       'A09_read', 'B01_read', 'B03_read', 'B04_read',
       'B06_read', 'B07_read', 'B09_read', 'B10_read']

sample_list=[]
for id in sample_id:
    sample_dict = {}
    for phylum, indexs in phylum_dict.items():
        counts=df.loc[indexs,id].sum()
        sample_dict[phylum]=counts
    sample_list.append(sample_dict)

print(sample_list)
print(len(sample_list))
#
# Checksum is 1
# for sample in sample_list:
#     sums=sum(sample.values())
#     print(sums)
#

'''drawing phylum abundance of each sample'''
import matplotlib.pyplot as plt
#
# 定义类别和变量
var_num_list=[]
categories = samples_id
for phylum in phylums:
    var_num=[]
    for sample in sample_list:
        var_num.append(sample[phylum])
    var_num_list.append(var_num)

print(var_num_list)
print(len(var_num_list))
#
#
# '''draw Stacked Chart'''
# # fig, ax = plt.subplots(figsize=(20, 10))
# # for i in range(0,len(var_num_list)):
# #     if i == 0:
# #         bar0 = ax.bar(categories, var_num_list[0], label=f'{phylums[0]}')
# #     elif i == 1:
# #         bar1 = ax.bar(categories, var_num_list[1], bottom=var_num_list[0], label=f'{phylums[1]}')
# #     elif i == 2:
# #         bar2 = ax.bar(categories, var_num_list[2], bottom=[sum(values) for values in zip(*var_num_list[0:2])], label=f'{phylums[2]}')
# #     else:
# #         bar = ax.bar(categories, var_num_list[i], bottom=[sum(values) for values in zip(*var_num_list[0:i])], label=f'{phylums[i]}')
# #
# #     # 添加图例
# #     # ax.legend()
# #     legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})
#
# # 设置标题和轴标签
# # ax.set_title('Phylum Abundance')
# # ax.set_xlabel('Samples')
# # ax.set_ylabel('Phylum Related Abundance')
#
# # 调整图表布局，防止注释遮挡图表内容
# # plt.tight_layout()
# # # 展示图表
# # plt.show()
#
# #
# # '''比较正常人和癌患者的总和'''
# # ab_id=['Normal_comparison','Cancer_comparison']
# # abid=['Normal','Cancer']
# # # normal=sum(df['Normal_comparison'].values)
# # # print(normal)
# #
# #
# # ab_list=[]
# # for id in ab_id:
# #     ab_dict = {}
# #     for phylum, indexs in phylum_dict.items():
# #         counts=df.loc[indexs,id].sum()
# #         ab_dict[phylum]=counts
# #     ab_list.append(ab_dict)
# # # print(ab_list)
# #
# # var_num_list=[]
# # categories = abid
# # for phylum in phylums:
# #     var_num=[]
# #     for sample in ab_list:
# #         var_num.append(sample[phylum])
# #     var_num_list.append(var_num)
# #
# # print(var_num_list)
# # print(len(var_num_list))
# #
#
# #
# # fig, ax = plt.subplots(figsize=(10, 5))
# # for i in range(0,len(var_num_list)):
# #     if i == 0:
# #         bar0 = ax.bar(categories, var_num_list[0],  width=0.6,label=f'{phylums[0]}')
# #     elif i == 1:
# #         bar1 = ax.bar(categories, var_num_list[1], width=0.6, bottom=var_num_list[0], label=f'{phylums[1]}')
# #     elif i == 2:
# #         bar2 = ax.bar(categories, var_num_list[2], width=0.6, bottom=[sum(values) for values in zip(*var_num_list[0:2])], label=f'{phylums[2]}')
# #     else:
# #         bar = ax.bar(categories, var_num_list[i],  width=0.6,bottom=[sum(values) for values in zip(*var_num_list[0:i])], label=f'{phylums[i]}')
# #
# #     # 添加图例
# #     # ax.legend()
# #     legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})
# #
# #
# # # 设置标题和轴标签
# # ax.set_title('Phylum Abundance')
# # ax.set_xlabel('Group Name')
# # ax.set_ylabel('Phylum Related Abundance')
# #
# # # 调整图表布局，防止注释遮挡图表内容
# # plt.tight_layout()
# # # 展示图表
# # plt.show()
#
#
#
# '''cluster with phylum'''
# ''''Firmicutes' 'Bacteroidota' were decreased; 'Verrucomicrobiota','Proteobacteria' 'Actinobacteria' were increased'''

target_phylum=['Firmicutes','Bacteroidota', 'Verrucomicrobiota','Proteobacteria','Actinobacteriota']
phylums=['Actinobacteriota', 'Firmicutes', 'Bacteroidota', 'Verrucomicrobiota', 'Cyanobacteria', 'Fusobacteriota', 'Patescibacteria', 'Proteobacteria', 'Synergistota', 'Desulfobacterota', 'Euryarchaeota']

indices = [phylums.index(phylum) for phylum in target_phylum if phylum in phylums]
# print(indices)

var_num_list_target=[var_num_list[i] for i in indices]
print(var_num_list_target)
print(len(var_num_list_target))

#Phylum Abundance rewrite
df_target=pd.DataFrame(index=target_phylum,columns=samples_id)
df_target.loc['Firmicutes',:]=var_num_list_target[0]
df_target.loc['Bacteroidota',:]=var_num_list_target[1]
df_target.loc['Verrucomicrobiota',:]=var_num_list_target[2]
df_target.loc['Proteobacteria',:]=var_num_list_target[3]
df_target.loc['Actinobacteriota',:]=var_num_list_target[4]
print(df_target)
x= df_target.transpose()


# df_target_new=df_target
# for sample in samples_id:
#     sum_of = sum(df_target_new[sample])
#     value = df_target_new[sample].values/sum_of
#     df_target_new[sample] = value
# print(df_target_new)
#
# var_num_target=[]
# for target in target_phylum:
#     values=df_target_new.loc[target,:].to_list()
#     var_num_target.append(values)
# print(var_num_target)
#
#
#
#
# # var_num_list_target=[var_num_list[i] for i in indices]
# # print(len(var_num_list_target))
# # print(var_num_list_target)
#
#
# #  # Phylum Abundance rewrite
# # df_target=pd.DataFrame(index=target_phylum,columns=abid)
# # df_target.loc['Firmicutes',:]=var_num_list_target[0]
# # df_target.loc['Bacteroidota',:]=var_num_list_target[1]
# # df_target.loc['Verrucomicrobiota',:]=var_num_list_target[2]
# # df_target.loc['Proteobacteria',:]=var_num_list_target[3]
# # df_target.loc['Actinobacteriota',:]=var_num_list_target[4]
# # print(df_target)
#
# # df_target_new=df_target
# # for abi in abid:
# #     sum_of = sum(df_target_new[abi])
# #     value = df_target_new[abi].values/sum_of
# #     df_target_new[abi] = value
# # print(df_target_new)
# #
# #
# # var_num_target=[]
# # for target in target_phylum:
# #     values=df_target_new.loc[target,:].to_list()
# #     var_num_target.append(values)
# # print(var_num_target)
# #
#
# #
# # fig, ax = plt.subplots(figsize=(20, 10))
# # for i in range(0,len(var_num_target)):
# #     if i == 0:
# #         bar0 = ax.bar(categories, var_num_target[0], width=0.6, label=f'{target_phylum[0]}')
# #     elif i == 1:
# #         bar1 = ax.bar(categories, var_num_target[1],width=0.6, bottom=var_num_target[0], label=f'{target_phylum[1]}')
# #     elif i == 2:
# #         bar2 = ax.bar(categories, var_num_target[2],width=0.6, bottom=[sum(values) for values in zip(*var_num_target[0:2])], label=f'{target_phylum[2]}')
# #     else:
# #         bar = ax.bar(categories, var_num_target[i],width=0.6, bottom=[sum(values) for values in zip(*var_num_target[0:i])], label=f'{target_phylum[i]}')
# #
# #     # 添加图例
# #     # ax.legend()
# #     legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})
# #
# # # 设置标题和轴标签
# # ax.set_title('5 Target Phylum Abundance')
# # ax.set_xlabel('Group name')
# # ax.set_ylabel('Phylum Related Abundance')
# #
# # # 调整图表布局，防止注释遮挡图表内容
# # plt.tight_layout()
# # # 展示图表
# # plt.show()
#
#
# '''classification for normal or cancer '''
# # 逻辑回归（Logistic Regression）:
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# print(df_target_new)
# df_target= df_target_new.transpose()
# print(df_target)
# #0 is normal; 1 is cancer
# label=[0,0,0,0,0,0,0,1,1,1,1,1,1,1]
#
# x =df_target
# y= pd.DataFrame({'label': label})
# print(y)
# #
# # 划分数据集为训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#
# # 创建逻辑回归模型
# model = LogisticRegression()
#
# # 训练模型
# model.fit(x_train, y_train)
#
# # 预测测试集
# y_pred = model.predict(x_test)
#
# # 评估模型性能
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)
#
# print(f'Accuracy: {accuracy}')
# print(f'Confusion Matrix:\n{conf_matrix}')
# print(f'Classification Report:\n{classification_rep}')
#

# 支持向量机（Support Vector Machines, SVM）:
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# #
# # 将数据集拆分为训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
# #
# # 创建SVC模型
# svm_model = SVC(kernel='linear', C=1.0)
#
# # 在训练集上拟合模型
# svm_model.fit(x_train, y_train)
#
# # 在测试集上进行预测
# y_pred = svm_model.predict(x_test)
#
# # 计算分类准确度
# accuracy = accuracy_score(y_test, y_pred)
# print(f"分类准确度：{accuracy}")


#计算相关系数
# df_reset = x.reset_index(drop=True)
# df_reset=df_reset.astype('float32')
# cc = df_reset.corr()
# print(cc)
# sns.heatmap(cc, annot=True, cmap='coolwarm')
# plt.show()



# #
# 决策树（Decision Trees）:
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier,plot_tree
# x=x.loc[:,('Actinobacteriota','Bacteroidota','Firmicutes')]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=20) #42
#
# # 创建决策树分类器
# decision_tree_model = DecisionTreeClassifier()
#
# # 拟合模型
# decision_tree_model.fit(x_train, y_train)
#
# # 在测试集上进行预测
# y_pred = decision_tree_model.predict(x_test)
#
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
#
#
# plt.figure(figsize=(20, 10))
# plot_tree(decision_tree_model, filled=True, feature_names=x.columns, class_names=['Normal','Cancer'], rounded=True)
# plt.show()
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)


# Accuracy: 0.75
# Confusion Matrix:
# [[2 2]
#  [0 3]]

# [[1 0]
#  [1 2]]
# True Negative (TN)：1，表示实际为负类别且被正确分类为负类别的样本数。2
# False Positive (FP)：0，表示实际为负类别但被错误分类为正类别的样本数。2
# False Negative (FN)：1，表示实际为正类别但被错误分类为负类别的样本数。0
# True Positive (TP)：2，表示实际为正类别且被正确分类为正类别的样本数。3
# 可以计算一些评估指标：
#
# 准确率 (Accuracy)：(TN + TP) / (TN + FP + FN + TP) = (1 + 2) / (1 + 0 + 1 + 2) = 0.75    0.7143
# 精确度 (Precision)：TP / (FP + TP) = 2 / (0 + 2) = 1.0   1.0
# 召回率 (Recall)：TP / (FN + TP) = 2 / (1 + 2) = 0.6667   0.6667
# F1 分数 (F1 Score)：2 * (Precision * Recall) / (Precision + Recall) = 2 * (1.0 * 0.6667) / (1.0 + 0.6667) ≈ 0.8
#准确率衡量整体正确预测的比例，而精确度和召回率则关注正例的预测情况。精确度表示模型在预测为正例的样本中有多少是真正的正例，即模型的预测有 100% 的准确性。
# 召回率表示模型成功找到了多少正例，即模型在所有实际正例中找到了0.6667。
# F1 分数是精确度和召回率的调和平均值，它提供了对精确度和召回率的综合评估。在某些情况下，F1 分数可能更具启发性，因为它考虑了两者之间的平衡。
# F1 分数为 80% 是一个相对较好的性能。F1 分数是精确度和召回率的综合度量，它考虑了两者之间的平衡。在二分类问题中，F1 分数越接近 1，表示模型在精确度和召回率之间取得了很好的平衡。

# 在决策树的图示中，属性：
# Gini系数的取值范围在0到1之间，值越小表示节点的纯度越高，即节点中的样本越属于同一类别。当Gini系数为0时，节点是纯净的，即所有样本属于同一类别；当Gini系数为1时，节点是不纯的，即样本均匀地分布在各个类别中。
# value 属性是一个数组，表示该节点中每个类别的样本数量。例如，对于二分类问题，value 属性可能是 [10, 5]，表示该节点中有10个属于类别1的样本，5个属于类别2的样本。
# samples 属性表示该节点的样本数量。继续上面的例子，如果节点中总共有15个样本，samples 就是15。




# #交叉验证法
# from sklearn.tree import DecisionTreeClassifier,plot_tree
# from sklearn.model_selection import cross_val_score
#
# # 创建一个决策树分类器
# clf = DecisionTreeClassifier(random_state=20)
#
# # 使用K折交叉验证评估模型性能
# # 这里使用了5折交叉验证，可以根据需要调整
# scores = cross_val_score(clf, x.loc[:,('Bacteroidota','Verrucomicrobiota', 'Actinobacteriota')], y, cv=4, scoring='accuracy')
#
# # 打印每折的准确度
# print("Cross-validated Accuracy Scores:", scores)
#
# # 打印平均准确度
# print("Average Accuracy:", scores.mean())
#
#
#
# '''计算阿尔法多样性和贝塔 多样性'''
# import numpy as np
# from scipy.stats import entropy
# from sklearn.metrics import pairwise_distances

# print(x)
# otu_table=np.array(x.values)
# print(otu_table.shape)
#
#
# 计算 Shannon 多样性指数
# def shannon_diversity(counts):
#     # 将零值替换为一个小的非零值
#     counts_nonzero = np.where(counts == 0, 1e-10, counts)
#     counts_nonzero=counts_nonzero .astype(float)
#
#     # 将数据类型转换为浮点数
#     proportions = counts_nonzero / np.sum(counts_nonzero, axis=1, keepdims=True)
#
#     return -np.sum(proportions * np.log(proportions), axis=1)
#
# shannon_alpha_diversity = shannon_diversity(otu_table)
#
# print("Shannon Alpha Diversity:")
# print(shannon_alpha_diversity)

# # Shannon Alpha Diversity:
# # [1.05021956 0.9539036  0.69887474 0.89656228 1.13352764 1.01838181
# #  1.21929454 0.75884766 1.01876107 0.91298512 1.09221488 0.83709092
# #  1.03041695 0.98822073]
#
# sample_labels = samples_id
# alpha_diversity_values = shannon_alpha_diversity

# # # 绘制阿尔法多样性的柱状图
# plt.figure(figsize=(8, 6))
# sns.barplot(x=sample_labels, y=alpha_diversity_values, palette="viridis")
# plt.title("Alpha Diversity")
# plt.xlabel("Samples")
# plt.ylabel("Alpha Diversity Value")
# plt.show()
#
# s=alpha_diversity_values.reshape((2,7)).tolist()
# print(s)
# #
# # 创建箱线图
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=s)
# plt.xticks(ticks=[0, 1], labels=['Normal', 'Cancer'])
# plt.title("Shannon Alpha Diversity Boxplot")
# plt.xlabel("Group Name")
# plt.ylabel("Shannon Alpha Diversity")
# plt.legend()
# plt.show()
#
# #
# # 计算 Bray-Curtis 相似性指数
# bray_curtis_beta_diversity = pairwise_distances(otu_table, metric="braycurtis")
#
# print("Bray-Curtis Beta Diversity:")
# print(bray_curtis_beta_diversity)
# beta_diversity_matrix=bray_curtis_beta_diversity
#
# # 绘制贝塔多样性的热图
# plt.figure(figsize=(8, 6))
# sns.heatmap(beta_diversity_matrix, annot=True, cmap="viridis", xticklabels=sample_labels, yticklabels=sample_labels)
# plt.title("Beta Diversity Heatmap")
# plt.xlabel("Samples")
# plt.ylabel("Samples")
# plt.show()

import numpy as np



# 创建一个示例的 OTU 表 (Operational Taxonomic Units)
file_path=r'C:\Users\KDG\PycharmProjects\clurster\bio\bar_plot_breast_cancer.xlsx'
df=pd.read_excel(file_path)
columns=['speicies',
       'A01_read', 'A03_read', 'A04_read', 'A05_read', 'A06_read', 'A07_read',
       'A09_read', 'B01_read', 'B03_read', 'B04_read', 'B06_read', 'B07_read',
       'B09_read', 'B10_read']

# 创建样本标签
sample_ids = ['A01_read', 'A03_read', 'A04_read', 'A05_read', 'A06_read', 'A07_read',
       'A09_read', 'B01_read', 'B03_read', 'B04_read', 'B06_read', 'B07_read',
       'B09_read', 'B10_read']

# 创建物种标签
otu_ids = df[sample_ids].transpose()
print(otu_ids)

from skbio import TreeNode
# otu_table = TreeNode.from_taxonomy(otu_table, otu_ids, sample_ids)

from skbio.diversity import alpha_diversity
from skbio.diversity import beta_diversity

# 计算阿尔法多样性
alpha_div = alpha_diversity(metric='shannon', counts=otu_ids.values)

print("Alpha Diversity:")
print(alpha_div)
s=alpha_div.tolist()
ss=np.array(s)
s=ss.reshape((2,7)).tolist()
print(s)
plt.figure(figsize=(8, 6))
sns.boxplot(data=s)
plt.xticks(ticks=[0, 1], labels=['Normal', 'Cancer'])
plt.title("Shannon Species Alpha Diversity Boxplot")
plt.xlabel("Group Name")
plt.ylabel("Shannon Alpha Diversity")
plt.legend()
plt.show()


# # ## Bacteroidota Verrucomicrobiota Actinobacteriota;1,2,4
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier,plot_tree
#
# x_train, x_test, y_train, y_test = train_test_split(x.loc[:,('Bacteroidota','Verrucomicrobiota', 'Actinobacteriota')], y, test_size=0.6, random_state=42)
#
# # 创建决策树分类器
# decision_tree_model = DecisionTreeClassifier()
#
# # 拟合模型
# decision_tree_model.fit(x_train, y_train)
#
# # 在测试集上进行预测
# y_pred = decision_tree_model.predict(x_test)
#
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
#
#
#
# plt.figure(figsize=(20, 10))
# plot_tree(decision_tree_model, filled=True, feature_names=x.columns, class_names=True, rounded=True)
# plt.show()
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(cm)
#



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#
# #随机森林
# # 将数据集分割为训练集和测试集
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=40)
#
# # 创建随机森林分类器
# random_forest_model = RandomForestClassifier(n_estimators=100, random_state=29)
#
# # 拟合模型
# random_forest_model.fit(x_train, y_train)
#
# # 在测试集上进行预测
# y_pred = random_forest_model.predict(x_test)
#
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
#
# # 打印混淆矩阵、分类报告等其他性能指标
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# print(r"\nClassification Report:")
# print(classification_report(y_test, y_pred))



# kmeans
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
#
# # 创建 KMeans 模型
# kmeans_model = KMeans(n_clusters=2, random_state=42)
#
# score_list=[]
# 拟合模型
# for i in range(5):
#     a=target_phylum[i]
#     for j in range(5):
#         if i!= j:
#             b=target_phylum[j]
#             for s in range(5):
#                 if j != s:
#                     c=target_phylum[s]
#                     for f in range(5):
#                         if s != f:
#                             d = target_phylum[f]
#                             kmeans_model.fit(x.loc[:,(a,b,c,d)])
#
#                             # 获取每个样本的簇标签
#                             labels = kmeans_model.labels_
#
#                             # 获取簇的中心点
#                             centroids = kmeans_model.cluster_centers_
#                             true_label=[0,0,0,0,0,0,0,1,1,1,1,1,1,1]
#
#                             # 计算 ARI
#                             ari_score = adjusted_rand_score(true_label, labels)
#
#                             # 计算 NMI
#                             nmi_score = normalized_mutual_info_score(true_label, labels)
#
#                             print(f"Adjusted Rand Index (ARI): {ari_score}")
#                             print(f"Normalized Mutual Information (NMI): {nmi_score}")
#                             score_list.append([ari_score,nmi_score])

# 计算分数
# kmeans_model.fit(x)
#
# # 获取每个样本的簇标签
# labels = kmeans_model.labels_
#
# # 获取簇的中心点
# centroids = kmeans_model.cluster_centers_
# true_label=[0,0,0,0,0,0,0,1,1,1,1,1,1,1]
#
# # 计算 ARI
# ari_score = adjusted_rand_score(true_label, labels)
#
# # 计算 NMI
# nmi_score = normalized_mutual_info_score(true_label, labels)
#
# print(f"Adjusted Rand Index (ARI): {ari_score}")
# print(f"Normalized Mutual Information (NMI): {nmi_score}")
# score_list.append([ari_score,nmi_score])

# 可视化结果
# plt.scatter(x.loc[:,a], x.loc[:,b], c=labels, cmap='viridis', edgecolor='k', s=100,label='Cluster Label')
# scatter=plt.scatter(x.loc[:, a], x.loc[:,b], c=true_label, cmap='viridis',edgecolor='k', s=100, marker='x',label='True Label')

# plt.title('KMeans Clustering')
# plt.xlabel(a)
# plt.ylabel(b)
# plt.legend()
# plt.show()
#
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
#
# # 计算 ARI
# ari_score = adjusted_rand_score(true_label, labels)
#
# # 计算 NMI
# nmi_score = normalized_mutual_info_score(true_label, labels)
#
# print(f"Adjusted Rand Index (ARI): {ari_score}")
# print(f"Normalized Mutual Information (NMI): {nmi_score}")

# 朴素贝叶斯（Naive Bayes）:
