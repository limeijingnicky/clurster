import numpy as np
import pandas as pd
from utils import Nantozero,replace_text_with_zero,Km_pca_show,corrdic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



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


# #将处理后文件进行储存
# df1.to_excel('file1.xlsx', index=False)
# df2.to_excel('file2.xlsx', index=False)
# df3.to_excel('file3.xlsx', index=False)


# # #将数据的空缺值，用0处理
# file1=Nantozero('file1.xlsx')
# file2=Nantozero('file2.xlsx')
# file3=Nantozero('file3.xlsx')
# # # print(file.iloc[:, 7:])
# #
# # # 去除dataframe里的字符串，使用 applymap 方法应用函数到整个 DataFrame
# df1 = file1.iloc[:, 7:].applymap(replace_text_with_zero)
# df2 = file2.iloc[:, 7:].applymap(replace_text_with_zero)
# df3 = file3.iloc[:, 7:].applymap(replace_text_with_zero)
#
# # print(df.columns)
# # column_name=[' 수심', '유량', '생물화학적산소요구량(BOD)', '클로로필-a(Chlorophyll-a)',
# #        '화학적산소요구량(COD)', '용존산소(DO)', '용존총질소(DTN)', '용존총인(DTP)', '전기전도도(EC)',
# #        '분원성대장균군', '암모니아성 질소(NH3-N)', '질산성질소(NO3-N)', '수소이온농도(pH)',
# #        '인산염 인(PO4-P)', '부유물질(SS)', '총대장균군', '수온', '총질소(T-N)', '총유기탄소(TOC)',
# #        '총인(T-P)']
#
# #合并多个文件
# df = pd.concat([df1, df2, df3])
# # df.to_excel('Prefile.xlsx', index=False)
#


df=pd.read_excel('Prefile.xlsx')
df.columns = [i for i in range(20)]
#
# km=Km_pca_show(df,n_clusters=5,n_components=2)
# km.plot_cluster()
#
# print(km.principal_components)
# [[ 7.28374505e-08  6.70031116e-06  3.72752429e-06  2.64607774e-06
#    3.27158603e-06 -1.17347401e-06  3.59868746e-06  3.74197868e-09
#    8.98924003e-05  2.64737351e-03  1.61649603e-08  1.05563127e-07
#    1.87708807e-08  3.18727470e-09  4.05486687e-06  9.99996492e-01
#    2.81877631e-06  3.61572372e-06  1.45412050e-07  4.60819193e-09]

# [-5.71437156e-07  2.99367240e-05 -2.45044900e-06  6.46251234e-05
#    5.41091331e-07 -8.88907830e-06  1.17580145e-05  7.16524264e-07
#   -8.60831319e-04  9.99996113e-01  4.15310186e-06  1.69170341e-05
#    2.30272653e-07  6.41072713e-07  1.32912380e-04 -2.64729620e-03
#    2.58249879e-05  1.04799663e-05  2.23445686e-05  9.83877753e-07]]

# print(km.explained_variance)
# #[0.9977966  0.00134115]


#
##测试物理条件之间的相关关系

# 计算 Pearson 相关系数
cc = df.corr()

# 设置阈值，大于0.5为强正相关，小于-0.5为强负相关
cc= cc.applymap(lambda x: 1 if x > 0.5 else x)
cc = cc.applymap(lambda x: 0 if x < 0.5 and x> -0.5 else x)
cc = cc.applymap(lambda x: -1 if x < -0.5 else x)
correlation_matrix = cc .round(2)

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()

corr,corr_re=corrdic(correlation_matrix,20)
print(corr) #[[2, 4], [5, 12], [6, 17], [7, 11], [7, 13], [7, 18], [7, 19], [11, 13], [11, 18], [13, 19], [18, 19]] ,
# 也可以看成是[[2, 4], [5, 12], [6, 17], [7, 11 ，13 ，18 ，19]]

##根据相关性，选择不同数量的条件作为它的分类条件
##不选择[4,11,12,13,17,18,19]
df.columns = [i for i in range(20)]

km=Km_pca_show(df,n_clusters=5,n_components=2)
km.plot_cluster()
print(km.principal_components)
