import numpy as np
import pandas as pd
from utils import Nantozero,replace_text_with_zero,Km_pca_show
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

# km=Km_pca_show(df,n_clusters=5,n_components=2)
# km.plot_cluster()
#
# print(km.principal_components)
# print(km.explained_variance) #[0.9977966  0.00134115]



##测试物理条件之间的相关关系

# 计算 Pearson 相关系数
cc = df.corr()

# 设置阈值，大于0.5为强正相关，小于-0.5为强负相关
cc= cc.applymap(lambda x: 1 if x > 0.5 else x)
cc = cc.applymap(lambda x: 0 if x < 0.5 and x> -0.5 else x)
cc = cc.applymap(lambda x: -1 if x < -0.5 else x)
correlation_matrix= cc .round(2)

# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()


#提取强相关关系的对象列表
#正相关关系
correlation_dic= {}
#负相关关系
correlation_re_dic={}


#提取第一行
row=correlation_matrix.iloc[0]
#读出几列
row_len=len(row)
print(row_len)
for i in range(row_len):
    #提取每一列
    col=correlation_matrix.iloc[:,i]
    if col[col==1].index.to_list()[0]:
        if col[col==1].index.to_list()[0] != i:
            correlation_dic[i]=(col[col==1].index.to_list()[0])
    # if col[col == -1].index.to_list()[0]:
    #     if col[col == -1].index.to_list()[0] != i:
    #         correlation_re_dic[i]=(col[col == -1].index.to_list()[0])


print(correlation_dic)
print(correlation_re_dic)

