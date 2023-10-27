import numpy as np
import pandas as pd
from utils import Nantozero,replace_text_with_zero,Km_pca_show,replace_a_to_num
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager



'''data processing '''
#
# # #将数据的空缺值，用0处理
# file=Nantozero('C:/Users/KDG/PycharmProjects/clurster/reference_data.xlsx')
# # print(file.iloc[:, 7:])
#
# # # 去除dataframe里的字符串，使用 applymap 方法应用函数到整个 DataFrame
# df = file.iloc[:, 7:-2].applymap(replace_text_with_zero)
#
# df = pd.concat([df, file.iloc[:, -2::]],axis=1)
#
#将物理信息和生物信息标签合并
# df.to_excel('ref_data.xlsx', index=False)


# # column_name=[' 수심', '유량', '생물화학적산소요구량(BOD)', '클로로필-a(Chlorophyll-a)',
# #        '화학적산소요구량(COD)', '용존산소(DO)', '용존총질소(DTN)', '용존총인(DTP)', '전기전도도(EC)',
# #        '분원성대장균군', '암모니아성 질소(NH3-N)', '질산성질소(NO3-N)', '수소이온농도(pH)',
# #        '인산염 인(PO4-P)', '부유물질(SS)', '총대장균군', '수온', '총질소(T-N)', '총유기탄소(TOC)',
# #        '총인(T-P)','건강성등급(A~E)', '종']




'''data analysis'''


'''data cluster'''
# df=pd.read_excel('ref_data.xlsx')
# df=replace_a_to_num(df,"A",1)
# df=replace_a_to_num(df,"B",2)
# df=replace_a_to_num(df,"C",3)
# df=replace_a_to_num(df,"D",4)
# df=replace_a_to_num(df,"E",5)
# df.to_excel('re_data.xlsx', index=False)
df=pd.read_excel('re_data.xlsx')

#不同生物物种
df_0=df[df['종']==0]
targets_0=df_0.iloc[:,-2].tolist()
print(df_0)

df_1=df[df['종']==1]
targets_1=df_1.iloc[:,-2].tolist()
print(df_1)

df_2=df[df['종']==1]
targets_2=df_2.iloc[:,-2].tolist()
print(df_2)


#
# ##绘制环境聚类#类标签画图
# km=Km_pca_show(df_0.iloc[:,:-2],n_clusters=5,n_components=2,targets=targets_0)
# km.plot_cluster()
#

from matplotlib import font_manager
korean_font=font_manager.FontEntry(fname=(r'C:\Windows\Fonts\HMFMOLD.ttf'),name='HMFMOLD')
font_manager.fontManager.ttflist.insert(0,korean_font)
plt.rcParams.update({'font.size':18,'font.family':'HMFMOLD'})


# '총대장균군' '분원성대장균군'
select_columns=['총대장균군','분원성대장균군']
df=df_0
kmeans = KMeans(n_clusters=5)
kmeans.fit(df[select_columns])
labels = kmeans.labels_
df['Cluster'] = labels
plt.scatter(df[select_columns[0]], df[select_columns[1]], c=df['Cluster'], cmap='viridis')
plt.xlabel('분원성대장균군')
plt.ylabel('총대장균군')
plt.legend()
plt.show()


plt.scatter(df[select_columns[0]], df[select_columns[1]], c=df['건강성등급(A~E)'], cmap='viridis')
plt.xlabel('분원성대장균군')
plt.ylabel('총대장균군')
plt.legend()
plt.show()