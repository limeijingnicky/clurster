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
# filepath1="C:/Users/KDG/PycharmProjects/clurster/ref_data_0.xlsx"
# filepath2="C:/Users/KDG/PycharmProjects/clurster/ref_data_1.xlsx"
# filepath3="C:/Users/KDG/PycharmProjects/clurster/ref_data_2.xlsx"
#
# f1 = pd.read_excel(filepath1)
# f2 = pd.read_excel(filepath2)
# f3 = pd.read_excel(filepath3)
#
# print(f1)
# print(f2)
# print(f3)
# df = pd.concat([f1, f2],axis=0)
# print(df)
# df1 = pd.concat([df, f3],axis=0)
# print(df1)
# df1.to_excel('ref_data_all.xlsx', index=False)

# file=Nantozero(filepath)
# print(file.iloc[:, 7:])
#
# # 去除dataframe里的字符串，使用 applymap 方法应用函数到整个 DataFrame
# df1 = file.iloc[:, 7:-2].applymap(replace_text_with_zero)
# df2 = file.iloc[:, -2::]
# print(df1)
# print(df2)
# df = pd.concat([df1, df2],axis=1)
# print(df)
# # #
# # ##将物理信息和生物信息标签合并
# df.to_excel('ref_data_3.xlsx', index=False)


# # column_name=[' 수심', '유량', '생물화학적산소요구량(BOD)', '클로로필-a(Chlorophyll-a)',
# #        '화학적산소요구량(COD)', '용존산소(DO)', '용존총질소(DTN)', '용존총인(DTP)', '전기전도도(EC)',
# #        '분원성대장균군', '암모니아성 질소(NH3-N)', '질산성질소(NO3-N)', '수소이온농도(pH)',
# #        '인산염 인(PO4-P)', '부유물질(SS)', '총대장균군', '수온', '총질소(T-N)', '총유기탄소(TOC)',
# #        '총인(T-P)','건강성등급(A~E)', '종']




# '''data analysis'''
#
#
# '''data cluster'''
# df=pd.read_excel('ref_data_all.xlsx')
# df=replace_a_to_num(df,"A",1)
# df=replace_a_to_num(df,"B",2)
# df=replace_a_to_num(df,"C",3)
# df=replace_a_to_num(df,"D",4)
# df=replace_a_to_num(df,"E",5)
# df.to_excel('re_data_all.xlsx', index=False)
df=pd.read_excel('re_data_all.xlsx')
print(df)

# # #
#不同生物物种
df_0=df[df['종']==0]
targets_0=df_0.iloc[:,-2].tolist()
print(df_0)
print(len(targets_0))
print(targets_0[10])

#
# df_1=df[df['종']==1]
# targets_1=df_1.iloc[:,-2].tolist()
# print(df_1)
#
# df_2=df[df['종']==2]
# targets_2=df_2.iloc[:,-2].tolist()
# print(df_2)


#
##绘制环境聚类#类标签画图
# km=Km_pca_show(df_2.iloc[:,:-2],n_clusters=5,n_components=2,targets=targets_2)
# km.plot_cluster()
# #
#
# from matplotlib import font_manager
# korean_font=font_manager.FontEntry(fname=(r'C:\Windows\Fonts\HMFMOLD.ttf'),name='HMFMOLD')
# font_manager.fontManager.ttflist.insert(0,korean_font)
# plt.rcParams.update({'font.size':18,'font.family':'HMFMOLD'})
#
#
# # '총대장균군' '분원성대장균군'
# select_columns=['총대장균군','분원성대장균군']
# df=df_0
# kmeans = KMeans(n_clusters=5)
# kmeans.fit(df[select_columns])
# labels = kmeans.labels_
# df['Cluster'] = labels
# plt.scatter(df[select_columns[0]], df[select_columns[1]], c=targets_0, cmap='viridis')
# plt.xlabel('분원성대장균군')
# plt.ylabel('총대장균군')
# plt.legend()
# plt.show()

#
# plt.scatter(df[select_columns[0]], df[select_columns[1]], c=df['건강성등급(A~E)'], cmap='viridis')
# plt.xlabel('분원성대장균군')
# plt.ylabel('총대장균군')
# plt.legend()
# plt.show()


import keras
from keras.layers import Dense, Dropout, Flatten,LSTM,Conv1D,Input,MaxPooling1D,Normalization
from keras.activations import softmax
from keras.models import Model
def model():
    ip= Input(shape=(1,20))
    c1= Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(ip)
    n1= Normalization()(c1)

    c2 = Conv1D(filters=16, kernel_size=3, padding='same', activation='relu')(n1)
    n2 = Normalization()(c2)

    ds=Dense(units=5)(n2)
    op=softmax(ds)

    m=Model(inputs=ip,outputs=op)
    return m

cl=model()
cl.summary()

import numpy as np

x_df = np.array(df_0.iloc[:,:-2].values)
print(x_df)
print(x_df.shape)

x=x_df.reshape(x_df.shape[0],1,x_df.shape[1])
print(x.shape)

y_array= np.zeros((len(targets_0),5))
for i in range(len(targets_0)):
    if targets_0[i] == 1:
        y_array[i]=[1,0,0,0,0]
    if targets_0[i] == 2:
        y_array[i] = [0, 1, 0, 0, 0]
    if targets_0[i] == 3:
        y_array[i] = [0, 0, 1, 0, 0]
    if targets_0[i] == 4:
        y_array[i] = [0, 0, 0, 1, 0]
    if targets_0[i] == 5:
        y_array[i] = [0, 0, 0, 0, 1]
print(y_array)
y=y_array.reshape(y_array.shape[0],1,y_array.shape[1])
cl.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["accuracy"],
        )

result=cl.fit(x,y,batch_size=10,epochs=100)
print(result)



