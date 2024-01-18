import numpy as np
import pandas as pd
from utils import Nantozero,replace_text_with_zero,Km_pca_show,corrdic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import  font_manager
import os

'''label file pressing'''

##合并文件

# # 指定目标文件夹的路径
# folder_path = r"C:\Users\KDG\PycharmProjects\clurster\row_data\2011_2022_hg_z"
# # 使用 os 模块中的 listdir 函数获取目标文件夹中的所有文件和子文件夹
# items = os.listdir(folder_path)
# # 遍历 items 列表以获取文件和子文件夹的完整路径
# df=pd.DataFrame()
# for item in items:
#     item_path = os.path.join(folder_path, item)
#     # 使用 os.path 模块的 isdir 函数检查是否为子文件夹
#     if os.path.isdir(item_path):
#         print("子文件夹:", item_path)
#     else:
#         print(f"读取文件: {item_path}")
#         pd.read_excel(item_path)
#         s= pd.read_excel(item_path)
#         print(s)
#         if 'No' in s.columns:
#             c = pd.read_excel(item_path,index_col='No',sheet_name=2)
#             df=pd.concat([df,c])
#         else:
#             df = pd.concat([df, s])
# print(df)
# df.to_excel('2011_2021_hg_z_2.xlsx', index=False)


# df=pd.DataFrame()
# f1=pd.read_excel('file1.xlsx')
# if 'No' in f1.columns:
#     c = pd.read_excel('file1.xlsx',index_col='No')
#     df = pd.concat([df, c])
# else:
#     df = pd.concat([df, f1])
#
# f2=pd.read_excel('file2.xlsx')
# if 'No' in f2.columns:
#     c = pd.read_excel('file2.xlsx',index_col='No')
#     df = pd.concat([df, c])
# else:
#     df = pd.concat([df, f2])
#
# f3=pd.read_excel('file3.xlsx')
# if 'No' in f3.columns:
#     c = pd.read_excel('file3.xlsx',index_col='No')
#     df = pd.concat([df, c])
# else:
#     df = pd.concat([df, f3])
#
# print(df)
# df.to_excel('file.xlsx', index=False)

#环境指标
f = pd.read_excel(r'C:\Users\KDG\PycharmProjects\clurster\file.xlsx')
# print(f)
# ind_f=f.shape[0]
# print(f[' 조사구간 명'])
# name='춘성교'
# if name in f[' 조사구간 명'].values:
#     d=f[f[' 조사구간 명'] == name]
#     print(d)
#     print(d['년'])
#     print(d.index.values)
#
#     for j in d.index.values[0:5]:
#         print(j)
#         print(d.loc[j,['년']].astype(int).values[0])
#         print(d.loc[j,[' 회차']].astype(int).values[0])
#


#生物指标
# 读取jp里的조사지점名称，提取对应环境文件里的index，生成df文件
# ['년', ' 회차', ' 수계 명', ' 중권역 명', ' 분류코드', ' 조사구간 명', '건강성등급(A~E)', '종']

# df=pd.DataFrame()
folder_path = r'C:\Users\KDG\PycharmProjects\clurster\row_data\2011_2022_z'
items = os.listdir(folder_path)


for item in items:
    print(item)
    count = 0
    count_not = 0
    df = pd.DataFrame()
    item_path = os.path.join(folder_path, item)
    print(f"读取文件: {item_path}")
    s = pd.read_excel(item_path)
    if 'No' in s.columns:
        c = pd.read_excel(item_path,index_col='No')
    else:
        c = pd.read_excel(item_path)
    index_c=c.shape[0]
    print(f'当前文件总数量为：{index_c}')

    # f_list=[]
    # print(c.loc[0, ['년']].astype(int).values)
    # print(c.loc[0, [' 회차']].astype(int).values)
    # print(c.loc[0, [' 조사구간 명']].values)
    for i in range(index_c):

        #判断当前记录的区域名是否存在于file文件夹中
        # if c.loc[i, [' 조사구간 명']].values[0].strip() in f[' 조사구간 명'].values:
        #     d = f[f[' 조사구간 명'] == c.loc[i, [' 조사구간 명']].values[0].strip()]
        #     d_index=d.index.values
        if c.loc[i, [' 조사구간 명']].values[0].strip() in f[' 조사구간 명'].values:
            d = f[f[' 조사구간 명'] == c.loc[i, [' 조사구간 명']].values[0].strip()]
            d_index = d.index.values

            print(c.loc[i, [' 조사구간 명']].values[0].strip())
            # print(d_index)

            for j in d_index:
                if c.loc[i,['년']].astype(int).values[0] == d.loc[j,['년']].astype(int).values[0] and c.loc[i,[' 회차']].astype(int).values[0] == d.loc[j,[' 회차']].astype(int).values[0]:
                    df = pd.concat([df, f.iloc[[j], :]],ignore_index=True)
                    # df.loc[count, ['건강성등급(A~E)', '종']] = c.loc[i, ['건강성등급(A~E)', '종']]
                    df.loc[count, [' 학명',' 국명',' 개체 수', '종']] = c.loc[i, [' 학명',' 국명',' 개체 수', '종']]
                    count = count+1
                # else:
                    # f_list.append(i)
                    # count_not = count_not+1
        else:
            count_not = count_not + 1
        print(f'共{index_c}条记录，已读完第{i}条记录')
        # print(f'共{count}条区间记录，未读{count_not}条区间记录')
    # df.to_excel('reference_data.xlsx', index=False)
    print(index_c)
    print(count+count_not)
    print(df)
    # np.save(f'{item}_no_reference_data.npy',f_list)

    df.to_excel(f'reference_data{item}.xlsx', index=False)



#合并三个文件，找出对应的生物个数







