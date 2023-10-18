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
#             c = pd.read_excel(item_path,index_col='No')
#             df=pd.concat([df,c])
#         else:
#             df = pd.concat([df, s])
# print(df)
# df.to_excel('2011_2022_hg_z.xlsx', index=False)

