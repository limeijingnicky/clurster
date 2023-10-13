import numpy as np
import pandas as pd


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
# # '수심'
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