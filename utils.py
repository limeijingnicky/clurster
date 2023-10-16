import pandas as pd
import numpy as np

def Nantozero(filename):

    #读取文件
    file_frame_index=filename.find('.')+1
    file_frame=filename[file_frame_index:]

    if file_frame == 'xlsx' or file_frame == 'excel':
        f = pd.read_excel(filename)

    #将nan变为0
    ff = f.fillna(0, inplace=False)
    return ff




# 定义一个函数，用于将文字替换为数字0
def replace_text_with_zero(value):
    if isinstance(value, str):  # 如果是字符串
        return 0
    return value  # 如果不是字符串，保持不变


