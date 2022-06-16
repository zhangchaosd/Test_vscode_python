from matplotlib import use
import pandas as pd
import numpy as np

z=np.loadtxt('/Users/zhangchao/Downloads/multi_stock.csv',delimiter=',',usecols=(1,2,3),unpack=True,skiprows=1)

# df1=pd.read_excel('/Users/zhangchao/Downloads/StudentsDataSet.xlsx',index_col='id',usecols='A,B')
# print(df1)