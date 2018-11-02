import tensorflow as tf
import numpy as np
import os
import pandas as pd

def compare_core(Scode):
    flag = False
    dir = os.getcwd() + '/' + 'Stock_Data' + '/' + Scode + '.txt'
    data = pd.read_csv(dir, sep=" ", header=None)
    data.columns = ["Symbol", "Open", "High", "Low", 'Close', 'Volume', 'Date']
    data.drop(['Symbol', 'Date'], axis=1, inplace=True)
    data_processed = data.as_matrix()
    diff = (data_processed[len(data_processed)-1][3] - data_processed[len(data_processed)-2][3])/data_processed[len(data_processed)-2][3]
    if diff > 0.02:
        flag = True
    else:
        pass
    return flag,diff

def main(Scode):
    flag,diff = compare_core(Scode)
    return flag,diff

if __name__ == '__main__':
    Scode = 'AA'
    main(Scode)