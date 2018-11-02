import numpy as np
import os

def filter(Scode):
    dir = os.getcwd()
    f = open(dir + '/' + 'Stock_Data/' + Scode +'.txt','r')
    data = f.readlines()

    for i in range(len(data)):
        data_row = [str(x) for x in data[i].split(' ')]
        # data_row.pop(7)
        data_row.pop(0)
        data[i] = data_row

    data = np.array(data)
    length = len(data)
    width = len(data[0])
    open_value = 0
    close_value = 3
    flag = True

    data_year = np.zeros([365,len(data[0])])

    for i in range(365):
        for j in range(len(data[0])-1):
            data_year[i][j] = data[len(data)-i-1][j]

    diff_pos = 0
    diff_neg = 0
    for i,j in enumerate(data_year):
        a = float(j[open_value])
        b = float(j[close_value])
        if a > b:
            diff_neg += a - b
        elif a < b:
            diff_pos += b - a
        else:
            continue

    vitality = (diff_pos + diff_neg)/float(data[length-1][close_value])

    if (diff_pos/diff_neg)>1.05 and vitality>10:
        flag = False

    if flag == False:
        os.remove(os.getcwd()+'/Stock_Data/'+Scode + '.txt')

def main(dir):
    files = os.listdir(dir)
    for i in range(len(files)):
        files[i] = files[i][:-4]
    for file in files:
        filter(file)


if __name__ == '__main__':
    main()