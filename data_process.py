import os
import pandas as pd
import read_data_from_file
import shutil
import Sort_stock
import tfsvmexp
import pureNN

group_num = 5


def readtxtfiletopandas(dir):

    data = pd.read_csv(dir, sep=" ", header=None)
    data.columns = ["Symbol", "Open", "High", "Low",'Close','Volume','Date']
    return data


def main(dir,margin,period,upvalue,val):
    try:
        shutil.rmtree('Stock_Data_IDV')
    except FileNotFoundError:
        pass
    finally:
        os.mkdir('Stock_Data_IDV')
    files = os.listdir(dir)
    counter = 1


    # ff = open('50.txt','r')
    # vlist = ff.readlines()
    # for i in range(len(vlist)):
    #     vlist[i] = vlist[i].replace('\n','')


    for file in files:
        data = readtxtfiletopandas(dir + '/' + file)
        Scode = file[:-4]
        # if Scode not in vlist:
        #     os.remove('Stock_Data/'+Scode+'.txt')
        read_data_from_file.main(Scode, data,margin,period,upvalue,val)
        print('------'+str(counter)+'/'+str(len(files))+'-----------')
        counter += 1
    currentdir = os.getcwd() + '/' +'Stock_Data_IDV'
    files_1 = os.listdir(currentdir)
    counter = 1
    num_file = len(files)
    num_file = num_file % group_num + int(num_file/group_num)
    Scode_list = []
    for i in range(num_file):
        temp = []
        try:
            for j in range(group_num):
                temp.append(files[i*group_num+j][:-4])
        except IndexError:
            pass
        Scode_list.append(temp)
    print('afasf')
    ccc = 0
    if len(Scode_list[len(Scode_list)-1]) == 0:
        Scode_list.pop(len(Scode_list)-1)
    for i in Scode_list:
        try:
            pureNN.main(i,val,margin,period)
            ccc += 1
            print("Process:",len(Scode_list),ccc)
        except MemoryError:
            pass
        pass




if __name__ == '__main__':
    dir = os.getcwd() + '/' + 'Stock_Data_20180605'
    main(dir)
