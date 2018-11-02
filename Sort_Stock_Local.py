import os
validperiod = 50
price_low_limit = 5
volumn_low_limit = 100
on_market_date_limit = 2000


def step1(dir):
    file = os.listdir(dir)
    file1 = open('processed_NASDAQ.txt','r')
    file2 = open('processed_NYSE.txt','r')
    file1list = [str(i).replace('\n','') for i in file1]
    file2list = [str(i).replace('\n','') for i in file2]
    file_treated = [i[:-4] for i in file]
    poplist = []
    for i,j in enumerate(file_treated):
        if j in file1list or j in file2list:
            pass
        else:
            poplist.append(i)
    poplist.sort(reverse=True)
    removelist = []
    for i in poplist:
        removelist.append(file[i])
    try:
        for i in removelist:
            os.remove(dir + '/' + i)
    except ValueError:
        pass
    file3 = open('blacklist.txt','r')
    file3data = file3.readlines()
    data=[]
    for i in file3data:
        data.append(i.replace('\n',''))
    try:
        for i in data:
            os.remove(dir+'/'+i+'.txt')
    except ValueError and FileNotFoundError:
        pass


def step2(dir):
    files = os.listdir(dir)
    deletelist = []
    for file in files:
        f = open(dir + '/' + file,'r')
        content = f.readlines()
        content1 = []
        for i in content:
            i = i.replace('\n','')
            i = i.split(' ')
            content1.append(i)
        f.close()
        num_content = len(content1)
        if num_content < on_market_date_limit:
            deletelist.append(file)
            continue
        for i in range(num_content-validperiod):
            content1.pop(0)
        close_price = 0
        volumn = 0
        for i in range(validperiod):
            close_price += float(content1[i][4])
            volumn += int(content1[i][5])
        close_price /= validperiod
        volumn /= validperiod

        if close_price < price_low_limit or volumn < volumn_low_limit:
            deletelist.append(file)
    for i in deletelist:
        os.remove(dir + '/' + i)


if __name__ == '__main__':
    #dir = os.getcwd()+'/'+'Stock_Data_20180605'
    step1(dir)
    step2(dir)
