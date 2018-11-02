import Sort_Stock_Local as SSL
import datetime as dt
import os
import data_process
import filter_up_down as fud
import Pick_goodies as PG

def create_new_folder():
    now = dt.datetime.now()
    end = str(dt.datetime(now.year, now.month, now.day))
    end = end.replace('-','')
    end = end[:8]
    try:
        os.mkdir('Stock_Data_'+str(end))
    except FileExistsError:
        print('Folder is already existed!')
    try:
        os.remove('Pos.txt')
        os.remove('Neg.txt')
    except FileNotFoundError:
        pass
    return str('Stock_Data_'+str(end))


def main(margin,period,upvalue,val):
    folder_loction = os.getcwd() + '/Stock_Data'
    try:
        os.remove('res.txt')
    except FileNotFoundError:
        pass
    SSL.step1(folder_loction)
    print('Step 1 completed!')
    SSL.step2(folder_loction)
    print('Step 2 completed')
    fud.main(folder_loction)
    f = open('Low_dev_Stock.txt','r')
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].replace('\n','')
    files = os.listdir('Stock_Data')
    for file in files:
        if file[:-4] not in data:
            os.remove('Stock_Data/'+file)
    data_process.main(folder_loction,margin,period,upvalue,val)
    PG.main()
    pass


if __name__ == '__main__':
    main()


# SS.y completed  SS.x not tested
