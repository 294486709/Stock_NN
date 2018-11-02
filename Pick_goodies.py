import os
import pandas as pd
iter = 10
limit = 26


def main():
    flag = False
    dir = os.getcwd() + '/' + 'res.txt'
    data = pd.read_csv(dir, sep=",", header=None)
    data.columns = ["Symbol", "Repeat", "Predict", "Acc1", 'Acc2', 'Acc3', 'Counter']
    Scode = data.Symbol
    Counter = data.Counter
    data.drop(['Symbol', 'Acc1','Acc2','Acc3'], axis=1, inplace=True)
    data_processed = data.as_matrix()
    print(data_processed[0][1])
    for i in range(len(data_processed)):
        a = data_processed[i][1]
        a = a[1]
        a = int(a)
        data_processed[i][1] = a

    idx = []
    check = []
    for i in range(len(data_processed)):
        a = (i+1)%10
        b = 0
        if a != 0:
            pass
        else:
            sum = 0
            for j in range(iter):
                sum += data_processed[i-j][1]
                if data_processed[i-j][1]<2:
                    b = 1
            if sum > limit and b == 0:
                idx.append([i,data_processed[i][1],data_processed[i-1][1],data_processed[i-2][1],Scode[i]])
            elif sum > limit and b == 1:
                check.append([i,data_processed[i][1],data_processed[i-1][1],data_processed[i-2][1],Scode[i]])

    f = open('goodie.txt','w')
    for i in idx:
        f.write(str(i))
        f.write('\n')
    f.close()

    f = open('check.txt','w')
    for i in check:
        f.write(str(i))
        f.write('\n')
    f.close()
    f = open('recheck.txt','w')
    c = 0
    for i in idx:
        f.write(str(i[4]))
        f.write(',')
        f.write(str(Counter[c]))
        f.write('\n')
        c += iter
    f.close()
    c = 0
    f = open('recheck.txt','a')
    for i in check:
        f.write(str(i[4]))
        f.write(',')
        f.write(str(Counter[c]))
        f.write('\n')
        c += iter
    f.close()


if __name__ == '__main__':
    main()
