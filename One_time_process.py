import pandas as pd
import os

def main():
    dir = os.getcwd() + '/' + 'stand_dev.txt'
    data = pd.read_csv(dir, sep=",", header=None)
    data.columns = ["Symbol", "Stand_dev"]

    data['sort_idx'] = data['Stand_dev'].rank(ascending=0, method='first')

    a = round(len(data) * 0.6)

    Scode = []
    for i in range(len(data)):
        if data.sort_idx[i] > a:
            Scode.append([data.Symbol[i]])

    f = open('Low_dev_Stock.txt','w')
    for i in Scode:
        ############################################################################
        f.write(str(i[0]))
        ############################################################################
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()