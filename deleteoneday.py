import os


def deleteoneday(Scode):
    f = open(os.getcwd() + '/Stock_Data/'+Scode,'r')
    content = f.readlines()
    pass
    content.pop((len(content)-1))
    f1 = open(os.getcwd() + '/Processed/'+Scode,'w')
    for i in content:
        f1.write(i)
    f1.close()
    f.close()
    print(Scode,'finish')



def main():
    try:
        os.mkdir('Processed')
    except FileExistsError:
        pass
    dir = os.getcwd() + '/Stock_Data/'
    files = os.listdir(dir)
    for file in files:
        deleteoneday(file)



if __name__ == '__main__':
    main()