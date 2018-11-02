import main as mn
import os


def main():
    try:
        os.remove('Neg.txt')
        os.remove('Pos.txt')
    except FileNotFoundError:
        pass
    margin = [20]
    period = [1]
    effectup = [3]
    val = [8,13,20,33]
    # val = [3,5,7,8,9,25,26,34]
    # mn.main(17,1,2,val)
    for i in margin:
        for j in effectup:
            for k in period:
                for g in range(1):
                    #print(i, k, j, v)
                    mn.main(i, k, j, val)
                    f=open('RESULT.TXT','a')
                    f.write('margin = ')
                    f.write(str(i))
                    f.write('     effectup = ')
                    f.write(str(j))
                    f.write('     period = ')
                    f.write(str(k))
                    f.write('\n')
                    f.write('-------------------------------------------------')
                    f.close()


if __name__ == '__main__':
    main()
