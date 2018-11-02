global a
global b
a = 1
b = 2


def aaa():
    print(a,b)
def main(a=0,b=0):
    aaa()

if __name__ == '__main__':
    main(3,5)