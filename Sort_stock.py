import numpy as np
import os




def read_data(name):
    data = list(np.load(name))
    data.pop(0)
    data = np.array(data)
    return data


def generate_x(data,margin = 10,period = 3,upvalue = 10):
    # close_price = np.zeros([len(data),1])
    #
    # print(len(data))
    # counter = 1
    # for i in range(len(data)):
    #     data[i,0] = counter
    #     counter += 1
    #     close_price[i] = data[i,4]
    #
    # a = data[0][1]
    # print(a)

# dim 0-23 * 4000+

    sort_data = np.zeros([(len(data)-margin-1),(margin*len(data[0]))])
# dim 4616 * 360

    for i in range(len(sort_data)-period):
        counter = 0
        for j in range(len(data[0])):
            for k in range(margin):
                sort_data[i,(j*margin+k)] = data[i+k,j]

    return sort_data

def generate_3D_x(data,margin = 10,period = 3,upvalue = 10):
    X_3d = np.zeros([margin,(len(data[0]))])
    for i in range(margin):
        for j in range(len(data[0])):
            X_3d[i][j] = data[0 + i - 1][j]
    X_3d = X_3d.transpose()
    for k in range(51, len(data)-margin - period):
        X_2d = np.zeros([margin,len(data[0])])
        for i in range(margin):
            for j in range(len(data[0])):
                X_2d[i][j] = data[k+i-1][j]
        X_2d = X_2d.transpose()
        X_3d = np.dstack((X_3d, X_2d))

    return X_3d

def gen3d_new(data,y,Scode,margin = 10,period = 3,upvalue = 10):
    data = data.transpose().tolist()
    num_feature = len(data)
    data_sep = []
    for i in range(num_feature):
        temp = []
        for j in range(margin-1,len(data[0])-1):
            temmp = []
            for k in range(margin):
                temmp.append(data[i][j+k-margin])
            temp.append(temmp)
        data_sep.append(temp)
    for i in range(50):
        for j in range(len(data_sep)):
            data_sep[j].pop(0)
    for i in range(1):
        for j in range(len(data_sep)):
            data_sep[j].pop(len(data_sep[j])-1)
    for i in range(num_feature):
        currentfeature = data_sep[i]
        ttemp = []
        for j in range(len(currentfeature)-1):
            temp = []
            temp.append(currentfeature[j])
            temp.append(int(y[j]))
            ttemp.append(temp)
        data_sep[i] = ttemp
    for i in range(num_feature):
        current = data_sep[i]
        xs = []
        ys = []
        for j in range(len(current)):
            xs.append(current[j][0])
            ys.append(current[j][1])
        xs = np.array(xs)
        ys = np.array(ys)
        np.savetxt(os.getcwd()+'/Stock_Data_IDV/'+Scode+'/'+str(i)+'_x.txt',xs)
        np.savetxt(os.getcwd()+'/Stock_Data_IDV/'+Scode+'/' + str(i) + '_y.txt', ys)



    print('data')


def generate_y(data,margin = 10,period = 3,upvalue = 10):
# sort_data dim 4616 * 360
    flag1 = False
    flag2 = False
    y = np.zeros([len(data)- period,1],dtype=np.float64)
    for i in range(margin, len(data) - period):
        perioplist = []
        perioplist = [data[i+j][0] for j in range(period)]
        counter = 0
        for k in perioplist:
            if k > 0:
                counter += 1
        if counter == period:
            flag1 = True
        else:
            continue
        value = 0
        for k in perioplist:
            value += k
        if value > upvalue:
            flag2 = True
        #print(flag1,flag2,i)
        if flag2 == True and flag1 == True:
            y[i] = 1

    # for i in range(len(sort_data)):
    #     price_0 = np.float(data[i,4])
    #     price_1 = np.float(data[(i+1),4])
    #     diff = (price_1 - price_0)/price_0
    #     if diff>0 and diff<=0.01:
    #         y[i] = 1
    #     elif diff>0.01 and diff<=0.02:
    #         y[i] = 2
    #     elif diff>0.02 and diff<=0.03:
    #         y[i] = 3
    #     elif diff>0.03 and diff<=0.04:
    #         y[i] = 4
    #     elif diff>0.04 and diff<=0.05:
    #         y[i] = 5
    #     elif diff>0.05 and diff<=0.06:
    #         y[i] = 6
    #     elif diff>0.06 and diff<=0.07:
    #         y[i] = 7
    #     elif diff>0.07 and diff<=0.08:
    #         y[i] = 8
    #     elif diff>0.08 and diff<=0.09:
    #         y[i] = 9
    #     elif diff>0.09 and diff<=0.1:
    #         y[i] = 10
    #     elif diff>0.1 and diff<=0.15:
    #         y[i] = 15
    #     elif diff>0.15 and diff<=0.3:
    #         y[i] = 30
    #print('Stop')
    return y

def generate_y_2(y,margin = 10,period = 3,upvalue = 10):
    y_2 = np.zeros([len(y),1])

    for i,j in enumerate(y):
        if j > 0:
            y_2[i] = 1
    return y_2

def popy(y,margin = 10,period = 3,upvalue = 10):
    y = list(y)
    for i in range(margin+50):
        y.pop(0)
    return y


def generatesample(X3D,Y,margin = 10,period = 3,upvalue = 10):
    num_sample = len(Y)
    newlist = []
    for i in range(num_sample):
        temp = []
        temp.append(X3D[:,:,i])
        temp.append(Y[i])
        newlist.append(temp)
    zipped = newlist
    #print(len(newlist),len(newlist[0]))
    num_90 = int(0.9*num_sample)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(num_90):
        train_x.append(zipped[i][0])
        train_y.append(zipped[i][1])
    for i in range(num_90,len(Y)):
        test_x.append(zipped[i][0])
        test_y.append(zipped[i][1])
    test_y = np.array(test_y)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    train_x = np.array(train_x)
    return train_x, test_x, train_y, test_y


def deletevalue(data):
    data = list(data)
    for i in range(len(data)):
        data[i].pop(0)
        data[i].pop(0)


def main(name,Scode,margin = 10,period = 3,upvalue = 10):
    data = read_data(name)
    #sort_data = generate_x(data)
    y = generate_y(data,margin,period,upvalue)
    y = popy(y,margin,period,upvalue)

    X_3d = gen3d_new(data,y,Scode,margin,period,upvalue)


    # train_x, test_x, train_y, test_y = generatesample(X_3d,y)
    # np.save(name + "train_x", train_x)
    # np.save(name + "test_x", test_x)
    # np.save(name + "train_y", train_y)
    # np.save(name + "test_y", test_y)
    # np.save("sample_bulk",sort_data)
    # np.save("y_comp", y)
    # np.save("y_simp", y_2)

