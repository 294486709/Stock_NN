NUM_ITERS = 4000
#DISPLAY_STEP = NUM_ITERS - 1
DISPLAY_STEP = 10
BATCH = 1000
test_ratio = 0.2
nnc = 1100

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np


def nncore(x_vals,y_vals,xtest,ytest,Scode,acc_tst_best,valid_count,Scode_list):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, x_vals.shape[1]],name='X')
    Y_ = tf.placeholder(tf.float32, [None,5],name='YY')
    L1 = nnc
    L2 = nnc
    L3 = nnc
    L4 = nnc
    L5 = nnc
    L6 = nnc
    L7 = nnc
    L8 = nnc
    L9 = nnc
    L10 = 5
    # weights - initialized with random values from normal distribution mean=0, stddev=0.1
    # output of one layer is input for the next
    W1 = tf.Variable(tf.truncated_normal([x_vals.shape[1], L1], stddev=0.1),name='W1')
    b1 = tf.Variable(tf.zeros([L1]))

    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1),name='W2')
    b2 = tf.Variable(tf.zeros([L2]))

    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([L3]))

    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    b4 = tf.Variable(tf.zeros([L4]))

    W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
    b5 = tf.Variable(tf.zeros([L5]))

    W6 = tf.Variable(tf.truncated_normal([L5, L6], stddev=0.1))
    b6 = tf.Variable(tf.zeros([L6]))

    W7 = tf.Variable(tf.truncated_normal([L6, L7], stddev=0.1))
    b7 = tf.Variable(tf.zeros([L7]))

    W8 = tf.Variable(tf.truncated_normal([L7, L8], stddev=0.1))
    b8 = tf.Variable(tf.zeros([L8]))

    W9 = tf.Variable(tf.truncated_normal([L8, L9], stddev=0.1))
    b9 = tf.Variable(tf.zeros([L9]))

    W10 = tf.Variable(tf.truncated_normal([L9, L10], stddev=0.1))
    b10 = tf.Variable(tf.zeros([L10]))
    # flatten the images, unrole eacha image row by row, create vector[784]
    # -1 in the shape definition means compute automatically the size of this dimension
    XX = tf.reshape(X, [-1, x_vals.shape[1]])
    # Define model
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
    Y5 = tf.nn.relu(tf.matmul(Y4, W5) + b5)
    Y6 = tf.nn.relu(tf.matmul(Y5, W6) + b6)
    Y7 = tf.nn.relu(tf.matmul(Y6, W7) + b7)
    Y8 = tf.nn.relu(tf.matmul(Y7, W8) + b8)
    Y9 = tf.nn.relu(tf.matmul(Y8, W9) + b9)
    Ylogits = tf.matmul(Y9, W10) + b10
    Y = tf.nn.softmax(Ylogits,name='Y1')

    # we can also use tensorflow function for softmax
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_,name='ce')
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='ac')
    predict = tf.arg_max(Ylogits,1,name='predict')
    # training,
    learning_rate = 0.003
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # matplotlib visualisation
    allweights = tf.concat(
        [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1]), tf.reshape(W6, [-1]), tf.reshape(W7, [-1]), tf.reshape(W8, [-1]), tf.reshape(W9, [-1]), tf.reshape(W10, [-1])], 0)
    allbiases = tf.concat(
        [tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1]), tf.reshape(b6, [-1]), tf.reshape(b7, [-1]), tf.reshape(b8, [-1]), tf.reshape(b9, [-1]), tf.reshape(b10, [-1])], 0)

    # Initializing the variables
    init = tf.global_variables_initializer()

    train_losses = list()
    train_acc = list()
    test_losses = list()
    test_acc = list()


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        saver = []
        for i in range(NUM_ITERS+1):
            # training on batches of 100 images with 100 labels
            rand_index = np.random.choice(len(x_vals), size=BATCH)
            batch_X = x_vals[rand_index]
            batch_Y = y_vals[rand_index]


            if i % DISPLAY_STEP == 0:
                # compute training values for visualisation
                acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                                   feed_dict={X: batch_X, Y_: batch_Y})

                acc_tst, loss_tst = sess.run([accuracy, cross_entropy],feed_dict={X: xtest, Y_: ytest})

                print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i, acc_trn, loss_trn, acc_tst, loss_tst))

                train_losses.append(loss_trn)
                train_acc.append(acc_trn)
                test_losses.append(loss_tst)
                test_acc.append(acc_tst)
                res = acc_tst
                if acc_trn > 0.98:
                    break

            acc_test = sess.run(predict, feed_dict={X: xtest})
            # the backpropagationn training step
            sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

        xdatax = np.load(os.getcwd()+'/Stock_Data_IDV/'+Scode+'/oneday.npy')
        result = sess.run(predict,feed_dict={X: xdatax})
        print(result)
        testvres = []
        pluscount = 0
        pureplus = 0
        plus = 0
        plusns = 0
        for i in range(len(acc_test)):
            if acc_test[i] == 4:
                pluscount += 1
                if ytest[i][4] == 1 or ytest[i][3] == 1:
                    plus += 1
                if ytest[i][4] == 1:
                    pureplus += 1
                if ytest[i][4] == 1 or ytest[i][3] == 1 or ytest[i][2] == 1:
                    plusns += 1
        try:
            print('PurePlus:',pureplus/pluscount)
            pureplus /= pluscount
        except ZeroDivisionError:
            print('PurePlus:0.0')
            pureplus = 0
        try:
            print('Plus:',plus/pluscount)
            plus /= pluscount
        except ZeroDivisionError:
            print('Plus:0.0')
            plus = 0
        try:
            print('Plusns:',plusns/pluscount)
            plusns /= pluscount
        except ZeroDivisionError:
            print('Pluens:0.0')
            plusns = 0

        if pureplus > 0.6 and plus > 0.7 and plusns > 0.85:
            valid_count += 1
            saver = tf.train.Saver()
            for i in Scode_list:
                saver.save(sess, os.getcwd() + '/Stock_Data_IDV/' + i + '/'+str(valid_count)+'/stock')

        pass

    return res, acc_tst,acc_test,result,pureplus,plus,plusns,pluscount,valid_count


def main(Scode_List,val,margin,period):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    txdata = np.load(os.getcwd()+'/Stock_Data_IDV/'+Scode_List[0]+'/train.npy')
    tydata = np.load(os.getcwd()+'/Stock_Data_IDV/'+Scode_List[0]+'/ylabel.npy')
    try:
        for i in range(1,len(Scode_List)):
            temp_xdata = np.load(os.getcwd()+'/Stock_Data_IDV/'+Scode_List[i]+'/train.npy')
            temp_ydata = np.load(os.getcwd()+'/Stock_Data_IDV/'+Scode_List[i]+'/ylabel.npy')
            txdata = np.vstack((txdata,temp_xdata))
            tydata = np.vstack((tydata,temp_ydata))
    except FileNotFoundError:
        pass

    res = 'T'

    acc_tst_best = 0
    avg = 0
    Scode = Scode_List[0]
    valid_count = 0
    for i in range(80):
        rand_index = np.random.choice(len(txdata), size=len(txdata))
        sep = int(len(txdata) * (1 - test_ratio))
        txdata = np.array(txdata)
        tydata = np.array(tydata)
        batch_X = txdata[rand_index]
        batch_Y = tydata[rand_index]
        xtrain = batch_X[:sep]
        ytrain = batch_Y[:sep]
        xtest = batch_X[sep:]
        ytest = batch_Y[sep:]
        print('loop count:',str(i))
        try:
            res, acc_tst,acc_test, result,pureplus,plus,plusns,pluscount,valid_count = nncore(xtrain, ytrain, xtest, ytest, Scode, acc_tst_best,valid_count,Scode_List)
            if valid_count == 10:
                print("number of tries is:{}".format(str(i)))
                break
            acc_tst = str(acc_tst)
            fff = open('res.txt','a')
            fff.write(Scode)
            fff.write(',')
            fff.write(str(i))
            fff.write(',')
            fff.write(str(result))
            fff.write(',')
            fff.write(str(pureplus))
            fff.write(',')
            fff.write(str(plus))
            fff.write(',')
            fff.write(str(plusns))
            fff.write(',')
            fff.write(str(pluscount))
            fff.write('\n')
            fff.close()

        except MemoryError:
            print('ERRR')
            pass
        if res == 'T':
            pass
        pass
    return 0



if __name__ == '__main__':
    Scode = 'AB'
    main(Scode)
