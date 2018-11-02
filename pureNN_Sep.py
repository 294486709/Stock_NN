NUM_ITERS = 5000
DISPLAY_STEP = NUM_ITERS - 1
BATCH = 100
test_ratio = 0.1

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np


def nncore(x_vals,y_vals,xtest,ytest):
    X = tf.placeholder(tf.float32, [None, 17])
    Y_ = tf.placeholder(tf.float32, [None, 2])
    L1 = 200
    L2 = 100
    L3 = 60
    L4 = 30
    L5 = 2
    # weights - initialized with random values from normal distribution mean=0, stddev=0.1
    # output of one layer is input for the next
    W1 = tf.Variable(tf.truncated_normal([17, L1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([L1]))

    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([L2]))

    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([L3]))

    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    b4 = tf.Variable(tf.zeros([L4]))

    W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
    b5 = tf.Variable(tf.zeros([L5]))
    # flatten the images, unrole eacha image row by row, create vector[784]
    # -1 in the shape definition means compute automatically the size of this dimension
    XX = tf.reshape(X, [-1, 17])
    # Define model
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
    Ylogits = tf.matmul(Y4, W5) + b5
    Y = tf.nn.softmax(Ylogits)

    # we can also use tensorflow function for softmax
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training,
    learning_rate = 0.003
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # matplotlib visualisation
    allweights = tf.concat(
        [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
    allbiases = tf.concat(
        [tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)

    # Initializing the variables
    init = tf.global_variables_initializer()

    train_losses = list()
    train_acc = list()
    test_losses = list()
    test_acc = list()

    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        for i in range(NUM_ITERS + 1):
            # training on batches of 100 images with 100 labels
            rand_index = np.random.choice(len(x_vals), size=BATCH)
            batch_X = x_vals[rand_index]
            batch_Y = y_vals[rand_index]


            if i % DISPLAY_STEP == 0:
                # compute training values for visualisation
                acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases],
                                                   feed_dict={X: batch_X, Y_: batch_Y})

                acc_tst, loss_tst = sess.run([accuracy, cross_entropy],
                                             feed_dict={X: xtest, Y_: ytest})

                print(
                    "#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i, acc_trn, loss_trn, acc_tst, loss_tst))

                train_losses.append(loss_trn)
                train_acc.append(acc_trn)
                test_losses.append(loss_tst)
                test_acc.append(acc_tst)
                res = acc_tst

            # the backpropagationn training step
            sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
    return res

def loadfile(Scode,inter):
    xfiledir = os.getcwd() + '/Stock_Data_IDV/' + Scode + '/'+str(inter) + '_x.txt'
    yfiledir = os.getcwd() + '/Stock_Data_IDV/' + Scode + '/' + str(inter) + '_y.txt'
    xdata = np.loadtxt(xfiledir)
    ydata = list(np.loadtxt(yfiledir))
    counter = 0
    for i in range(len(ydata)):
        if ydata[i] == 1:
            ydata[i] = [0,1]
            counter += 1
        else:
            ydata[i] = [1,0]
    ydata = np.array(ydata)
    rand_index = np.random.choice(len(xdata), size=len(xdata))
    xdata = xdata[rand_index]
    ydata = ydata[rand_index]
    seperator = int(len(xdata)*(1-test_ratio))
    xtrain = xdata[:seperator]
    ytrain = ydata[:seperator]
    xtest = xdata[seperator:]
    ytest = ydata[seperator:]
    return xtrain,ytrain,xtest,ytest


def main(Scode):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    Scode = 'AB'
    resultlist = []
    valuable = [0,1,6,11,15,16,17,18,21,31,38,40]
    for i in range(41):
        if i in valuable:
            xtrain, ytrain, xtest, ytest = loadfile(Scode,i)
            resultlist.append(nncore(xtrain,ytrain,xtest,ytest))
    print(resultlist)


if __name__ == '__main__':
    Scode = 'AB'
    main(Scode)
# Restults

# mnist_single_layer_nn.py acc= 0.9237
# mnist__layer_nn.py TST acc = 0.9534
# mnist__layer_nn_relu_adam.py TST acc = 0.9771

# sample output for 5k iterations
# 0 Trn acc=0.10000000149011612 , Trn loss=229.3443603515625 Tst acc=0.11999999731779099 , Tst loss=230.12518310546875
# 100 Trn acc=0.9300000071525574 , Trn loss=30.25579071044922 Tst acc=0.8877000212669373 , Tst loss=35.22196578979492
# 200 Trn acc=0.8799999952316284 , Trn loss=33.183040618896484 Tst acc=0.9417999982833862 , Tst loss=19.18865966796875
# 300 Trn acc=0.9399999976158142 , Trn loss=21.5306396484375 Tst acc=0.9406999945640564 , Tst loss=19.576183319091797
# ...
# 4800 Trn acc=0.949999988079071 , Trn loss=16.546607971191406 Tst acc=0.9739999771118164 , Tst loss=10.48233699798584
# 4900 Trn acc=1.0 , Trn loss=0.8173556327819824 Tst acc=0.9768000245094299 , Tst loss=11.440749168395996
# 5000 Trn acc=1.0 , Trn loss=0.5762706398963928 Tst acc=0.9771000146865845 , Tst loss=10.08562183380127

