import tensorflow as tf
import os
import numpy as np




def true_evaluate(predict,label):
    nData = len(predict)
    counter_predict_money = 0
    counter_predict_right = 0
    for i in range(nData):
        if predict[i][0] < 0.5:
            counter_predict_money += 1
            if label[i][0] == 0:
                counter_predict_right += 1

    print('Predict zhang:',counter_predict_money, 'Zhang accuracy:', counter_predict_right/counter_predict_money)


def fatchindector(Scode):
    xdata = np.load(os.getcwd()+ '/Stock_Data_Pred/'+ Scode+'/today.npy')
    xdata = np.transpose(xdata)
    xdata = xdata.tolist()
    x = np.zeros([1,204])
    for i in range(204):
        x[0][i] = xdata[i]
    y = np.zeros([1,2])
    y[0][0] = 0
    return x,y

def predict(Scode):
    direct = os.getcwd() + '/Stock_Data_IDV/' + Scode + '/'
    filedir = direct + 'stock.meta'
    # xtest = np.loadtxt(direct + 'xtest')
    # ytest = np.loadtxt(direct + 'ytest')
    xtest,ytest = fatchindector(Scode)
    tf.reset_default_graph()
    try:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(filedir)
            saver.restore(sess, tf.train.latest_checkpoint(direct))
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            YY = graph.get_tensor_by_name("YY:0")
            op = graph.get_tensor_by_name('Y:0')
            acc = graph.get_tensor_by_name('ac:0')
            acc_tst = sess.run(op, feed_dict={X: xtest, YY: ytest})
            print(Scode)
            print(acc_tst)
            fl = open(os.getcwd() + '/Stock_Data_IDV/'+Scode +'/acc.txt','r')
            acc = fl.readlines()
            ac = float(acc[len(acc)-1])
            print(ac)
            postive = np.zeros([1,2])
            postive[0][1] = 1
            if acc_tst[0][0] < 0.5 and acc_tst[0][1] > 0.5:
                print('pos')
                f = open('Pos.txt','a')
                f.write(Scode)
                f.write(',')
                f.write(str(ac))
                f.write(',')
                f.write(str(acc_tst))
                f.write('\n')
                f.close()
            else:
                print('neg')
                f = open('Neg.txt','a')
                f.write(Scode)
                f.write('\n')
                f.close()
    except OSError:
        f = open('Trash.txt', 'a')
        f.write(Scode)
        f.write('\n')
        f.close()
    # true_evaluate(acc_tst, ytest)


def main():
    try:
        os.remove('Pos.txt')
        os.remove('Neg.txt')
    except FileNotFoundError:
        pass
    ScodeDir = os.getcwd() + '/Stock_Data_IDV/'
    files = os.listdir(ScodeDir)
    for file in files:
        predict(file)



if __name__ == '__main__':
    main()
