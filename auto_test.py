import os
import tensorflow as tf
import numpy as np

num_model = 10

def predict(model_path, sample_path,Scode,i):
    sample = np.load(sample_path)
    graph = tf.Graph()
    tf.reset_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path+'/stock.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y1 = graph.get_tensor_by_name("Y1:0")
        result = sess.run(Y1, feed_dict={X: sample})
        result = np.array(result)[0].tolist()
        maxnum = max(result)
        result = result.index(maxnum)
        print(model_path)
        print(result)
        f = open('res.txt','a')
        f.write(Scode)
        f.write(',')
        f.write(str(i))
        f.write(',')
        f.write('[')
        f.write(str(result))
        f.write(']')
        f.write(',')
        f.write('1')
        f.write(',')
        f.write('1')
        f.write(',')
        f.write('1')
        f.write(',')
        f.write('1000')
        f.write('\n')
        f.close()

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    files = os.listdir('Stock_Data_IDV')
    for Scode in files:
        content_label = os.listdir('Stock_Data_IDV/' + Scode + '/')
        if '10' not in content_label:
            print('InVaild')
            continue
        for i in range(1, num_model + 1):
            dir = os.getcwd()
            dir =dir.replace('\\','/')
            model_path = dir+ '/Stock_Data_IDV/' + Scode + '/' + str(i)
            sample_path = dir + '/Stock_Data_IDV/' + Scode + '/oneday.npy'
            predict(model_path, sample_path,Scode,i)



if __name__ == '__main__':
    main()