import tensorflow as tf
import numpy as np

from Read_data_set import *

##
#next batch
##

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

####################
# 신경망 모델 구성 #
####################
#global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32,[None, 57])# [None, 60])
Y = tf.placeholder(tf.float32,[None, 57])# [None, 60])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('Layer1'):
    W1 = tf.Variable(tf.random_normal([57, 256], stddev = 0.01))#60, 256], stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(X, W1))
    L1 = tf.nn.dropout(L1, keep_prob)
with tf.name_scope('Layer2'):
    W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, W2))
    L2 = tf.nn.dropout(L2, keep_prob)
with tf.name_scope('Output'):
    W3 = tf.Variable(tf.random_normal([256,57], stddev = 0.01))# 60], stddev=0.01))
    model = tf.matmul(L2, W3)
with tf.name_scope('Optimizer'):
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
    cost = tf.reduce_mean(tf.square(model - Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#    tf.summary.scalar('cost', cost)
####################
# 신경망 모델 학습 #
####################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    batch_size = 100
    total_batch = int(len(train_x_data) / batch_size)

    for epoch in range(100):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, train_x_data, train_y_data)

            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
            total_cost += cost_val 

        print('Epoch:', '%04d' % (epoch + 1),
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    
    print('최적화 완료!')


#############
# 결과 확인 #
#############
    is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    # print('Training 정확도:', sess.run(accuracy, feed_dict={X: train_x_data, Y: train_y_data, keep_prob:1}))
    print('Test 정확도:', sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1}))
    #f = open("./dataset/position_Result/result.p", 'w')
    #print(model.eval(feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1}))
    array = np.zeros(0, dtype = np.float32)
    sparray = np.zeros(0, dtype = np.float32)
    array = model.eval(feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1})
    #print(array.shape)
    sparray = np.vsplit(array, 999)
    dataset_path = "./dataset/position_Result/"
    print(sparray) # array list
    for number, array in enumerate(sparray):
        file_name = 'Frame'+str(number+1)+'.p'
        f = open(dataset_path+file_name, 'w')
        array = array.astype(np.string_)
        #print(array)
        string = ""
        for i, val in np.ndenumerate(array):
            string += val.decode('UTF-8')+" "
            #print("{}번째\n".format(i[1]))
            if (i[1]+1)%3 == 0:
                #print("{}번째,{}".format((i[1]+1)/3, string))
                f.write("{}\n".format(string))
                string = ""
        f.close()
    
    #f.write(data)
    #f.close()
