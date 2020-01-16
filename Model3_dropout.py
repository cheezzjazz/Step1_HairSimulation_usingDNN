#import tensorflow as tf
import tensorflow as tf
#tf.disable_v2_behavior

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
#print('x-----------------------\n')
#print(train_x_data)
#print('y----------------------\n')
#print(train_y_data)
#batch_x, batch_y = next_batch(100, train_x_data, train_y_data)
#print('batch x-----------------\n')
#print(batch_x.shape)
#print('batch y-----------------\n')
#print(batch_y.shape)
####################
# 신경망 모델 구성 #
####################

X = tf.placeholder(tf.float32, [None, 60])
Y = tf.placeholder(tf.float32, [None, 60])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([60, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 60], stddev=0.01))
model = tf.matmul(L2, W3)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
cost = tf.reduce_mean(tf.square(model - Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

####################
# 신경망 모델 학습 #
####################

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(999 / batch_size)
#total_batch = int(mnist.train.num_examples / batch_size)

#for epoch in range(15): # dropout 사용 전
for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, train_x_data, train_y_data)
#        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

#        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys}) # dropout 사용 전
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val 

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')


#############
# 결과 확인 #
#############

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})) # dropout 사용 전
print('정확도:', sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1}))


