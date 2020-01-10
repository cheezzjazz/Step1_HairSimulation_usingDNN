import tensorflow as tf
import numpy as np

from Read_data_set import *

###
# 신경망 모델 구성
###

X = tf.placeholder(tf.float32, [None, 60])
Y = tf.placeholder(tf.float32, [None, 60])

W = tf.Variable(tf.random_uniform([60, 60], -1., 1.))
print(W)
b = tf.Variable(tf.zeros([60]))
L = tf.add(tf.matmul(X, W), b)
print(L)
L = tf.nn.relu(L)
model = tf.nn.softmax(L)
print(model)
cost = tf.reduce_mean(tf.square(model - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train_op = optimizer.minimize(cost)

###
# 신경망 학습
###
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#batch_size = 9
#total_batch = int(HairData.train.num_examples / batch_size)
for epoch in range(10):
#    total_cost = 0
    #print(train_op)
    cost_val =sess.run(train_op, feed_dict={X:train_x_data, Y:train_y_data})
    #print(cost_val)

#print('Epoch :','%04d'% (epoch + 1), 'cost =', '{:.3f}'.format(cost_val))


###
# 결과 확인
###
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict = {X:train_x_data, Y:train_y_data}))#test_x_data, Y:test_y_data}))
