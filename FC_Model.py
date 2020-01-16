import tensorflow as tf
import numpy as np

from Read_data_set import *

###
# 신경망 모델 구성
###

X = tf.compat.v1.placeholder(tf.float32, [999, 60])
Y = tf.compat.v1.placeholder(tf.float32, [999, 60])

W1 = tf.Variable(tf.random.uniform([60, 32], -1., 1.))
W2 = tf.Variable(tf.random.uniform([32, 60], -1., 1.))
b1 = tf.Variable(tf.zeros([32]))
b2 = tf.Variable(tf.zeros([60]))

L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)

model = tf.nn.softmax(L2)

cost = tf.reduce_mean(tf.square(model - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(cost)

###
# 신경망 학습
###
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(15):
    _, cost_val = sess.run([train_op, cost], feed_dict={X:train_x_data, Y:train_y_data})
    print(step + 1, '-cost = ', cost_val)

###
# 결과 확인
###
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:%.2f'% sess.run(accuracy*100, feed_dict = {X:test_x_data, Y:test_y_data}))
