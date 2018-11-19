# coding:utf-8
# 两层简单神经网络（全连接）
import tensorflow as tf


# 定义输入参数
x1 = tf.placeholder(tf.float32, shape=(None, 2))
# stddev 标准差 seed 随机种子
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# 定义前向传播过程
a1 = tf.matmul(x1, w1)
y1 = tf.matmul(a1, w2)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y1 is:\n", sess.run(y1, feed_dict={x1: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
    print("w1 :\n", sess.run(w1))
    print("w2 :\n", sess.run(w2))