import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# We use thr TF helper function to pull down the data from MNIST site

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x is placecholder for the 28 X 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784], name="abc")
y_true = tf.placeholder(tf.int32, shape=[None, 10], name="pl2")

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_pred = tf.add(tf.matmul(x, W), b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_predictor = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# Probabilities of each outcome, for three samples(one-hot encoding, though these are not actually probability)
# y_pred =[ [1.0, 1.5, 1.8],         y_true=[[1.2, 1.6, 1.9],
#           [2.0, 2.8, 2.4],                 [2.1, 3.0, 2.5],
#           [3.7, 3.2, 3.1]]                 [3.1, 3.8, 3.2]]

# a= tf.argmax(y_pred, 1)= [2,  1, 0]
# b= tf.argmax(y_true, 1)= [2,  1, 1]
# Correct_predictor =  tf.equal(a,b)= [True, True, False]


# tf.cast converts all the False's to "0." and all the True's to "1." in numpy array Correct Predictor
# reduce_mean simply finding mean which is actually "total no of 1/ total values"
accuracy = tf.reduce_mean(tf.cast(correct_predictor, tf.float32))

num_iters = 1000
batch_size = 100
# tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(num_iters):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y_true: batch_y})
    # index = [8, 33, 63, 77, 124, 149, 195, 1, 11, 241, 245]
    # # plt.imshow(mnist.test.images[index].reshape((11,28, 28)), cmap='Greys')
    # predicted, actual = sess.run([tf.argmax(y_pred,1),tf.argmax(y_true,1)], feed_dict={x: mnist.test.images[index].reshape(11,784), y_true: mnist.test.labels[index].reshape(11,10)})
    # print("predicted = {}\n   actual = {}".format(predicted,actual))
    #
    # # print(sess.run(correct_predictor, feed_dict={x: mnist.test.images[index].reshape(11,784), y_true: mnist.test.labels[index].reshape(11,10)}))
    # arr =sess.run(correct_predictor, feed_dict={x: mnist.test.images[index].reshape(11,784), y_true: mnist.test.labels[index].reshape(11,10)})
    # print(len(arr.tolist()))
    # print(arr.tolist().count(True))
    # print("accuracy = {:.2%}".format(sess.run(accuracy, feed_dict={x: mnist.test.images[index].reshape(11,784), y_true: mnist.test.labels[index].reshape(11,10)})))
    # plt.show()

    # predicted, actual = sess.run([tf.argmax(y_pred,1),tf.argmax(y_true,1)], feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
    # print("predicted = {} \n actual = {}".format(predicted, actual))
    #
    # arr =sess.run(correct_predictor, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
#     list = arr.tolist()
#     print("total samples = ", len(list))
#     print("no of times False occurs = ",list.count(False))
#     indices = [i for i, j in enumerate(list) if j==False]
#     print("indices of false=",indices)
    
    # print("accuracy = {:.2%}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})))

#     index = 90
#     plt.imshow(mnist.test.images[index].reshape(28, 28), cmap='Greys')
#     predicted, actual = sess.run([tf.argmax(y_pred,1),tf.argmax(y_true,1)], feed_dict={x: mnist.test.images[index].reshape(1,784), y_true: mnist.test.labels[index].reshape(1,10)})
#     print("predicted = {}\n   actual = {}".format(predicted,actual))
#     #
#     # print(sess.run(correct_predictor, feed_dict={x: mnist.test.images[index].reshape(1,784), y_true: mnist.test.labels[index].reshape(1,10)}))
#     arr =sess.run(correct_predictor, feed_dict={x: mnist.test.images[index].reshape(1,784), y_true: mnist.test.labels[index].reshape(1,10)})
#     print(len(arr.tolist()))
#     print(arr.tolist().count(True))
#     print("accuracy = {:.2%}".format(sess.run(accuracy, feed_dict={x: mnist.test.images[index].reshape(1,784), y_true: mnist.test.labels[index].reshape(1,10)})))
#     plt.show()



    # print(type(mnist.test.images[index]))

    # print(sess.run(tf.cast(correct_predictor, tf.float32), feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))


