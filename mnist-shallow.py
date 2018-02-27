import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

model_path = 'mnist-shallow/'

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
hidden_units = 1024

he_init = tf.contrib.layers.variance_scaling_initializer()

hidden_1 = tf.layers.dense(inputs=x, units=hidden_units, activation=tf.nn.relu, kernel_initializer=he_init)
hidden_2 = tf.layers.dense(inputs=hidden_1, units=hidden_units, activation=tf.nn.relu, kernel_initializer=he_init)
hidden_3 = tf.layers.dense(inputs=hidden_2, units=hidden_units, activation=tf.nn.relu, kernel_initializer=he_init)
hidden_4 = tf.layers.dense(inputs=hidden_3, units=hidden_units, activation=tf.nn.relu, kernel_initializer=he_init)
hidden_5 = tf.layers.dense(inputs=hidden_4, units=hidden_units, activation=tf.nn.relu, kernel_initializer=he_init)
y = tf.layers.dense(inputs=hidden_5, units=10, kernel_initializer=he_init)

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(50):
        avg_cost = 0
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, cost = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            avg_cost += cost/1000   
        print("Epoch:", '%02d'%(epoch+1), "\tcost={:.9f}".format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('\nAccuracy:',sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
    saver.save(sess, model_path)
        
