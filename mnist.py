x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])	

hidden_1 = tf.layers.dense(inputs=x, units=512, activation=tf.nn.sigmoid)
hidden_2 = tf.layers.dense(inputs=hidden_1, units=512, activation=tf.nn.sigmoid)
hidden_3 = tf.layers.dense(inputs=hidden_2, units=512, activation=tf.nn.sigmoid)
hidden_4 = tf.layers.dense(inputs=hidden_3, units=512, activation=tf.nn.sigmoid)
hidden_5 = tf.layers.dense(inputs=hidden_4, units=512, activation=tf.nn.sigmoid)
y = tf.layers.dense(inputs=hidden_5, units=10, activation=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(50):
    avg_cost = 0
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        
        avg_cost += c/1000   
    print("Epoch:", '%03d' % (epoch+1), "\tcost={:.9f}".format(avg_cost))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('\n',sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
