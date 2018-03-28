import tensorflow as tf
import numpy as np
import librosa

s, sr = librosa.load('train_clean_male.wav', sr=None)
S = librosa.stft(s, n_fft=1024, hop_length=512)
sn, sr = librosa.load('train_dirty_male.wav', sr=None)
X = librosa.stft(sn, n_fft=1024, hop_length=512)

S_mod = abs(S).T
X_mod = abs(X).T

indices = np.arange(X_mod.shape[0])
np.random.shuffle(indices)
X_mod_shuffled = X_mod[indices]
S_mod_shuffled = S_mod[indices]

x = tf.placeholder(tf.float32, [None, X_mod.shape[1]])
y_ = tf.placeholder(tf.float32, [None, S_mod.shape[1]])

# boolean for is_traiaing in batch_norm
phase = tf.placeholder(tf.bool)

hidden_units = 1000
activation_func = tf.nn.leaky_relu
he_init = tf.contrib.layers.variance_scaling_initializer()

hidden_1 = tf.layers.dense(inputs=x, 
                           units=hidden_units, 
                           kernel_initializer=he_init, 
                           activation=activation_func)
#dropout_1 = tf.layers.dropout(hidden_1, rate=.8)

hidden_2 = tf.layers.dense(inputs=hidden_1, 
                           units=hidden_units, 
                           kernel_initializer=he_init, 
                           activation=activation_func)
#dropout_2 = tf.layers.dropout(hidden_2, rate=.8)

y = tf.layers.dense(inputs=hidden_2, 
                    units=S_mod.shape[1], 
                    kernel_initializer=he_init, 
                    activation=tf.nn.relu)


'''
# batch normalization

hidden_1 = tf.layers.dense(inputs=x, 
                           units=hidden_units, 
                           kernel_initializer=he_init)
batch_norm_1 = tf.contrib.layers.batch_norm(hidden_1, is_training=phase, activation_fn=activation_func)
#dropout_1 = tf.layers.dropout(batch_norm_1, rate=.8)


hidden_2 = tf.layers.dense(inputs=batch_norm_1, 
                           units=hidden_units, 
                           kernel_initializer=he_init)
batch_norm_2 = tf.contrib.layers.batch_norm(hidden_2, is_training=phase, activation_fn=activation_func)
#dropout_2 = tf.layers.dropout(batch_norm_2, rate=.8)

y = tf.layers.dense(inputs=batch_norm_2, 
                    units=S_mod.shape[1], 
                    kernel_initializer=he_init, 
                    activation=tf.nn.relu)
'''

mse = tf.reduce_mean(tf.losses.mean_squared_error(y_, y))
train_step = tf.train.AdamOptimizer().minimize(mse)

mini_batch_size = 32
mini_batches = int(X_mod.shape[0]/mini_batch_size)

batch_xs, batch_ys = np.array_split(X_mod_shuffled, mini_batches), np.array_split(S_mod_shuffled, mini_batches)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epochs = 5
display_step = 1

for epoch in range(epochs):
    avg_cost = 0
    for mini_batch in range(mini_batches):
        _, cost = sess.run([train_step, mse], {x: batch_xs[mini_batch], y_: batch_ys[mini_batch], phase: True})
        avg_cost += cost/mini_batches 
    if (epoch+1)%display_step==0:
        print("Epoch:", '%03d'%(epoch+1), "\tcost={:.9f}".format(avg_cost))

sn, sr = librosa.load('test_x_01.wav', sr=None)
X_test = librosa.stft(sn, n_fft=1024, hop_length=512).T
X_test_mod = abs(X_test)

Sh_test = sess.run(y, feed_dict={x:X_test_mod})
Sh = np.multiply(np.divide(X_test, X_test_mod), Sh_test)

sh_test = librosa.istft(Sh.T, win_length=1024, hop_length=512)
librosa.output.write_wav('test_s_01_recons.wav', sh_test, sr)

sn, sr = librosa.load('test_x_02.wav', sr=None)
X_test = librosa.stft(sn, n_fft=1024, hop_length=512).T
X_test_mod = abs(X_test)

Sh_test = sess.run(y, feed_dict={x:X_test_mod})
Sh = np.multiply(np.divide(X_test, X_test_mod), Sh_test)

sh_test = librosa.istft(Sh.T, win_length=1024, hop_length=512)
librosa.output.write_wav('test_s_02_recons.wav', sh_test, sr)

sess.close()