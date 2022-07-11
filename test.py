import tensorflow as tf

a=tf.reshape(tf.range(64),shape=[4,4,4])

tf.summary.scalar('a',a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.summary.