import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)
print(result)

# tf.Session() method is no longer used in TensorFlow versions 2.x and above.
# sess = tf.Session()
# print(sess.run(result))
# sess.close()

# with tf.Session() as sess:
#     output = sess.run(result)
#     print(output)
# print(output)

print(result.numpy())

