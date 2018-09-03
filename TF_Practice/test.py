import tensorflow as tf
import numpy as np


data = np.random.randint(1000, size=10000)
x = tf.constant(data, name='x')
y = tf.Variable(x + 5, name='y')
z = tf.Variable(0, name='z')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(5):
        z = z + 1
        print(session.run(z))
