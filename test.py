import tensorflow as tf

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5, 7, 9, 11, 13, 15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)

def loss_func(a, b):
    pred_y = train_x * a + b
    return tf.keras.losses.mse(train_y, pred_y)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(2900):
    opt.minimize(lambda:loss_func(a,b), var_list=[a, b])
    print(a.numpy(), b.numpy())