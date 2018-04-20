import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#输入节点
n_input = 784
#输出节点数
n_outout = 10
#权重
weights = {
    #卷积层权重
    'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
    #卷积层权重
    'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=0.1)),
    #全连接层权重
    'wd1': tf.Variable(tf.random_normal([7*7*128,1024],stddev=0.1)),
    #输出层权重
    'wd2':tf.Variable(tf.random_normal([1024,n_outout],stddev=0.1))
}
biases = {
    'bc1':tf.Variable(tf.random_normal([64],stddev=0.1)),
    'bc2':tf.Variable(tf.random_normal([128],stddev=0.1)),
    'bd1':tf.Variable(tf.random_normal([1024],stddev=0.1)),
    'bd2':tf.Variable(tf.random_normal([n_outout],stddev=0.1))
}


def conv_basic(_input, _w, _b, _keepratio):
    # 输入层
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # 卷积层1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding="SAME")
    # 激活函数
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    # 池化层
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 随机失活
    _pool_drl = tf.nn.dropout(_pool1, _keepratio)
    # conv layer2
    _conv2 = tf.nn.conv2d(_pool_drl, _w['wc2'], strides=[1, 1, 1, 1], padding="SAME")
    # 激活函数
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    # 池化层
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 随机失活
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    # 全连接层
    _densel = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    # 激活函数
    _fcl = tf.nn.relu(tf.add(tf.matmul(_densel, _w['wd1']), _b['bd1']))
    # 随机失活
    _fc_dr1 = tf.nn.dropout(_fcl, _keepratio)
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1,
        'pool_dr1': _pool_drl, 'conv2': _conv2, 'pool2': _pool2,
        'pool_dr2': _pool_dr2, 'dense1': _densel, 'fc1': _fcl,
        'fc_dr1': _fc_dr1, 'out': _out
    }

    return out
x = tf.placeholder(tf.float32,[None,784])
y= tf.placeholder(tf.float32,[None,10])
keepratio = tf.placeholder(tf.float32)
_pred = conv_basic(x,weights,biases,keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels=y))
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1),tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(_corr,tf.float32))

do_train =1
sess = tf.Session()
sess.run(tf.global_variables_initializer())
trian_epochs = 20
saver = tf.train.Saver(max_to_keep=2)
batch_size = 50
display_step = 1
if do_train == 1:
    for epoch in range(trian_epochs):
        avg_cost = 0
        #     total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = 10
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.6})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio: 0.6})

        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, trian_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio: 1.})
            print("Training accuracy: %.3f" % (train_acc))
            saver.save(sess, 'model/cnn/cnn-model-' + str(epoch))
    print("Finsh!")
