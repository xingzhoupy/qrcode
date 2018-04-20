import tensorflow as tf
import numpy as np
import time
from simple_barcode_detection import detect
import cv2
from scipy.misc import imread

def init_data(epoch,num):
    x_data = list()
    y_data = list()
    # n = [i for i in range(2515)]
    # m = [i for i in range(2504)]
    for i in range(epoch*num,(epoch+1)*num):
        # a = random.choice(n)
        # b = random.choice(m)
        img1 = imread('/home/zx/qrcode/data/_train/train_true/' + str(i) + '.jpg')
        img2 = imread('/home/zx/qrcode/data/_train/train_false/' + str(i) + '.jpg')
        if img1 is not None and img2 is not None:
            img1 = cv2.resize(img1, (100, 100)) //255
            # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
            # img1 = tf.image.per_image_standardization(img1)
            x_data.append(img1)
            y_data.append(1)
            img2 = cv2.resize(img2, (100, 100)) // 255
            x_data.append(img2)
            y_data.append(0)
    return np.array(x_data), np.array(y_data)

def test_data(epoch,num):
    x_data = list()
    y_data = list()
    for i in range(epoch*num,(epoch+1)*num):
        img1 = imread('/home/zx/qrcode/data/_test/_test/' + str(i) + '.jpg')
        img2 = cv2.imread('/home/zx/qrcode/data/test/test_false/'+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        if img1 is not None and img2 is not None:
            img1 = cv2.resize(img1, (100, 100)) // 255
            x_data.append(img1)
            y_data.append(1)
            img2 = cv2.resize(img2, (100, 100)) // 255
            x_data.append(img2)
            y_data.append(0)
    return np.array(x_data),np.array(y_data)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.InteractiveSession(config=config)


# with tf.name_scope("input_reshape"):
# image_shape_input = tf.reshape(x,[-1,260,399,1])
# tf.summary.image("input",image_shape_input,10)

def weight_variable(shape):
    var = tf.Variable(tf.random_normal(shape, stddev=0.01))
    # tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(0.01)(var))
    return var


def bias_variable(shape):

    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# def variable_summaries(var):
#     with tf.name_scope("summaries"):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar("mean", mean)
#         with tf.name_scope("stddev"):
#             sttdv = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar("sttdev", sttdv)
#         tf.summary.scalar("max", tf.reduce_max(var))
#         tf.summary.scalar("min", tf.reduce_min(var))
#         tf.summary.histogram("histogram", var)


def _CNN(x):
    # 输入层
    _input_layer = tf.reshape(x, shape=[-1, 100, 100, 1])
    # 卷积层1
    _conv1 = tf.nn.conv2d(_input_layer, weight_variable([3, 3, 1, 64]), strides=[1, 1, 1, 1], padding="SAME",
                          name="conv1")
    # 激活函数
    _relu1 = tf.nn.relu(_conv1 + bias_variable([64]))
    # 池化层
    _pool1 = tf.nn.max_pool(_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

    # 卷积层2
    _conv2 = tf.nn.conv2d(_pool1, weight_variable([3, 3, 64, 128]), strides=[1, 1, 1, 1], padding="SAME", name="conv2")
    # 激活函数
    _relu2 = tf.nn.relu(_conv2 + bias_variable([128]))
    # 池化层
    _pool2 = tf.nn.max_pool(_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    # 随机失活
    _dropout1 = tf.nn.dropout(_pool2, keep_prob=0.6)

    # 卷积层3
    _conv3 = tf.nn.conv2d(_dropout1, weight_variable([3, 3, 128, 256]), strides=[1, 1, 1, 1], padding="SAME",
                          name="conv3")
    _conv3 = tf.nn.relu(_conv3 + bias_variable([256]))
    # 池化层
    _pool3 = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    _dropout2 = tf.nn.dropout(_pool3, keep_prob=0.6)
    #卷积层
    _conv4 = tf.nn.conv2d(_dropout2,weight_variable([2,2,256,512]),strides=[1,1,1,1],padding="SAME",name="conv4")
    _conv5 = tf.nn.conv2d(_conv4,weight_variable([1,1,512,512]),strides=[1,1,1,1],padding="SAME",name="conv5")
    _pool4 = tf.nn.max_pool(_conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME",name="pool4")
    _dropout3 = tf.nn.dropout(_pool4,keep_prob=0.6)
    w = _dropout3.get_shape().as_list()
    # 全连接层
    _densel1 = tf.reshape(_dropout3, [-1, w[1] * w[2] * w[3]], name="densel1")
    # 激活函数
    _fc1 = tf.nn.relu(tf.matmul(_densel1, weight_variable([w[1] * w[2] * w[3], 1024])) + bias_variable([1024]))
    # 随机失活
    _dropout4 = tf.nn.dropout(_fc1, keep_prob=0.5)
    # 输出层
    _output = tf.nn.relu(tf.matmul(_dropout4, weight_variable([1024, 1])) + bias_variable([1]))

    return {"output": _output}

def train(do_train):
    # with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 100, 100], name="x_input")
    _y = tf.placeholder(tf.float32, [None, 1], name="y_input")
    _x = tf.placeholder(tf.float32,[None,1],name="predtion")
    _pred = _CNN(x)['output']

    # 计算最终输出与标准之间的loss
    # 把均方误差也加入到集合里

    # tf.add_to_collection("losses", tf.reduce_mean(tf.square(_pred - _y)))
    # cost = tf.add_n(tf.get_collection("losses"),name="loss")
    cost = tf.reduce_mean(tf.square(_pred - _y))
    #学习率呈指数下降
    # learning_rate = tf.train.exponential_decay(learning_rate=0.1,decay_rate=0.9,global_step = tf.Variable(0),decay_steps=10)
    #随机梯度下降算法
    # optm = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
    #Adam算法
    optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    _corr = tf.equal(_x,_y)
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 训练次数
    train_epochs = 100
    # 保存model
    saver = tf.train.Saver(max_to_keep=1)
    # # P处理大小
    # batch_size = 50
    if do_train == 0:
        loss = 1
        for epoch in range(train_epochs):
            x_data, y_data = init_data(epoch,35)
            batch_x = x_data
            batch_y = y_data.reshape(len(y_data), 1)
            sess.run(optm, feed_dict={x: batch_x, _y: batch_y})
            avg_cost = sess.run(cost, feed_dict={x: batch_x, _y: batch_y})
            # print(pred)
            print("Epoch %03d/%03d" % (epoch, train_epochs))
            print("Cost:" + str(avg_cost))

            if avg_cost < loss:
                now = time.strftime('%Y-%m-%d-%H')
                loss = avg_cost
                saver.save(sess, '/home/zx/qrcode/model/model-' + now + '-' + str(epoch))

    if do_train == 1:
        saver.restore(sess, '/home/zx/qrcode/model/model-2018-04-19-18-64')
        print("Read Model Ready!")
        train_acc = 0
        test_epochs = 3
        #p处理张数
        pnum = 50
        for i in range(test_epochs):
            x_data,y_data = test_data(i,pnum)
            y_data = y_data.reshape(len(y_data),1)
            prediction = sess.run(_pred, feed_dict={x: x_data})
            for i in range(len(prediction)):
                if prediction[i,] > 0.5:
                    prediction[i,] = 1
                else:
                    prediction[i,] = 0
            train_acc =train_acc+ sess.run(accr, feed_dict={_x: prediction, _y: y_data})
        train_acc = train_acc/test_epochs
        print("Accuracy: " + str(train_acc))

    if do_train == 2:
        saver.restore(sess,'/home/zx/qrcode/model/model-2018-04-19-18-64')
        print("Read Model Ready!")
        img = cv2.imread("/home/zx/qrcode/data/_test/no2.jpg")
        # img = cv2.imread("/home/zx/qrcode/data/_test/no3.jpg", cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (399, 260))
            cropImg = detect(img)
            # cv2.imshow("cropImg",cropImg)
            # cv2.waitKey(0)
            cropImg  = cv2.resize(cropImg,(100,100))
            cv2.imwrite("/home/zx/qrcode/data/_test/demo/0.jpg",cropImg)
            cropImg = cropImg.reshape(1, 100, 100)//255

            print("Load Image Ready!")
            prediction = sess.run(_pred, feed_dict={x: cropImg})
            print(prediction[0,][0])
            if prediction[0,] > 0.5:
                print("是")
            else:
                print("不是")
        # print(sess.run(_pred, feed_dict={x: image}))


if __name__ == '__main__':
    # 0表示训练
    # 1表示测试
    # 2表示使用
    do_train = 2
    train(do_train)
