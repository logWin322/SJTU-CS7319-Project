import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
from util import get_variable


# Lmser+DNN模型
class Lmser():
    def __init__(self,):
        self.input_x = tf.placeholder(tf.float32, [None, 32*32*3])    # value in the range of (0, 1)
        self.input_y = tf.placeholder(tf.int32, [None, 10])
        self.lr = tf.placeholder("float")
        # 计算编码器
        w_list = [None for i in range(4)]
        w_list[0] = get_variable('xavier', name='w0', shape=[32*32*3, 256])
        w_list[1] = get_variable('xavier', name='w1', shape=[256, 128])
        w_list[2] = get_variable('xavier', name='w2', shape=[128, 64])
        w_list[3] = get_variable('xavier', name='w3', shape=[64, 16])
        encode_h0 = tf.nn.leaky_relu(tf.matmul(self.input_x, w_list[0]))
        encode_h1 = tf.nn.leaky_relu(tf.matmul(encode_h0, w_list[1]))
        encode_h2 = tf.nn.leaky_relu(tf.matmul(encode_h1, w_list[2]))
        encode_h3 = tf.nn.leaky_relu(tf.matmul(encode_h2, w_list[3]))
        # 通过伪逆计算解码器
        deta_w = [None for i in range(4)]
        for i in range(4):
            s, u, v = tf.svd(w_list[i])
            mid = tf.eye(s.get_shape().as_list()[0])
            mid = mid * 1.0 / tf.expand_dims(s, 1)
            deta_w[i] = tf.matmul(tf.matmul(v, mid), tf.transpose(u, perm=[1, 0]))
            print(deta_w[i].shape)
        decode_h0 = tf.nn.leaky_relu(tf.matmul(encode_h3, deta_w[3]))
        decode_h1 = tf.nn.leaky_relu(tf.matmul(decode_h0, deta_w[2]))
        decode_h2 = tf.nn.leaky_relu(tf.matmul(decode_h1, deta_w[1]))
        self.output_x = tf.nn.sigmoid(tf.matmul(decode_h2, deta_w[0]))

        # classifier 将每一层的embedding做一个组合后过深度神经网络分类
        h = tf.concat([self.input_x, encode_h0, encode_h1, encode_h2, encode_h3], axis=1)
        classifier_w = [None for i in range(4)]
        classifier_b = [None for i in range(4)]
        classifier_nn = [3536, 1024, 256, 64, 10]
        for i in range(len(classifier_nn)-1):
            classifier_w[i] = get_variable('xavier', name='classifier_w_%d' % i, shape=[classifier_nn[i], classifier_nn[i+1]])
            classifier_b[i] = get_variable('zero', name='classifier_b_%d' % i, shape=[classifier_nn[i+1]])
            h = tf.matmul(h, classifier_w[i]) + classifier_b[i]
            if i < len(classifier_nn)-2:
                h = tf.nn.leaky_relu(h)
        self.output_y = h

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss1 = tf.losses.mean_squared_error(labels=self.input_x, predictions=self.output_x)
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8).minimize(self.loss1)
                self.loss2 = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=self.output_y)
                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8).minimize(self.loss2)
        
        correct_prediction = tf.equal(tf.argmax(self.output_y, 1), tf.argmax(self.input_y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Lmser+CNN模型
class Lmser_CNN():
    def __init__(self, batch_size=256):
        self.input_x = tf.placeholder(tf.float32, [None, 32 * 32 * 3])  # value in the range of (0, 1)
        self.input_y = tf.placeholder(tf.int32, [None, 10])
        self.lr = tf.placeholder("float")
        # 计算编码器
        w_list = [None for i in range(6)]
        w_list[0] = get_variable('xavier', name='w0', shape=[32 * 32 * 3, 1024])
        w_list[1] = get_variable('xavier', name='w1', shape=[1024, 512])
        w_list[2] = get_variable('xavier', name='w2', shape=[512, 256])
        w_list[3] = get_variable('xavier', name='w3', shape=[256, 128])
        w_list[4] = get_variable('xavier', name='w4', shape=[128, 64])
        w_list[5] = get_variable('xavier', name='w5', shape=[64, 16])
        encode_h0 = tf.nn.leaky_relu(tf.matmul(self.input_x, w_list[0]))
        encode_h1 = tf.nn.leaky_relu(tf.matmul(encode_h0, w_list[1]))
        encode_h2 = tf.nn.leaky_relu(tf.matmul(encode_h1, w_list[2]))
        encode_h3 = tf.nn.leaky_relu(tf.matmul(encode_h2, w_list[3]))
        encode_h4 = tf.nn.leaky_relu(tf.matmul(encode_h3, w_list[4]))
        encode_h5 = tf.nn.leaky_relu(tf.matmul(encode_h4, w_list[5]))
        # 通过伪逆计算解码器
        deta_w = [None for i in range(6)]
        for i in range(6):
            s, u, v = tf.svd(w_list[i])
            mid = tf.eye(s.get_shape().as_list()[0])
            mid = mid * 1.0 / tf.expand_dims(s, 1)
            deta_w[i] = tf.matmul(tf.matmul(v, mid), tf.transpose(u, perm=[1, 0]))
            print(deta_w[i].shape)
        decode_h0 = tf.nn.leaky_relu(tf.matmul(encode_h5, deta_w[5]))
        decode_h1 = tf.nn.leaky_relu(tf.matmul(decode_h0, deta_w[4]))
        decode_h2 = tf.nn.leaky_relu(tf.matmul(decode_h1, deta_w[3]))
        decode_h3 = tf.nn.leaky_relu(tf.matmul(decode_h2, deta_w[2]))
        decode_h4 = tf.nn.leaky_relu(tf.matmul(decode_h3, deta_w[1]))
        self.output_x = tf.nn.sigmoid(tf.matmul(decode_h4, deta_w[0]))

        # cnn卷积层获取信息
        cnn_h = tf.reshape(self.input_x, [-1, 32, 32, 3])
        w_conv1 = get_variable('xavier', name='w_conv1', shape=[5, 5, 3, 32])
        b_conv1 = get_variable('zero', name='b_conv1', shape=[32])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(cnn_h, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w_conv2 = get_variable('xavier', name='w_conv2', shape=[5, 5, 32, 64])
        b_conv2 = get_variable('zero', name='b_conv2', shape=[64])
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        cnn_h = tf.reshape(h_pool2, shape=[-1, 8 * 8 * 64])

        # classifier 将每一层的embedding和CNN卷积信息做一个组合后过深度神经网络分类
        h = tf.concat([cnn_h, encode_h0, encode_h1, encode_h2, encode_h3, encode_h4, encode_h5], axis=-1)
        classifier_w = [None for i in range(6)]
        classifier_b = [None for i in range(6)]
        classifier_nn = [6096, 2048, 1024, 256, 64, 10]
        # classifier_nn = [2128, 1024, 256, 64, 10]
        for i in range(len(classifier_nn) - 1):
            if i == len(classifier_nn) - 2:
                self.hidden_features = h
            classifier_w[i] = get_variable('xavier', name='classifier_w_%d' % i,
                                           shape=[classifier_nn[i], classifier_nn[i + 1]])
            classifier_b[i] = get_variable('zero', name='classifier_b_%d' % i, shape=[classifier_nn[i + 1]])
            h = tf.matmul(h, classifier_w[i]) + classifier_b[i]
            if i < len(classifier_nn) - 2:
                h = tf.nn.leaky_relu(h)
        self.output_y = h

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('loss'):
                self.loss1 = tf.losses.mean_squared_error(labels=self.input_x, predictions=self.output_x)
                self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8).minimize(self.loss1)
                self.loss2 = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=self.output_y)
                self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8).minimize(self.loss2)

        correct_prediction = tf.equal(tf.argmax(self.output_y, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        



if __name__ == "__main__":
    Lmser()
