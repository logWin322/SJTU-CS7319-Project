import tensorflow as tf
import numpy as np
from model import Lmser, Lmser_CNN
from scipy.io import loadmat as load
import os
import time
import pickle
import random

# 读取数据集文件
def load(file_name):
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        return data

# 提取数据集中的训练集
def get_train():
    data1 = load('datasets/data_batch_1')
    x1 = np.array(data1[b'data'])
    x1 = x1.reshape(-1, 3, 32, 32)
    y1 = np.array(data1[b'labels'])
    data2 = load('datasets/data_batch_2')
    x2 = np.array(data2[b'data'])
    x2 = x2.reshape(-1, 3, 32, 32)
    y2 = np.array(data2[b'labels'])
    train_data = np.r_[x1, x2]
    train_labels = np.r_[y1, y2]
    data3 = load('datasets/data_batch_3')
    x3 = np.array(data3[b'data'])
    x3 = x3.reshape(-1, 3, 32, 32)
    y3 = data3[b'labels']
    train_data = np.r_[train_data, x3]
    train_labels = np.r_[train_labels, y3]
    data4 = load('datasets/data_batch_4')
    x4 = np.array(data4[b'data'])
    x4 = x4.reshape(-1, 3, 32, 32)
    y4 = data4[b'labels']
    train_data = np.r_[train_data, x4]
    train_labels = np.r_[train_labels, y4]
    data5 = load('datasets/data_batch_5')
    x5 = np.array(data5[b'data'])
    x5 = x5.reshape(-1, 3, 32, 32)
    y5 = data5[b'labels']
    train_data = np.r_[train_data, x5]
    train_labels = np.r_[train_labels, y5]
    train_data = train_data.transpose(0, 2, 3, 1).astype("float")
    print("train")
    print(train_data.shape)
    print(train_labels.shape)
    return list(train_data), list(train_labels)

# 提取数据集中的训练集
def get_test():
    data1 = load('datasets/test_batch')
    x = np.array(data1[b'data'])
    x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    y = data1[b'labels']
    print("test")
    print(x.shape)
    return list(x), list(y)

# 改变原始数据的形状
# (图片高，图片宽，通道数，图片数)->(图片数,图片高，图片宽，通道数)
# labels 变成one-hot encoding
# 做归一化
def reformat(samples, labels):
    samples = np.array(samples)
    scalar = 1 / 255.
    samples = samples * scalar
    labels = np.array(labels)
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return samples, labels


batch_size = 256
# 训练函数
class Trainer:
    def __init__(self,):
        # self.model = Lmser()  Lmser模型
        self.model = Lmser_CNN()  # Lmser_CNN模型
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        train_x, train_y = get_train()
        test_x, test_y = get_test()
        self.train_x, self.train_y = reformat(train_x, train_y)
        self.test_x, self.test_y = reformat(test_x, test_y)
        self.train_set_size = len(self.train_y)
        self.test_set_size = len(self.test_y)
        print("train shape:", self.train_set_size)
        print('test shape:', self.test_set_size)

    def next_batch(self, batch_size, batch_idx):  
        # batch地读取数据
        batch_x = self.train_x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_y = self.train_y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_x = np.reshape(batch_x, (batch_size, -1))
        return batch_x, batch_y

    def evaluate(self, ):
        # 评价模型
        test_x = np.reshape(self.test_x, (self.test_set_size, -1))
        # accuracy = self.sess.run([self.model.accuracy], {self.model.input_x:test_x, self.model.input_y: self.test_y})
        accuracy, hidden_features = self.sess.run([self.model.accuracy, self.model.hidden_features],
                                                  {self.model.input_x: test_x, self.model.input_y: self.test_y})
        return accuracy, hidden_features

    def shuffle(self):
        # 打乱数据集，使得每个epoch训练集数据顺序不同
        print("shuffle")
        a = [i for i in range(self.train_set_size)]
        random_a = random.shuffle(a)
        self.train_x = self.train_x[random_a][0]
        self.train_y = self.train_y[random_a][0]
        print(self.train_x.shape)
        return

    def train(self, ):
        # 训练函数
        learning_rate = 1e-3
        print("begin")
        total_loss1_list = []
        total_loss2_list = []
        start_time = time.time()
        best_acc = 0.0
        for epoch in range(50):
            self.shuffle()
            n_batch = int(self.train_set_size / batch_size)
            loss_list1 = []
            loss_list2 = []
            for i in range(n_batch):
                batch_x, batch_y = self.next_batch(batch_size, i)
                _, loss1, _, loss2 = self.sess.run(
                    [self.model.optimizer1, self.model.loss1, self.model.optimizer2, self.model.loss2],
                    {self.model.input_x: batch_x, self.model.input_y: batch_y, self.model.lr: learning_rate})
                loss_list1.append(loss1)
                loss_list2.append(loss2)
                total_loss1_list.append(loss1)
                total_loss2_list.append(loss2)
                if i % 10 == 0:
                    now_time = time.time()
                    print("epoch:", epoch, "batch:", i, "time:", now_time-start_time)
            accuracy, hidden_features = self.evaluate()
            # if epoch % 5 == 0:
            #     np.save('hidden_features_%d.npy' % epoch, hidden_features)
            if accuracy>best_acc:
                best_acc=accuracy
            print("epoch:", epoch, "Accuracy:", accuracy, "Loss1:", sum(loss_list1)/len(loss_list1), "Loss2:", sum(loss_list2)/len(loss_list2), "now_lr:", learning_rate)
            learning_rate *= 0.95
        # np.save('loss1_256_lmser_6.npy', total_loss1_list)
        # np.save('loss2_256_lmser_6.npy', total_loss2_list)
        print("best accuracy:", best_acc)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()