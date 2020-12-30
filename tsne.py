import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
# 通过tsne画分类效果图
def plot_embedding(data, label):
    fig = plt.figure()
    #ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []
    type11_x = []
    type11_y = []

    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if label[i] == 4:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        if label[i] == 5:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])
        if label[i] == 6:
            type7_x.append(data[i][0])
            type7_y.append(data[i][1])
        if label[i] == 7:
            type8_x.append(data[i][0])
            type8_y.append(data[i][1])
        if label[i] == 8:
            type9_x.append(data[i][0])
            type9_y.append(data[i][1])
        if label[i] == 9:
            type10_x.append(data[i][0])
            type10_y.append(data[i][1])
    color = plt.cm.Set3(0)
    color = np.array(color).reshape(1, 4)
    color1 = plt.cm.Set3(1)
    color1 = np.array(color1).reshape(1, 4)
    color2 = plt.cm.Set3(2)
    color2 = np.array(color2).reshape(1, 4)
    color3 = plt.cm.Set3(3)
    color3 = np.array(color3).reshape(1, 4)
    type1 = plt.scatter(type1_x, type1_y, s=5, c='r')
    type2 = plt.scatter(type2_x, type2_y, s=5, c='g')
    type3 = plt.scatter(type3_x, type3_y, s=5, c='b')
    type4 = plt.scatter(type4_x, type4_y, s=5, c='k')
    type5 = plt.scatter(type5_x, type5_y, s=5, c='c')
    type6 = plt.scatter(type6_x, type6_y, s=5, c='m')
    type7 = plt.scatter(type7_x, type7_y, s=5, c='y')
    type8 = plt.scatter(type8_x, type8_y, s=5, c=color)
    type9 = plt.scatter(type9_x, type9_y, s=5, c=color1)
    type10 = plt.scatter(type10_x, type10_y, s=5, c=color2)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return fig


labels = np.load('label.npy')
features = np.load('hidden_features_0.npy')
print("begin")
labels_number = np.argmax(labels, axis=1)
# 将高维信息通过pca转化为二维信息
tsne = TSNE(n_components=2, init='pca', random_state=0)
x = tsne.fit_transform(features)
plot_embedding(x, labels_number)

