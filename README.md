这是类脑智能课程的大作业：基于CNN对Lmser的改进，我们这个附的是在cifar10数据集下的代码，mnist数据集和svhn数据集只需要修改一下数据的读取方式和神经网络的深度和一些参数即可。

## Team members
120033910009 黄泽人 \
120033910031 朱晨旭 \
120034910065 周扬 


## Requirements
* tensorflow
* numpy

## About codes
* main.py 包括数据读取和训练模型的部分
* model.py 包括Lmser和Lmser_CNN两个模型类
* util.py 包括tensorflow数据初始化的函数
* tsne.py 通过tsne查看神经网络分类的结果 

## Usage
* 需要先下载数据集 （http://www.cs.toronto.edu/~kriz/cifar.html） 并放于指定的文件夹下。
* 运行main.py就开始训练

## Experimental result
* 我们给出在cifar10、mnist、svhn三个数据集的结果。

模型 | MNIST |  CIFAR10| SVHN  
-|-|-|-
Lmser | 0.9849 | 0.5393|0.8606
Lmser-CNN | 0.9941 | 0.7254|0.9561 

