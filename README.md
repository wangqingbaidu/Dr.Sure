### Attention!
我的Dr.Sure项目正式上线了，主旨在分享学习Tensorflow以及DeepLearning中的一些想法。期间随时更新我的论文心得以及想法。

Github地址：[https://github.com/wangqingbaidu/Dr.Sure](https://github.com/wangqingbaidu/Dr.Sure)

CSDN地址：[http://blog.csdn.net/wangqingbaidu](http://blog.csdn.net/wangqingbaidu)

个人博客地址：[http://www.wangqingbaidu.cn/](http://www.wangqingbaidu.cn/)

# Dr.Sure

Some examples and urls are not available because of copyright. Join us, Join [Kwai](https://www.kuaishou.com/joinus.html) for details. 

文档中的一些例子和连接可能无法显示，因为这些代码被部署在了公司的内网。当然如果想进一步了解，欢迎加入我们，加入[快手](https://www.kuaishou.com/joinus.html)多媒体内容理解组

---

此目录包括2个文件夹，一个是[Algorithm](./Algorithm)，一个是[LearningTensorflow](./LearningTensorflow)。
>1. [Algorithm](./Algorithm)文件夹整理目前最新的论文分享详解以及在CangJe项目中的代码支持等。
>2. [LearningTensorflow](./LearningTensorflow)文件夹存放的是使用Tensorflow过程中的一些经验以及一些抽象出来的utils使用总结。

---

## [Algorithm](./Algorithm)
1. [Attention-based Extraction of Structured Information from Street View Imagery.md](./Algorithm/Attention-based_Extraction_of_Structured_Information_from_Street_View_Imagery.md), Tensorflow中OCR识别的的论文介绍。

2. [DSSMs: Deep Structed Semantic Models](./Algorithm/DSSMs.md) 深度语义模型，不同信息源映射到一个相同的语义空间。

3. [KL散度](./Algorithm/KL散度.md)， KL散度的一些基本知识以及应用场景，相关性质的证明。

4. [信息检索评价指标](./Algorithm/信息检索评价指标.md)，信息检索中的多种评价指标，衡量一个检索系统的好坏。

5. [分类、检测问题总结](./Algorithm/Clf_and_Detection.md)，总结了从12年到17年图像分类任务以及目标检测任务的发展脉络。

---

## [LearningTensorflow](./LearningTensorflow)
1. [TFrecord&QueueRunner.md](./LearningTensorflow/TFrecord&QueueRunner.md)，简单介绍如何针对原始数据生成TFrecord以及从TFrecord中解析出一个样本。QueueRuuner部分介绍如何将TFrecord的文件应用到计算图中。
2. [Losses.md](./LearningTensorflow/Losses.md)，Loss function相关的介绍。

3. [Optimizer.pdf](./LearningTensorflow/Optimizer.pdf)，Tensorflow中相关优化函数介绍。

---

## [code](./code)
1. [distance.py](./code/distance.py)基于Tensorflow，用于计算两个tensor的距离的代码，目前已经添加cosine距离。