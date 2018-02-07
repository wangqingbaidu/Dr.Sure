## 分类任务（网络结构）发展的脉络：

### 一、Accuracy

1. [LeNet](http://deeplearning.net/tutorial/lenet.html)，深入学习大佬LeCun的CNN奠基之作。这个网络的论文我没看过，好像是90年代的工作了，年代太久远，不过现在看估计也没什么了。

2. [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)，卷积神经网络首次在大规模图像识别中崭露头角。

3. [VGG](https://arxiv.org/pdf/1409.1556.pdf)，首次引入小核，深层的网络结构。感受野是VGG最本质上的贡献。

4. [GoogleNet(Inception-v1)](https://arxiv.org/pdf/1409.4842)，14年的冠军模型，引入Inception块，根本的也是在不同的层数解决多感受野的问题。

5. [ResNet](https://arxiv.org/pdf/1512.03385)，残差网络，使用更深的网络图像特征进行提取，这个网络结构是后面很多视觉应用里面基本的特征提取工具。个人感觉这个网络模型一定带有一点根据不同数据自适应网络结构的意思。

6. [NasNet](https://arxiv.org/pdf/1707.07012)，从人工定义的网络结构到网络结构也是机器学习出来的重要跨越，这个思想与游戏AI的设计思路是一样的，可以[参考这篇论文](https://arxiv.org/pdf/1611.01578)理解Nas。

上面的这几个工作是在网络结构设计里面比较有代表性的工作，除了第6个，其余的都应该是深度学习的CV从业者必备的只是。当然在网络结构发展的过程中也存在着很多的Tricks，想[BatchNorm](https://arxiv.org/pdf/1502.03167)，Dropout这些基本上已经被验证屡试不爽的一些Trick，还有很多泛华能力不强的就不再一一列举。想更深入的了解，可以参考一下这个[文献](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)。

### 二、Performance
到Resnet，学术界的分类的准确率就已经刷的够高了，Top-5的准确率有97%左右了（人的识别准确率是95%左右），所以就要向工业界应用。但是在工业应用的时候，最大的问题就是速度，尤其是移动设备或者嵌入式设备，后面的两个工作就是从运行速度上进行改进。

1. [MobileNet](https://arxiv.org/pdf/1704.04861)，第一个端到端的模型解决速度问题，理论差不多有一个量级左右的提升（实测没有）。

2. [ShuffleNet](https://arxiv.org/pdf/1707.01083)，MobileNet虽然快，但是准确率损失还是挺明显的，ShuffleNet就是进一步提升准确率，但是不损失处理效率。

### 三、Conclusion
前面说的这些模型，相关的代码以及所有的预训练好的参数都已经开源，这个是Tensorflow框架下的[代码](https://github.com/tensorflow/models/tree/master/research/slim)。目前这些代码都已经放到了Tensorflow的开源包里面，可以不用修改任何代码，直接使用。

1. 分类是计算机视觉的基础，前面的几种分类模型，都在后面不同视觉任务上有或多或少的应用。
2. 核心概念——`感受野`，几乎所有模型的改进都围绕着这个概念再进行。
3. 模型从手工设计到程序自动生成，这个是增强学习带来的另外一个突破。

## 目标检测发展脉络
目标检测(Object detection)是在分类任务上的一个发展。分类任务要解决的问题是图像是什么，而目标检测不仅仅要回答图像是什么，还要回答目标的位置。

### 一、Two-Stage 模型
两阶段模型指的是将Region Proposal与分类回归区分开，分成两个阶段去解决目标检测问题。

1. [R-CNN](https://arxiv.org/pdf/1311.2524)，这个工作是使用深度学习做目标检测任务的开山之作，基本上奠定了后面所有的`XXX R-CNN`的基调，当然它也留下了一些问题：
	
	>1. Selective Search，效率太低，产生的proposal过多，严重影响处理的效率。
	>2. Proposal尺度不一的问题没有解决，在使用CNN提取特征时候，Object会出现变形。
	>3. N个分类以及回归器，不能端到端地训练模型。
	
2. [SPPNet](https://arxiv.org/pdf/1406.4729)，这个工作解决了R-CNN中的Proposal尺度不一致的情况。任意尺度图像的输入，不需要在开始的时候进行Resize，只需要在最后一层的featuremap上使用SPP层（特征图N等分），即可解决任意尺寸输入的问题。

3. [Fast R-CNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)，这个工作旨在解决分类器与边框回归器融入到CNN网络端到端训练的问题。它借鉴了SPPNet的思想，由于每一个边框在最后特征图上都对应一块区域，这时候对对应的区域进行RoI的Pooling，使用两层的DNN，就可以端到端地训练这个模型了。

4. [Faster R-CNN](http://www.cvlibs.net/projects/autonomous_vision_survey/literature/Ren2015NIPS.pdf)，R-CNN的遗留问题，前面两个工作解决了2/3，还有一个就是Region Proposal（区域提名）的问题，Faster R-CNN的提出就是通过引入一个RPN网络（Region Proposal Network）。他的出发点也很简单，就是既然网络可以分类和回归了，是不是也可以把区域提名也融入到网络，进行端到端学习。

两阶段的目标检测模型在后面有诸多的发展，但是R-CNN留下的这几个问题，基本上已经在后续的3篇工作上得以解决，后面的工作基本上也就是围绕着感受野的问题，进行若干的修修补补，其中比较有代表性的工作像，[R-FCN](https://arxiv.org/pdf/1605.06409)，以及17年ICCV的BestPaper：[Mask R-CNN](https://arxiv.org/pdf/1703.06870)。

### 二、End to End 模型
前面的两阶段模型最大的优势就是准，虽然越来越快，但是速度还是无法达到实时（30fps）的要求，对于一些特别关注速度的应用场景，前面的模型处理起来就显得捉襟见肘，所以就需要一个更快的结构，虽然可能准确率没有那么好。

1. [YOLO](https://pjreddie.com/media/files/papers/yolo.pdf)，这个是目标检测里面End2End模型的鼻祖，他的思想其实相对简单，就是要在最后面，让每一个神经元都要学习一个分类和一个回归，由于是一个端到端的模型，所以它的速度相当快。另外有一点需要说明的就是，这个文章的作者同时开发了一个基于纯C语言的深度学习框架，[Darknet](https://pjreddie.com/darknet/)，它的平台迁移能力简直无与伦比。还有多说已经就是，[XNORNet](https://arxiv.org/pdf/1603.05279)也是他们的工作，这个工作通过将权重量化到bit，实现了60X左右的加速效果（虽然证明有时模型难以收敛）。

2. [SSD](https://arxiv.org/pdf/1512.02325)，这个实在Yolo基础上的改进，主要解决的就是Yolo虽然速度提升，但是准确率下降的事情，思想与Yolo类似，唯一不同还是感受野的问题，不像Yolo只使用最后一层进行分类和回归，SSD把前面几乎所有的层都输出到后面进行一个分类和回归预测。

### 三、Conclusion
目标检测任务是很多应用的基础，像OCR，Caption等都需要Detection模型提供更加有效的特征。目标检测的相关代码也已经开源，[代码](https://github.com/tensorflow/models/tree/master/research/object_detection)同样是基于Tensorflow的。

1. 两阶段模型解决的是精度，end2end模型解决的是速度问题。
2. 每一个神经元都赋予其具体的作用，这个是end2end模型的精髓。
3. 无论是分类还是检测，感受野是永远无法绕开的话题。

 
