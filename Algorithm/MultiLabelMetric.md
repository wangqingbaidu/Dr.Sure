# 多标签评价指标
评价包括两大类，一个是位置无关的指标：`Precision`，`Recall`,`F-score`，以及位置相关的指标，包括`MAP`,`NDCG`,`ERR`。后者常见于信息检索的评价系统中，[详见](IR_Metric.md)。

多标签的评价标准中，使用最多的就是`Precision`，`Recall`,`F-score`, `MAP@n`。

[C-P,C-R,C-F1 and O-P, O-R, O-F1 and MAP@3](../../utils/Metrics.py) is available in CangJe.

## 一、位置无关的指标

### 1. Precision
准确率，指的是检索结果集合中，真正符合Query的结果的个数，除以检索结果的个数。

```math
Precision=\frac{|S\_{relevant} \cap S\_{retrieval}|}{|S\_{retrieval}|}
```


### 2. Recall
召回率，指的是检索结果集合中，真正符合Query的结果的个数，除以全部相关结果的个数。

```math
Precision=\frac{|S_{relevant} \cap S_{retrieval}|}{|S_{relevant}|}
```


### 3. Fn-score
`F值`是同时考虑到了准确率和召回率，相当于是准确率和召回率的一种加权。

```math
Fn=\frac{(1+\beta^2) \times (precision \times recall)}{\beta^2 \times (presicion + recall)}
```


β的值既是n，一般情况下使用的是`F1`值，即准确率和召回率同等重要。

```math
Fn=\frac{2 \times (precision \times recall)}{presicion + recall}
```

### 4. 类别平均准召，样本平均准召
类别平均准召，样本平均准召是衡量多标签系统的常用评价指标，前者指的是以类别为单位，衡量系统在每个类别上的准确率和召回率；后者指的是以样本为代为，衡量系统在每个样本上的准确率和召回率。

#### C-P, C-R
```math
CP=\frac{1}{C}\sum_{i \in C}\frac{N_i^c}{N_i^p}
```

```math
OP=\frac{1}{C}\sum_{i \in C}\frac{N_i^c}{N_i^g}
```
其中`N^c`代表预测成类别`c`并且Groundtruth也是`c`;`N^p`代表预测成类别`c`的个数;`N^g`Groundtruth是`c`的个数。

#### O-P, O-R
```math
CP=\frac{1}{N}\sum_{s \in N}\frac{N_s^c}{N_s^p}
```

```math
OP=\frac{1}{N}\sum_{s \in N}\frac{N_s^c}{N_s^g}
```

其中`N^c`代表预测成类别`c`并且Groundtruth也是`c`;`N^p`代表预测类别的个数;`N^g`Groundtruth的个数。

## 二、位置相关的指标
位置相关的评价指标，指的是对检索结果按照列表进行评价，不能忽略掉返回的顺序。这类指标是把多标签的每个标签想象成有顺序的，对系统进行评价。

### 1. MAP
准确率和召回率都只能衡量检索性能的一个方面，大多数情况下用户其实很关心搜索结果的排序。最理想的情况肯定是准确率和召回率都比较高。当我们想提高召回率的时候，肯定会影响准确率。所以可以把准确率看成是召回率的一种函数，Precision=f(Recall)，在R上进行积分，可以求P的期望均值。公式如下： 

```math
AveP=\int_0^1 P(r)dr  =\sum\_{k=1}^n P(k)  \Delta(k) =\frac{\sum\_{k=1}^n (P(k)\times rel(k))}{|S\_{relevant}|}
```


其中`rel(k)`表示第k个文档是否相关，若相关则为1，否则为0，`P(k)`表示前k个文档的准确率。 AveP的计算方式可以简单的认为是： 

```math
AveP=\frac{1}{R}\times\sum_{r=1}^R \frac{r}{position(r)}
```


`position(r)`为返回结果列表中的位置，例如一个返回列表，长度为10， 只有1，2，5是相关的结果，则:

```math
AveP=\frac{1}{10} \times (\frac{1}{1} + \frac{2}{2} + \frac{3}{5}) = 0.26
```
