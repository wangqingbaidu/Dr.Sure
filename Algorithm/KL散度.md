# KL散度
相对熵(relative entropy)又称为KL散度(Kullback–Leibler divergence，简称KLD),信息散度(information divergence),信息增益(information gain)。

KL散度是两个概率分布P和Q差别的非对称性的度量。

KL散度是用来度量使用基于Q的编码来编码来自P的样本平均所需的额外的比特个数。 典型情况下，P表示数据的真实分布，Q表示数据的理论分布，模型分布，或P的近似分布。

熵的定义：

```math
H(x)=\sum_{x \in X}P(x)log(\frac{1}{P(x)})
```

KL散度用来衡量两个分布之间的距离：

```math
D_{KL}(Q|P)=\sum_{x\in X}Q(x)log(\frac{1}{P(x)})-\sum_{x\in X}Q(x)log(\frac{1}{Q(x)})
```

## 性质
1. 不对称性，KL散度可以看成是一种距离度量的方式，所以她并不具备对称性，即`D{Q|P}≠D{P|Q}`。
2. 非负性。即KL散度的值一定是非负的。

	#### 证明一：
	
	>利用对数和不等式或者延森不等式
	
	>```math
	D(Q|P)=\sum_{x\in X}Q(x)log(\frac{1}{P(x)})-\sum_{x\in X}Q(x)log(\frac{1}{Q(x)})=-\sum_{x\in X}Q(x)log(\frac{P(x)}{Q(x)})
	```
	
	> ```math
	D(Q|P)=-E(log\frac{P(x)}{Q(x)}) \geq -logE(\frac{P(x)}{Q(x)})=-log\sum_{x \in X}\frac{Q(x)P(x)}{Q(x)}
	```
	
	>由于
	
	>```math
	\sum_{x \in X}P(x)=1
	```
	
	>所以
	
	>```math
	D(Q|P) \geq 0
	```
	
	#### 证明二：
	
	>已知
	
	>```math
	ln(x) \leq x - 1 \quad if  \quad x \leq 1
	```
	
	>```math
	D(Q|P)=-\sum_{x\in X}Q(x)log(\frac{P(x)}{Q(x)}) \geq -\sum_{x\in X}Q(x)(\frac{P(x)}{Q(x)} - 1)=0
	```
	>tips: 注意负号。

## 应用
1. 神经网络中，存在多个并行的网络，而最后又希望并行网络输出的结果再一个相同或者相近的分布事，使用KL散度作为一个监督信息。
2. 不同的信息源在映射到一个相同的语义空间的时候，可以引入KL散度，用来度量这两个映射的空间处于一个相同的分布。

