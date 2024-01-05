* Mdnotes File Name: [[@ZhouJiQiXueXi]]

---
title:机器学习概论
date: 2023-08-22 20:14:39
obsidianUIMode: source
---
# 评估 
A few Machine Learning terms:
Sample: the instances of the sample
Attribute space 
Sample space 
Feature vector

P问题：算法的时间复杂度为关于数据规模的多项式
NP问题：算法的时间复杂度可能为一个关于数据规模的多项式
约化：从一个较为简单的NP问题等效到更为复杂的NP问题
NPC问题：特指一个算法，首先此算法为NP问题，并且此算法是此类算法中最复杂的（NP complete）
NP-Hard：满足NPC问题的第二个条件但不用满足他是NP问题 

一个关于过拟合的有趣观点：只要相信P不等于NP，即当模型的复杂程度超过了经验误差最小，过拟合就无法避免
## 评估方法 
### 留一法
通过采样将样本分为test data and sample data. 
单次使用留出法得到的估计结果往往不够稳定可靠，在 使用留出法时，一般要采用若干次随机划分、重复进行实验评估后取平均值作 为留出法的评估结果.
### 交叉验证法
通过采样将样本划分为k个子集，每次将k-1个子集作为训练集，剩下的一个作为测试集。重复p次。



# 线性回归

# 决策树

# 神经网络

# 支持向量机
## SVM Model
考虑一个线性二分类问题
![[Pasted image 20230825140408.png]]
如果分类正确，那么对于两类样本点满足如下性质
$$
w^Tx_{i}+b\geq+1,y_{i}=+1 \\
$$
$$
w^Tx_{i}+b\leq-1,y_{i}=-1 \\
$$
因此分类问题转化为使$\gamma =\frac{2}{w}$（两类的最近距离，等价为$\frac{1}{2}w^2$被称为支持向量）最大
### 对偶问题
支持向量的优化其实为一个二次规划问题。可以通过拉格朗日得到其对偶问题
![[Pasted image 20230825143630.png]]
因满足KKT，采用SMO(Sequential Minimum Optimization)算法。
#SMO
SMO核心在于固定除$\alpha$外的其他变量，使用两个$\alpha_{i}$$\alpha_{j}$,通过类似MCMC的方法找到最优解（找$\alpha_{i}-\alpha_{j}$的最大值来最快速优化）
## Kernel Function
Purpose: During the classification, we can add the dimension of the dataset to find a hyperplane to classfy two clusters. As in the calculation of the supporting vecor, the inner product of $w$ and $w^T$ is involved. The idea is that we do not need to get the exact mapping from data space the veign space. We only need to maintain the inner product is the same in the above spaces. 
There is a theorem that the a function is the kernel  if and only if the kernel matrix (function of all the data) is 半正定的.
There are several kernel functions.
![[Pasted image 20230825145952.png]]
核函数经过线性变换、与其他核函数直积后也为核函数的性质。
## 软间隔与正则化
软间隔指对与不好划分的任务采用一个损失函数使得原式变为：
![[Pasted image 20230825153622.png]]
有多种损失函数，最后通过拉格朗日得到$\alpha$与损失函数无关
一般的可以将支持向量机写成结构风险＋经验风险
![[Pasted image 20230825154847.png]]
经验风险可以认为是降低了过拟合风险，可以理解为正则化，正则化可以理解为惩罚函数法对不希望的结果经行惩罚，从贝叶斯的角度可以理解为先验概率
## Supporting Vector Regression
**I dont fully understand why SVR can be generalized into a SVM problem:**

![[Pasted image 20230826114932.png]]
Different from oringinal linear regression problem, SVR specifies tolerance.$\epsilon$ that regard the regression as correct.
![[Pasted image 20230826163340.png]]
同时直线（平面）两边的点的的容忍度会不同，因此加入松弛变量
![[Pasted image 20230826163351.png]]

## 核方法 
#Kernel
表示定理：对于如下优化问题，最优解总可以写为核函数的线性组合
![[Pasted image 20230826165008.png]]
（h表示对应训练样本的希尔伯特空间）
**因此可以将线性问题转化为非线性问题**（我们为什么要这样做）
例如LDA聚类转化为核线性判别
![[Pasted image 20230826165225.png]]
# 贝叶斯分类

# 集成学习

# 聚类

# 降维与量度学习

# 特征选择与稀疏学习
稀疏学习：从数据属性中挑选学习到对学习结果影响最大的属性。
一般来说包括子集搜索阶段和评估阶段，将两者相结合即为特征选择方法：过滤式、包裹式以及嵌入式。
## 过滤式
-过滤式的特点在于与学习器分步进行
Relief特征选择：
首先对于某数据$x_{i}$寻找同类最近样本和异类最近样本$x_{i,nh},x_{i,nm}$(near hit & near miss)，得以计入对某属性j的统计分量：
![[Pasted image 20230826200032.png]]
上式说明的是若属性j使得$x_{i}$与同类的距离大于异类的距离，则增大j的统计量。
## 包裹式
-包裹式的特点在与特征选择挑选可以增强学习器性能的子集。
LVW方法：
随机选择子集 - 学习子集 - 交叉评价学习后误差
此方法训练开销大，并且可能在一定的时间内无法输出一个符合要求的结果。
蒙特利尔法则一定生成一个符合要求的结果
## 嵌入式
-学习的过程带有降维 
核心就是带有正则化的线性回归 ：
![[Pasted image 20230826201032.png]]
上式带有L2范数，也被称作岭回归
![[Pasted image 20230826201112.png]]
带L1范数，被称作LASSO(Least Absolute Shrinkage and Selection Operator).
L1范数相比与L2来说得到等到更加稀疏的学习分量$w$
具体可参考如下等值线图，(只考虑了二维样本)
![[Pasted image 20230826201326.png]]
其中范数等值线与平方误差等值线的交点才为上式的解，因此L1范数中$w$至少有一个为0.
因此此学习过程即是对样本的稀疏学习，上式求解可采用近端梯度下降法（Proximal Gradient Descent）.
#PGD
## 字典学习 
考虑实际运用中的一个字典矩阵，每行代表档案，每列代表汉字，矩阵表示了一个样本（档案）标题运用到的汉字，可以发现列数远远大于行数。
字典学习其实是将字典矩阵表现得更易于学习：
![[Pasted image 20230826202101.png]]
上式B为字典矩阵，$\alpha_{i}$指稀疏度（让B变得稀疏的矩阵）。上式前一项表示与数据得的差距要小，第二项表示要稀疏。
此式解法为变量交替优化，即单独优化B和α
## 压缩感知
在信号分析的复原目标信号时，压缩感知可以通过考虑稀疏性来最大化复原信号。
![[Pasted image 20230826202634.png]]
上式即为对目标信号y的复原，A类似于字典矩阵，s指稀疏向量 类似于α。
压缩感知包括感知测量和重构恢复，
列举两种方法：
1.Restricted Lsometry Property
#RIP
![[Pasted image 20230826202928.png]]
若A满足上式，则特征选择问题可以转化为：
![[Pasted image 20230826203204.png]]
2.矩阵补全技术
将矩阵A中不存在的元素给补全 

# 计算学习理论
## 几个重要的概念
经验误差 （Emperical Error）: 在训练集中学习器的输出与真实输出的差距 
![[Pasted image 20230830111155.png]]
泛化误差 （Generalized Error）:训练后的学习器在泛化表现出的误差 
![[Pasted image 20230830111309.png]]

假设空间$H$:
包含所有可能的从x到y的映射$h$
概念 $c$
表示从x到y的映射

**机器学习的实质为通过优化算法$L$从假设空间$H$发现假设$C$使其满足分类或学习的要求**

Hoeffding不等式：
对于m个独立同分布随机采样的样本：
![[Pasted image 20230830111335.png]]
$\epsilon$表示泛化误差的上限
由于独立同分布随机采样，经验误差的期望等于泛化误差，可以推出
![[Pasted image 20230830111814.png]]

## PAC

 ### PAC的几个概念
 PAC 辨识
 表示存在算法$L$使得泛化误差小于$\epsilon$的概率大于$1-\sigma$(置信区间)
 ![[Pasted image 20230830112453.png]]
 PAC可学习
 若存在算法 L 使得：
 $$
m\geq poly\left( \frac{1}{\epsilon}, \frac{1}{\sigma} ,size(x),size(c)\right)
$$
则称c 为可学习 
PAC可学习其实表示当样本量m足够大时，算法可以从H中辨识c。
满足以上条件的最小样本称为样本复杂度(Sample Complexity)

同样的对计算时间也有类似的定义：
PAC高效可学习：
若存在算法L使其运算时间满足
 $$
t\geq poly\left( \frac{1}{\epsilon}, \frac{1}{\sigma} ,size(x),size(c)\right)
$$
### 可分情况
有限空间的可分情况都是PAC可学习的
证明略

### 不可分情况
![[Pasted image 20230830114124.png]]
不可知PAC学习：
![[Pasted image 20230830114142.png]]
对于概念不属于假设空间的情况时，可以通过找到一个使泛化误差最小的算法使得上式成立

## VC维
假设空间的VC维是样本能被H打散的最大实例集大小
![[Pasted image 20230830114443.png]]
VC维与样本分布无关，但与假设空间的复杂度有很大关系

# 半监督学习
样本中有一堆未标记的数据和有标记的数据的学习就叫半监督学习，为被标记的数据通过一些关于数据与数据之间分布的相似性的假设来联系

## 生成式
生成式假设为所有样本均由同一个分布式产生。
假设样本的每个类都对应一个高斯分布 那么样本的概率密度函数可以表示为：
![[Pasted image 20230831202819.png]]
$\alpha$表示第i类的混合系数，p表示样本属于第i类的概率
假设$f(x)$表示对x检测出来的标记 则最大化后验概率来取j（标记）
$$
f(x)= arg_{j} max \sum_{i} p(y=j,\theta=i|x)
	= arg_{j} max \sum_{i} p(y=j|\theta=i,x)p(\theta=i|x)
$$
其中 
![[Pasted image 20230831204046.png]]
represents the posibility that x belongs to the class $i$.
The good thing is that the first term of the $f(x)$ is related to the dataset with labels and the second term is not. Therefore, the likelihood can be generated:
$$
\mathcal{L}(D_{l}\cap D_{u})= \sum_{D_{l}} \ln(p(y=j|\theta=i,x_{i})p(\theta=i|x_{i}) + \sum_{D_{u}} \ln(p(y=j,\theta=i|x_{i})
$$
The first term is the dataset with labels, the second is not.
Therefore, we can use the EM method to converge the function:
E-step: Use the classification on the dataset to label them.
M-step: Adjust the parameters like $\mu$ and $\sum$.
These two steps can be iterated to converge.
The distributuion can be changed into othters like the Bayesion.
The drawback is the model highly relies on the assumption.

## 半监督SVM 
The Half Supervised SVM utilizes the SVM to categorize the dataset to decide the labe of the unlabeled dataset. One way to do this is through randomly set the label of the unlabeled and calculate the SVM plane to classfy.
However, this algorithm consumes a lot of computation resources. A way to optimize  is through getting the SVM firstly on the labeled one and use it to label the unlabeled, which refers to Pseudo-label. For a set of pseudo-label, if they have different labels and their **松弛变量 added together is greater than 2,** they are most likely to be wrong.(Why??) So change their labels and run SVM again.
If the number of class is greatly uneven, which is very likely to occur, we can compulsorily preset the number of class.

## 半监督图模型
图模型将样本当成结点，样本间的相似度当成边。而标记的样本就是颜色，未标记的就是无颜色，训练的过程就是图的颜色传播过程 
### 二分类问题

## 分歧方法 

# 强化学习 
强化学习的实质为通过一系列动作后导致的环境变化得到的奖励来得到一个策略。
因此主要有四个参数：
X： 环境变量  ps. 环境在不同领域有不同的指向[[环境变量X]]
a： 动作
p：某一动作导致环境变化到新的环境的概率
R：某一环境改变得到的奖励

最后得到的策略$\pi$ $a=\pi(x)$（表示在特定的状态应该做什么动作使得总reward最大）
相比于监督学习有相似的地方也有不同的地方。
这个策略从动作到环境类似于监督学习中的x到y的映射;
不同的是强化学习中没有示例（样本）和标签（y）的说法，而只能通过a来得到x.
强化学习可以看作是延迟奖赏的标记（经过一系列a得到x后通过r标记此Markovchain可以近似看作监督学习中的样本）

## 探索-利用窘境
探索：得到每一个动作后X对应得R
利用：通过进行R最大的动作a来最大化奖赏

窘境：在有限的探索次数（运算次数中）来得到最大化的奖赏，探索与利用相冲突

## K-摇臂赌博机
存在K个摇臂（近似于动作），每个摇臂后有个奖励reward。
### $\epsilon$贪心
以一定概率$\epsilon$进行探索，$1-\epsilon$的概率来进行利用。
探索：对某一摇臂k（动作）的奖励函数$Q(k)$更新
![[Pasted image 20230831163101.png]]
n表示此摇臂进行的次数，$v_{n}$表示此次得到的奖励，上式是对n次尝试摇臂k得到的奖励的平均的增量式表示。

对$\epsilon$的选取
1. 当奖励不确定较大，$\epsilon$可选的较大 
2. 当奖励的不确定较小，$\epsilon$选取小的值
3. 当探索完所有的奖励后不再需要探索，因此$\epsilon$可以随探索次数降低

### Softmax算法 
Softmax通过Boltzman函数来选择摇臂k（把探索和利用看成同一件事，通过概率来选择）
每个摇臂选择概率为：
![[Pasted image 20230830151237.png]]
$\tau$越小则被选取的概率越大，$\tau$趋近于0时则只利用。

## 有模型的学习 
对 $E=<X,A,P,R>$有一定估计的模型
### 评估-优化的迭代 
值函数： $V^\pi_{T}(x)$,表示通过策略$\pi$从状态$x$经过$T$步得到的奖励（$\gamma$折扣奖励与T步扣奖励）
![[Pasted image 20230830155547.png]]、
由于具有马尔可夫性质（即下一状态仅由当前状态决定），可以通过全概率展开写成迭代式（$\gamma$折扣奖励）
![[Pasted image 20230830155809.png]]
$\gamma$折扣奖励由于不像T步折扣有个步数限制，可能会迭代很多次，因此可以设置一个限制当值函数更新的量小于一定阈值则停止
![[Pasted image 20230830160258.png]]
**（为什么上式的sum是对全体A中的a而不是pi中的a
?）** 因为一般来说动作a对所有状态 都适用，例如西瓜浇水和不交水
通过值函数可以得到状态-动作函数
![[Pasted image 20230830160108.png]]
状态&动作:同一动作可能会导致不同的状态，值函数只表示了从x开始通过策略$\pi$得到的reward
It is vital to emphersize that the T-step algorithm is based on accumulation of previous policy reward with a weight of '1/t'. The accumulation lasts T in total.

通过状态-动作函数选取reward最大的策略 （一系列的动作）
然后通过此策略计算状态-动作函数，并于上一种策略比较，若reward增大则将上一策略替换成此策略，直到出现两个策略的reward相等的情况**（为什么这就是最优的？）**
因为这个最优化的过程是基于对每一点x对a进行改进的。这种优化是动态的，没进行一步策略就对值函数更新。

## 免模型的学习 
在真实情况中各个参数常常是不确定的，尤其是策略和reward甚至可能连动作的数量都是未知的，因此往往要通过不断迭代探索reward来确定最优策略。
### 蒙特卡洛强化学习
通过蒙特卡洛采样来生成一系列的动作和状态，最后通过Q来对策略进行评估
#MCMC 
通过采样的策略得到一条轨迹，计算相应的值函数。
优化：**由于有些时候策略都是一致的(why？)** 采用$\epsilon$贪心采样法
![[Pasted image 20230830201108.png]]
这里可以理解为通过以$\frac{\epsilon}{A}$的概率随机采用一个新的策略。
由于被评估和被采样的策略为同一策略，所以也被称作“同策略蒙特卡洛强化学习法”。
因此得到值函数的计算和更新方法如下：
![[Pasted image 20230831154326.png]]
其中于模型的RL相比这个方法更像是先生成一条动作，然后反过来一步步优化每个状态下的动作。
而异策略采样就是对确定性的策略进行优化，由于确定的策略概率为1，而同策略的策略的概率为$1-\epsilon+\frac{\epsilon}{A}$或者$\frac{\epsilon}{A}$，所以两者的reward的比值为：
![[Pasted image 20230830205053.png]]

### 时序差分学习
由于蒙特卡洛没有利用到迭代性，因此效率非常的低。根据Q的增量式表达，可以把每一次状态更新经行采样（而非先采样生成一条完整的动作状态链），得到：
![[Pasted image 20230831163313.png]]
上式表示根据前t次得到的Q，在t+1采样时得到Q(Q是关于采样次数取了平均的)
![[Pasted image 20230831163459.png]]
相当于增量主要在后面一坨，因此可以采用一个系数来代替$\frac{1}{t+1}$
这个方法被称作Sarsa
对Sarsa执行异策略算法就得到了Q学习算法 

## 值近似函数
当值函数（R缺失时），采用近似的方法对其进行估计
首先假设值函数关于状态是线性相关的，但是这显然与真实的值函数有差距，因此用最小二乘法来和梯度下降的方法来最小化此差距
![[Pasted image 20230831165024.png]]
$\theta$为线性关系中的系数
因此得到关于样本的更新规则
![[Pasted image 20230831165131.png]]
$V^\pi$表示真实的值函数
采用增量的时序差分学习来更新
![[Pasted image 20230831165300.png]]

## 模仿学习 
由于有人类的榜样案例，因此可以通过学习榜样来当成初策略然后进行更新学习 

### 逆强化学习 
当值函数未知时，可以将榜样策略所带来的最大reward的值函数来计算值函数 即
![[Pasted image 20230831171717.png]]
也就是让范例和随机迭代出最优的轨迹的差距最大的值函数线性参数
完整流程为先随机产生轨迹，通过与范例比较产生最优值函数，然后产生新的轨迹，再依次迭代.
