---
title: Untitled
date: 2023-08-23 12:45:25
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: inprogress
destination: 
share: false
obsidianUIMode: source
---

# Probability Model
## Sample: 
- All the possible outcomes.
- The outcomes should be mutually exlusive.

## Experiment 

Subtle Points in Continuous Model:
![[Pasted image 20231125155811.png]]
之所以等式不成立，是关于点的集合为一个无限集合，那么他的概率自然也不能等于加起来的概率了
这里不该用sum集合，而应该用积分求概率的合。

## Random Variables:
对sample space里的event赋予一个实数值得到的变量 
例如 人群中人的身高，掷两次骰子得到的数值合

### PMF/PDF 
Random Variables 取特定的值(离散或者连续)时的概率 (或者离散情况下的频率)

## Conditional Probability
Given the condition that even B happens, we should revise our belief towards A .
总的来说就是改变你的sample space 从原来的所有到了B里面
$$
P(A \mid B)=\frac{P(A \cap B )}{P(B)}
$$
Condtional Probability are the same with ordinary probability. 依然遵循additivity
条件概率有些时候会给出非常反直觉的答案

Patition：
When using conditonal probability to partition another event, 
$$
P(B)=P(A_{1})(B\mid A_{1})+P(A_{2})(B\mid A_{2})+P(A_{3})(B\mid A_{3})+\dots
$$
these probalilities of these events A must add up to 1.
一个事件的背后总是有无数个小的影响因素，最后累加起来导致了该事件。

Reversing the conditioning:
![[Pasted image 20231125165216.png]]
Forms the foundation of Inference

Independence
Do not confuse the disjointness and independence
独立和互斥(Disjoint)没有直接关系，事实上两个说法衡量的事件都不在同一纬度上。
互斥指的是两个事件不可能同时发生，独立是指A发生并不影响B发生，时间上先后顺序
举例来说，第一次扔骰子和第二次的结果没有任何关系，这叫独立，disjoint是指在每一次扔骰子的过程中，扔到1和扔到2相互disjoint

# Couting
## Permutation

binomial coefficient
![[Pasted image 20231125204803.png]]
证明：解决从n个数按顺序取k个数
1.先随便拿k个数 再排序
2.n里取k个数放到篮子里，第一个篮子有n种选择 依次类推
$$
{N\choose k}k!=\frac{N!}{(N-k)!}
$$
检验： 当k=0 
空集 必须要考虑 因为不取也是一种选择 上式等于1

### Summation of Permutation
![[Pasted image 20231125204752.png]]
为什么是$2^n$:
	对于每个数字，有两种选择：被包括进集合或者不被包括进集合
### Binomial Equation
生成一个含有k个p事件和n-k个p的相反事件的序列的PMF
$$
{n \choose k}p^k(1-p)^{n-k}
$$
 
# Discrete Random Variables

Random Variables:
对sample space里的event赋予一个实数值得到的变量 
例如 人群中人的身高，掷两次骰子得到的数值合

PMF
Random Variables 取特定的值时的概率 (或者离散情况下的频率)

## 期望&方差

期望的感觉就像求gravity
期望代表了数据分布中最平衡的点
$E(x)=\sum xp(x)$

$E(g(x))\neq g(E(x))$

方差 
$D(x)=\sum(x-E(x))^2p(x)$
$D(x)=E(x^2)-(E(x))^2$

Independent Events:
独立事件的random variables 相加得到的事件的期望，期望仅仅是向右移动
举个例子，在我们抛硬币算抛多少次才能得到正面
我第一次失败了，后面几次还有多久能胜利的期望，
和我重新来抛一次的期望一样
因为我第一次失败与否不影响后面的概率
![[Pasted image 20231127210846.png]]

Partition:
说明条件概率的期望可以组合起来
![[Pasted image 20231127210902.png]]

How to measure the relation? 如何描述两个事件是否有关联
### Joint PMF 
两个事件同时发生的概率 
 

$E(x+y)=E(x)+E(y)$
$E(ax \mid y )=aE(x \mid y)$

when x y are independent
$E(xy)=E(x)E(y)$
！！！ **$D(x+y)=D(x)+D(y)$**  ！！！
只有在两个事件相互独立才成立
反证法 当 y=x 

$D(xy)=?$无信息

Binomial distribution:
$E(x)=\sum_{k=0}^n k {n \choose k}p^k(1-p)^{n-k}$
数目为n的序列出现k个概率为p的事件的期望 
将p的值x定为1 反之为0
$E(x_{i})=p$
$E(x)=np$

$D(x_i)=p(1-p)^2+(1-p)(0-p)^2=(1-p)p$
$D(x)=n(1-p)p$

### 例题 on Lec7

n个人随机分帽子的问题

# Continuous Distribution 
Continuous random variables:
从一个连续的数据轴随机选择 

PDF(Density)
在离散随机变量情况中考虑的是PMF(Mass)
PDF就像是某一点的重量分配了多少
注意 单独看某一点的PDF为0 $\int_{a}^a f(x) \, dx=0$

CDF(cumulative)
讨论这个点之前的点的PDF的积分

Covariance：
$$
Cov(x,y)=E((x-E(x))(y-E(y)))
$$
$$
Cov(x,y)=E(xy)-E(x)E(y)
$$
Corelation Coveriance 无量纲的协方差
$$
Cov(x,y)=E(\frac{x-E(x)}{\sigma_{x}} \frac{y-E(y)}{\sigma_{y}})
$$

## Gaussian Distribution 
根据公式出发
$\frac{1}{\sqrt{ 2 }\pi}e^{-x^2}$
所以高斯分布的曲线和 $x^2$有很紧密的联系
因此移动这个曲线 或者 让这个曲线变胖变窄 需要对 $x^2$进行缩放  $\frac{(x-\mu)^2}{\sigma^2}$

## Continuous & Conditional Probability 
类似于普通的PDF, 条件概率下的PDF像是对联合分布关于事件y的放缩：
$$
f_{x\mid y}(x\mid y)=\frac{f_{x y}(x,y)}{f_{y}{(y)}}
$$
例如下图，
![[Pasted image 20231228020259.png]]
最左上角的图指的是x与y的联合分布，中间的图指的是固定x值的y的分布 
在求y on condition of x的时候，其实就是把这个分布归一化，使得这个条件x下的概率密度函数的积分合等于1

## 贝叶斯
什么是推理？
已知的y有一个分布，实际上这个y是由x通过某种变换得到的，现在要求p(x|y)
通过y反推x的分布 

![[Pasted image 20231228021315.png]]

上面给出了xy都为离散或者连续的时候的贝叶斯，以下讨论两个不同时为离散或者连续的情况：
当x为离散时y为连续时，
![[Pasted image 20231228022026.png]]

当有一个离散另一个连续时，他们的联合概率分布还是依然相等:
两个事件的联合概率分布可以写为
![[Pasted image 20231228022631.png]]
而用连续和离散来写可以写为
![[Pasted image 20231228022804.png]]
因此回到上面的那个例子：
我们在做推断的时候，我们根据y判断x=0或者x=1的概率，实际上就是比较两个条件概率的大小
![[Pasted image 20231228023514.png]]

## Derived Distribution 
已知随机变量的分布（一个或者多个），他们经过函数的变换后的分布是什么？

离散情况下，通过函数的映射关系找变换前的对应的点就可以得到变换后的分布了。但在连续的情况，因为点的概率为0，所以只能通过CDF来找两个之间的联系。

例如：
$$
y=ax+b；
f_{y}(y)=\frac{1}{|a|}f_{x}\left( \frac{y-b}{a} \right)
$$
得到的y的分布，相当于把x缩放了再平移了
除以a是因为为了使变换后的分布的和为1

什么时候可以跳过求CDF直接去求变换后的PDF呢？
当g(x)单调时
$$
f_{x}(x)=f_{y}(y)|\frac{dy}{dx}|
$$
因为单调时，两个分布的一小段分布$\delta$也是一一对应的，所以可以根据微分得出两个分布的$\delta$的关系来求两个分布的关系。

### 多元分布 
$$
W=\frac{X}{Y}
$$
x&y are independent 
$$
E(w)=E(X)E\left( \frac{1}{Y} \right)
$$

A special case:
$$
w=x+y
$$
convolution:
$$
f_{w}(w)=\sum_{x}f_{x}(x)f_{y}(w-x)
$$
固定w，通过w-x替换y对所有的x求和。

Independent多元高斯分布 
$$
f_{xy}(xy)=f(x)f(y)=\frac{1}{2\pi \sigma_{x}\sigma_{y}}\exp\left( -\frac{(x-\mu_{x})^2}{\sigma_{x}} -\frac{(y-\mu_{y})^2}{\sigma_{y}} \right)
$$
对于特定的xy的分布：固定xy的值得到：
$$
 -\frac{(x-\mu_{x})^2}{\sigma_{x}} -\frac{(y-\mu_{y})^2}{\sigma_{y}} 
$$
这是一个椭圆方程，所以双元高斯的分布的横截面为一个椭圆，椭圆的长由方差决定，方差越大椭圆越长

sum of normal distribution is also a normal distribution

Dependent 多元高斯分布 
Dependent 的情况下横截面得到的椭圆的长轴和短轴不再平行于x轴和y轴
这个椭圆会发生一定倾斜
因为越大的x往往会得到越大的y（或者越小的y），所以xy相互dependent

## Iterated Expectation
介绍条件概率下的两个dependent随机变量的期望的关系
$$
E(E(x|y))=\sum_{y}p_{y}E(x|y)=E(x)
$$
$$
E(var(x|y))+var(E(x|y))=var(x)
$$
上式的两个term分别代表了两种观测
可以把y当成对x的分类
第一个term表示每类下的x分布的方差的期望，代表类内的期望
第二个term表示不同类之间x分布的期望的方差，代表类间的方差

# Random Process
描述一系列的随机事件
Two views:
1. 一系列的随机事件
2. 多个子系列的随机事件的joint PDF 

对无限长的序列的概率为0

## Bernoulli Process
PMF：
$$
{n \choose k}p^k(1-p)^{n-k}
$$
Interval Times -Geometric Distribution 

Time of k arrival - Pascal PMF 


## Poison Process
泊松分布认为单个小的interval出现一次的概率为$\lambda\sigma$,$\lambda$代表rate
![[Pasted image 20231230163254.png]]
$\tao$个时间内出现k个事件的概率：
![[Pasted image 20231230163534.png]]
Interval Time
![[Pasted image 20231230163717.png]]

## Markov Chain
Market Model：
每一个state 可以代表当前排队的人数；每个state由新来一个排队的概率和离开一个的概率连接。

可以写成一个迭代式，由n-1的state 得出 第n个state（由于markov的assumption）
Key recursion:
$$
r_{ij}(n)=\sum_{k}r_{ik}(n-1)p_{ik}
$$
k代表其他可能到i的states

### Steady States
与initial state 无关 最后的state的概率会趋近于一个stationary state 

什么时候initial state matters？
1. Recurrent vs Transient： recurrent 指从该点出发，总有可能可以回到该点。
 当一个markov有多个recurrent loops 时，initial state matters
2. Periodic  States  

大量的n之后，每个state都有出现概率$\pi$，这个概率和initial state无关
$$
\lim_{ n \to \infty }P(X_{n}=j|X_{0}=i)=\pi_{j} 
$$
要求这个概率，解下面的方程：
$$
\pi_{j}=\sum_{k}\pi_{k}p_{kj}
$$
$$
\sum_{j}\pi_{j}=1
$$
从频率角度：
-从大量的数据来看 对不同的states $\pi$代表他们出现的频率；
-可以根据state transition的概率得到多个state的 $\pi$的等式，通过概率和为1 求解

Special case：
当state transition间的概率在每个state间都是恒定的，即p/q 都一致，
即使这个markov有无限个states 由于迭代式 模型的本质其实还是两个states的问题：
$$
\pi_{j}p=\pi_{j+1}q
$$
这个qp的关系决定了这个markov的稳态，到底是无限的增长还是平衡，
因此用$\rho=\frac{p}{q}$
1. 当 $\rho=1$: 代表每个state都有可能 
2. 当 $\rho<1$: 
$$
\pi_{n}=\pi_{0}\rho^n
$$
假设有m个states
$$
\sum_{j}\pi_{j}=1=\sum_0^m\pi_{0}\rho^n
$$
取m limit to infinity 解得
$$
\pi_{0}=1-\rho
$$
最后得到每个states的pi为关于$(1-\rho)\rho^n$geometric下降的分布 

关于Markov中recurrent 和 transient 的两个问题 
1. 进入特定的recurrent loop的概率 （有多个recurrent loops时）
2. 进入recurrent loop的时间（期望）
求解这两个问题需要与initial state有关，需要写出一个迭代的方程，一直到进入到recurrent loops中

### Periodic 
periodic  周期性振荡：
 周期性的在两个group间来回振荡

# 大数定律

## 基本概念
概率的收敛：
对于任意的$c$，存在$\sigma$
使得$P(|X-\mu|>c)<\sigma$ 成立
则概率收敛于μ

这个概率与n有关（采样数目）

概率收敛能告诉我们什么？
只能告诉我们随着采样数目的增多，这个样本的分布会集中在一个数上。
但不能告诉我们关于这个分布的细节，
例如下面这个例子 
![[Pasted image 20240103142950.png]]
Yn确实收敛于0，但是分布的期望并不为0

## 两个不等式
Markov不等式 
![[Pasted image 20240103140655.png]]
这个等式联系均值和概率
告诉我们当均值小时，采样的数据很大概率也是一个小数目

Chebyshev's inequality
![[Pasted image 20240103140132.png]]
切比雪夫不等式说明了远离均值的分散的点的概率与方差的关系 

## 弱大数定律
弱大数定律证明了采样的均值是否依概率收敛于样本的期望。

从样本收敛到实际分布的期望和方差：

对于独立同分布（期望为μ，方差为σ）的样本采样n次得到n个样本点Mn
$$
Mn=\frac{X_{1}+X_{2}+..+Xn}{n}
$$
他的期望和方差
$$
E(Mn)=\mu
$$
$$
Var(Mn)=\frac{\sigma}{n}
$$
根据切比雪夫不等式可以得出Mn和样本真正的期望的收敛情况：
![[Pasted image 20240103141504.png]]
这个不等式告诉我们n取多大时，这个样本的均值与真正的分布的期望误差才可以控制在一定范围内

## 中心极限定理
中心极限定理证明了采样点的分布是否是normal distribution 

标准化：
由于上述的Mn的计算会导致Var不等于σ（将样本的分布堆叠起来的后果是样本分散得很开）
于是换了种计算方式-标准化：
![[Pasted image 20240103145456.png]]

中心极限定理告诉我们标准化后的Zn的CDF和正态分布一致（为什么是CDF，因为即使是PDF不一样但CFDF也相似）



