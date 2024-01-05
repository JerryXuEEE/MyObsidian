---
title: 未命名 1
date: 2023-09-25 19:57:46
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: inprogress
destination: 
share: false
obsidianUIMode: source
---

# Neural Encoding
## Firing Rates and Spike Statistics
### Properties of Neuron



**Axons and dendrites** : Axons from single neurons can traverse large fractions of the brain or, in some cases, of the entire body. In the mouse brain, it has been estimated that cortical neurons typically send out a total of about 40 mm of axon and have approximately 4 mm of total dendritic cable in their branched dendritic trees. The axon makes an average of 180 synaptic connections with other neurons per mm of length and the dendritic tree receives, on average, 2 synaptic inputs per μm.

**Hyperpolarization** : Current in the form of positively charged ions flowing out of the cell (or negatively charged ions flowing into the cell) through open channels makes the membrane potential more negative, a process called hyperpolarization.

**Action potential** : An action potential is a roughly 100 mV fluctuation in the electrical potential across the cell membrane that lasts for about 1 ms.

### Recordings
Membrane potentials are measured intracellularly by connecting a hollow glass electrode filled with a conducting electrolyte to a neuron, and comparing the potential it records with that of a reference electrode placed in the extracellular medium.

### Neural Decoding
神经信息包含了什么？
spike 间的average interval； stimulus-response relationship ;

Decoding 的困难：
Isolating features of the response that encode changes in the stimulus can be difficult, especially if the time scale for these changes is of the same order as the average interval between spikes.

The complexity and trial-to-trial variability of action potential sequences make it unlikely that we can describe and predict the timing of each spike deterministically. Instead, we seek a model that can account for the probabilities that different spike sequences are evoked by a specific stimulus.

we introduce the firing rate and spike-train correlation functions, which are basic measures of spiking probability and statistics.

### Spike Trains and Firing Rate 
当谈论到Firing Rates 这个概念，我们一般指spike的概率，因为spike具有随机性。
在计算这个概率时，我们可以理解为对trial和时间的平均，因此定义为：
![[Pasted image 20230925203347.png]]

Here the $\rho(t)$ refers to the spike trains during the time interval.
![[Pasted image 20230925203249.png]]
由于定义中积分的性质，firing rates 在积分中可以和spike trains 相互转换
![[Pasted image 20230925211631.png]]

firing rate简单点的定义还可以为仅仅对时间的平均，不过使用很局限因此出现的不多

### Measuring Firing Rates 
spike trains 是离散的信号， 而 probility density都是连续的，怎么从离散的时间序列得到连续的spike的分布呢？

One way to avoid quantized firing rates is to vary the bin size so that a fixed number of spikes appears in each bin. The firing rate is then approximated as that fixed number of spikes divided by the variable bin width.

To avoid the arbitrariness in the placement of bins, we can instead take a single bin or window of duration 1t and slide it along the spike train,
![[Pasted image 20230925211812.png]]

![[Pasted image 20230925211821.png]]

In a continuous form:
![[Pasted image 20230925211838.png]]

Moreover we can use some other kernels in continuous form such as Gaussian kernel:
![[Pasted image 20230925211932.png]]

In this case, σw controls the temporal resolution of the resulting rate, playing a role analogous to delta_t.

A postsynaptic neuron monitoring the spike train of a presynaptic cell has access only to spikes that have previously occurred. An approximation of the firing rate at time t that depends only on spikes fired before t can be calculated using a window function that vanishes when its argument is negtive. 
![[Pasted image 20230925212012.png]]

![[Pasted image 20230925212020.png]]
where 1/α determines the temporal resolution of the resulting firing-rate estimate.

Above estimation are illustrated:
![[Pasted image 20230925212111.png]]

### Tuning Curves
简单来说就是用一个函数来联系刺激s和firing rate。 
$$
<r>=f(s)
$$
例如在V1的neuron对不同方向的光线的firing rate可以用一个函数来表示
$$
f(s)=r_{max}\exp( -\frac{1}{2} (\frac{s-s_{max}}{\sigma_{f}})^2  )
$$
![[Pasted image 20231013142510.png]]
### Spike-Count Variability 
用Spike  count得到的tuning curve会有很大的噪音 因为trial和 trial之间神经元的发放会有随机性 

有些噪音会与tuning curve 无关 被称作 additive noise 有些则与tuning curve 有关

## Why does neuron fires?
回答这个问题 我们主要关注的是什么样的刺激会引起神经元的发放。

Spike-triggered average： 将spike前特定时间内出现的刺激加起来了，可以反应刺激与spike之间的关系，一般来说这个刺激是与时间连续的。

由于spike 的 firing rate 是用 凯迪拉克函数的集合表示的，因此spike-triggered average 的计算可以用 firing rate 乘以 stimulus function 关于这个刺激时间的积分。

Correlation function 是用于表示两的量之间的关系，计算是两个量相乘后关于时间积分。由于correlation function与上面提到的spike triggered average计算上的相似性，因此spike triggered average 也被称作stimulus和spike的反向correlation（反向是因为先考虑的spike在反过去找stimulus）

Multiple Spike ： 为了研究某类spike pattern（具体指多个spike且时间间隔特定）与stimulus的关系 
	一个例子![[Pasted image 20231018142032.png]]
	这张图来自于 H1 movement-sensitive visual neuron of the blowfly. 当对这中绿头苍蝇展现的飞行模拟的画面的飞行速度改变时，H1 neuron 呈现出不同的发放模式。

### White Noise 
当刺激呈现一定的fluctuation 时 神经元的发放会怎么变呢？

这类研究就需要用到白噪声刺激，因为白噪声刺激的强度不变，并且刺激间的correlation为0.

## 频率视角下的Spike sequence
这里讨论如何measure sequence 的概率 

为什么不能用firing rate来表示spike sequence出现的概率？
因为一般来说两个连续的spike的概率（联合分布）不等于两个单独的spike的r相乘。

Point Process： 生成一个sequence的随机过程

Renewal Process:  spike 之间的相关性只在一定的interval存在。 例如考虑了recovery process的spike sequence 

Poisson Process： spike的出现与前一个spike没有任何相关 
	Poisson 又可以根据firing rate 是否恒定分为Homogeneous and inhomogeneous

### Homogeneous Poisson Process 
不管怎么说，一个序列出现的概率都可以用在某一时刻出现spike的概率通过乘上后得到

需要做点概率计算： 将T时间内划分为M个小的时间间隔bin，从M个随机取n个间隔且不考虑先后，则有
$$
\frac{M!}{(M-n)!n!}
$$
个取法。那么假设在一个bin中出现spike的概率为p，而在Homogeneous Poisson中p等于firing rate r乘以bin的时间interval，因此在M中出现n个spike的几率为
$$
p^n(1-p)^{m-n}
$$
将两式相乘则得到在T内中出现n个spike的概率，当我们使这个bin的时间长度趋近于0，这个式子等于 
$$
P_{r}[n]=\frac{(rT)^n}{n!}\exp(-rT)
$$
这就是泊松分布，刻画的是一段时间T内出现个n个spike的概率分布，spike的firing rate 为r #Poisson 

Poisson 和 Gassian Distribution 的关系：
	当rT=10（T时间内出现spikes的数量的平均值为10）， 两个分布基本重合
	![[Pasted image 20231018161956.png]]
Poisson 的方差也为rT (deduction is omitted here) 这说明Poisson的方差和均值是相同的
(方差除以均值，quantified by Fano Factor)

Interspike Interval 
	Measure the possibility that the two spikes' invertal is $\tau$.
	According to Poisson distribution, not having spike in $\tau$ is $\exp(-r\tau)$. Therefore, the possibility is 
	$$P_{\tau}=r \nabla t\exp(-r\tau)$$
	So the distribution is  $r \exp(-r\tau)$
	The mean is 1/r and deviation is $\frac{1}{r^2}$
很有趣的是，从这里我们可以看出，当tau非常小时，这个值取决于r的大小，当tau非常大时，这个值会随着tau指数下降。这也非常符合生物视角。

**The Interspike Innterval measures the distribution over time interval. Can we generalize this idea by summarizing all the spike pairs during a time period.**

Histogram view: Take every spike pair into consideration. Every spike pair has a time interval, $m \nabla t$.(相差m个bin). We summrize how many spike pair under a specific m, which is quantified by $N_{m}$

When the spike interval follows a uniform distribution, then the $N_{m}$ becomes $\frac{n^2\nabla t}{T}$ as there are $n^2$ spike pairs and $\frac{T}{\nabla t}$ bins. 

The final quantity of the histogram is defined by dividing the resulting numbers by T, so the value of the histogram in bin m is $Hm = \frac{Nm}{T} − \frac{n^{2}\nabla t}{T^2}$.

When the bin $\nabla t$ goes to zero, this quantity becomes the so-called spike-train aurocorrelation. 

### Spike-train aurocorrelation. 
$$
Q_{pp}(\tau)=\frac{1}{T}\int_{0}^T <(\rho(t)-r(t))(\rho (t+\tau)-r(t))> \, dt 
$$

## Inhomogeneous Poisson Process 
书上没有仔细讲只给了个公式，公式的推导在附录c中

Poisson Spike Generator
	根据stimulus s(t) 来生成一系列 spike train。 具体来说有两种方法：
	1. 通过概率，在特定时刻的概率与一个0-1的随机数比较，如果大于，就在此时刻生成一个spike 
	2. 通过ISI 在τ时间内生成Spike的概率大于随机数则生成 实际上用公式可以这么写：
$$
t_{i+1}=t_{i}-\frac{\ln(X_{rand})}{r}
$$ 
因为ISI其实是exp（r）,所以对随机数X取了对数，实际上是一样的

总结一下：对比实验数据和模拟数据的三个参数 Fano Factor; ISI; variation 

## Neural code
	当我们讨论神经信息时，到底看单个神经的firing rate还是多个神经元的correlation code。
Independent neuron code: 在多数情况下，单个神经元的发放率已经足以表现出tuning curve了 correlation code 的 significant information 可能只有10% 

Correlation code: 包括Synchrony and oscillations

*我认为这是一个挺反直觉的结论 
另外 当我读到Independent code 我还以为是讨论spike 的发放与否和历史信息的关系 或许这个不包括significant的神经信息？
还有 对于significance的定义 我认为是基于tuning curve的 就是当刺激改变时 neuron的表现还是这样吗*

### Temporal Code 
这里讨论spike的时间分辨率 或者是spike的频率最高到多少

困难：刺激频率增大 spike的频率也会增大 但是我们在讨论neural information 时， 我们关注的是stimulus的种类 

两种方法，一是用spike的峰值来表示spike，因为当刺激频率增大而引起的spike的值总是小于peak的； 二是用刺激本身，刺激种类改变的速度来刻画分辨率。

作者最后指出，虽然有这么多debate 但是重要的应该还是the relationship between  firing patterns of different neurons in a responding population and to understand their significance for neural coding.

# Reverse Correlation 
With spike-triggered average, here we discuss the technique of reverse correlation and some others to find the linear and nonlinera relationship between stimulus and spikes , the tuning curve.

首先我们假设spikes come from linear combination of previous stimulus:
$$
r_{est}=r_{0}+\int _{0}^{inf} D(\tau)s(t-\tau)\, d\tau 
$$
接下来是根据实验测得的r来对D进行优化
Optimization：
$$
E=\frac{1}{T}\int _{0}^{T}(r_{est}-{r(t)})^2 \, dt 
$$
And this gives:
$$
\int _{0}^{inf}Q_{ss}(\tau-\tau')D(\tau) \, d\tau=Q_{rs}(-\tau) 
$$
This is called reverse correlation as there is $-\tau$ in $Q_{rs}$
The following shows an example of white noise stimulus:

White Noise 
	$$
	Q_{ss}=\sigma^2 δ(τ)
	$$
	This gives the $D(\tau)=\frac{Q_{rs}(-\tau)}{\sigma^2}$
	This means that the optimal kernel function is the same with the tuning curve.

## Static Nonlinearities
线性叠加刺激来模拟spike的问题主要有两个，线性叠加无法避免负值的出现，因为线性无法给出一个upbound or lower bound 因此他的增加也是无限的

所以我们要使用一个非线性的函数 类似于sigmoid的那种 给出一个取值区间 因此可以将原来的线性式子改写为：
$$
r_{est}=r_{0}+F(\int _{0}^{inf} D(\tau)s(t-\tau)\, d\tau) 
$$
F 这个filter function 可以有多种取法 具体参考原作

总的来说 根据测到的firing rate 和 stimulus 我们生产 模拟spike trains 的过程总结下来为：
![[Pasted image 20231024144429.png]]
最后的spike generator 是上一章介绍的 Poison Generator 

## Early Visual System
不太清楚这里为什么要讲Visual System 可能拿来举reverse correlation 的例子？

一张非常牛逼的图：
![[Pasted image 20231025133811.png]]

视网膜，LGN，初级视觉皮层三者的连接如上，其中左右的神经轴突在中线相交于optic chiasm

视觉皮层的神经元只对某一视线内的刺激产生相应，这个刺激被称作感受野。感受野临近的刺激也会对视觉反应产生影响，不过在本章内没有讨论。

本章主要讨论现实世界的画面和视觉皮层的神经元的一对一的映射关系，这也是视觉皮层非常重要的一个特点。
书上举了一个果蝇的例子。


