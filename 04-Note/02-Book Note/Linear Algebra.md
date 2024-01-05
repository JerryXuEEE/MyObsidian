---
title: Untitled 1
date: 2023-08-30 17:41:02
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: inprogress
destination: 
share: false
obsidianUIMode: source
---
# Gilbert String
# Ax=b
方程组

$$
\begin{bmatrix}
2x+3y+z=2  \\
x+5y+2z=1 \\
9x+2y+3z=6
\end{bmatrix} 
$$
$$
\begin{bmatrix}
2 & 3 & 1 \\
1 & 5 & 2 \\
9 & 2 & 3
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z \\
\end{bmatrix}
=
\begin{bmatrix}
2 \\
1 \\
6 \\
\end{bmatrix}
$$
Colum picture and row picture looks at the matrix calculation in different perspectives. Column takes the each column of A as vectors and output answer is the linear combination of these vecotrs. The row picture regard the output answer as the linear combination of the rows of x, which is the right matrix.
## Row Picture
2x+3y+z=2 组成一条线 三个方程分别组成一条线 得到的交点则为解
Or Let's take a look at another form:
$$
\begin{bmatrix}
1  & 2 & 3
\end{bmatrix}
*
\begin{bmatrix}
4 & 3 & 2 \\
3 & 2 & 1 \\
2 & 1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1*row_{1}+2*row_{2}+3*row_{3}
\end{bmatrix}
$$

## Column Picture
$$
\begin{bmatrix}
2 \\
1 \\
9
\end{bmatrix}x+
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}y+
\begin{bmatrix}
1 \\
2 \\
3 \\
\end{bmatrix}z=
\begin{bmatrix}
2 \\
1 \\
6
\end{bmatrix}
$$


# Elimination
But why don't we, as long as we're sort of seeing how elimination works, see hou could it fail.

用矩阵的来表示方程式的消元 A to U (upper triangular matrix)  每一行剩下一个pivot 如果pivot为0 还要做row exchange.

## Elementary Matrix 
Elementary matrix refers to a series matrix that generate the A into U by multiplication.
We have to look at it in row picture.
**For example,**  if we do elimination for matrix like this:
$$
\begin{bmatrix}
2 & 3 \\
4 & 2  \\
\end{bmatrix}
$$
we need to substract the second row of two times the first row 
So E should be 
$$
\begin{bmatrix}
1 & 0 \\
2 & 1
\end{bmatrix}
$$
Because the second row of the E does 2 times the first row and add to 1 times the second row.
# Inverses
When does Inverse exist?
singular matrix: Ax=0 for non-zero x

The columns can not span $R^n$

Triangular matices 的逆为对角元素不变 三角区的元素取相反数
# Factorization A=LU
Let's do elimination  on A, which is a non-singular matrix and assume no row exchange.
E A = U
Take the inverse steps give A=L U.
Here the L refers to Lower triangular matrix. 
Furthermore, we can expand this U into a diagonal matrix and another upper triangular, and this can be tested with another random matrix.
## A = L D U


Why is this factorization nicer than EA=U?
Because L only keeps the multipliers
Lets take a look at $E_{21}$ and $E_{32}$,  suppose there is no $E_{31}$, meaning thr entry A31 is 0. 
$$
\begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
0 & 0 & 1  \\
\end{bmatrix}
$$
This matrix adds the two times the first row to the second row
$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 5 & 1
\end{bmatrix}
$$
This matrix adds the five times the second to the third row.
So if we look at them together, which gives the total picture of elimination E from $E_{32}$$E_{21}$:
$$
\begin{bmatrix}
1 & 0 & 0 \\
2 & 1 & 0 \\
10 & 5 & 1 \\
\end{bmatrix}
$$
So this E keeps the multiplier 10, which results from the $E_{32}$ five times the second row containing  the $E_{21}$ two times the first row. So the 10 times the first row is added to the third row.
But for the LU factorization this could be avoided because the L takes the inverse of $E_{32}$ first, where the row 1 is not involved.

# Permutations and Transpose
	Permutation 本质就是做row exchanges
对3x3 的矩阵：

$$
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$
有六种打乱的方法 每一种都可以写成row picture来做row exchanges 

**非常有意思的是上面列的矩阵的逆就是他的transpose**
例如 
$$
\begin{bmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$
他的逆就是第一行放第三行 第二行放第一行 第三行放第二行
$$
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
\end{bmatrix}
$$

## Symmetric matrix 
$R^TR$永远是 对称矩阵 
证明
$(R^TR)^T=R^TR$

# Vector space and subspace
How to define vector space?
	Take linear combination of any n unaliaed vectors in this space and the result is still in this space 
A counterexample: 第一象限

subspace 其旗下的vector space 
eg. subspace of R2
	all of R2
	any line through [0 0]
	[0 0 ] itself 

## Nullspace
Ax=0 

What is inside the Nullspace?
	When solving Ax=0 good thing is we can do collumns operations and row operations anything we want 
	When we do ellimination on A to get the reduced echelon form:	
	$$
	\begin{bmatrix}
	I & F \\
	0 & 0
	\end{bmatrix}
	$$
	This shows that x = 
	$$
	\begin{bmatrix}
	-F \\
	I
	\end{bmatrix}
	$$
So what inside nullspace, is the linear combination of the non-pivot elements in reduced matrix A. 
An example here:
A:
$$
\begin{bmatrix}
1 & 2 & 2 & 4 \\
2 & 4 & 6 & 8 \\
3 & 6 & 8 & 14
\end{bmatrix}
$$
The elimination goes to:
$$
\begin{bmatrix}
1 & 0 & 2 & 2 \\
0 & 1 & 0 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$
Therefore, the x goes to:
$$
\begin{bmatrix}
-2 & -2 \\
0 & -1 \\
1 & 0 \\
0 & 1
\end{bmatrix}
$$
This shows that the nullspace is spanned by these two vectors.

**Why do we need the nullspace?
What does the nullspace tell us?**
The nullspace tells us how many free varables are there.

## Ax=b 

The solution contains a particular solution and null solutions, because Ax=0. 

relationship among r (rank) m (row numbers) and n (collumn numbers). 
1. r = m : no limitations on b; there might be free variables, = nullspace 
2. r = n : every collumns are indepent, so there is no free variables and nullspace but there are limitations on b for existence of x.
3. m = n

the question about solvability

## Four important subspaces:
Collumn space of A, Null space of A
Rows of A ( C(trans(A)) ), null space of trans(A)
**注意是 nullspace of A transpose 垂直于 collumn of A**
When we do ellimination , we can extract the information of the row space and the null of row space fron ellimination matrix E. Because the ellimination does the row operations to get 
$$
\begin{bmatrix}
I & F \\
0 & 0
\end{bmatrix}
$$
So the 0 gives the null row space. 

## Matrix Space 
将vector当成一个matrix  将会得到一个船新的视角

The basis for a 3 by 3 matrix contains 9 vector matrices.

We can do union, sum and mutiply by scalor to these vector matrices 

## Rank 1 matrix
A=u trans(v)
秩一矩阵永远可以被分解成两个一维向量
## Small World and Graph Theory
图论 
4nodes 5 edges can be represented by a 5x4 matrix 
假设一个图：
![[Pasted image 20231106202953.png]]
那么他矩阵 注意列代表node 行代表edge -1 +1 代表方向
$$
\begin{bmatrix}
-1 & 1 & 0 & 0 \\
0 & -1 & 1 & 0 \\
-1 & 0 & 1 & 0 \\
-1 & 0 & 0 & 1 \\
0 & 0 & -1 & 1
\end{bmatrix}
$$

这个矩阵包含了非常多有意思的特征
首先就是他的四大subspaces：
1. 列空间 说明了有多少个independent variable 上面的A的rank 为 3 因此有三个node是independent的
2. Ax=0 他的列nullspace 说明了他有多少的free variables  如果你解开这个方程组你会发现这个方程将四个nodes联系起来了 从电路角度来看这个方程Ax(因为A里都是-1 1)这个结果得出两个node间的差所以其实可以看成两点的电压差 
4. 行空间 说明有多个edge是independent的 上面的A为3 
5. 行null space 说明有多少个edge是dependent 说明有多少个loop 或者验证了基尔霍夫电流公式
![[Pasted image 20231106205032.png]]

**这节课挺有意思的 可以反复看看 时不时会有新的理解**
从应用数学的角度看这节课
首先A给出了这个图里点与点的连接和方向
Ax 给出电压差 乘以系数c （矩阵 代表每个edge的电阻）得到电流
trans(A)y得到基尔霍夫电流定律 
如果这个系统有外部输入的电流 那么 
$$
A^TCAx=f
$$
解上述方程就可以解出电压  非常神奇 然而我大五了才明白

## Orthogonal 

两个平面一旦有intersection（原点除外） 则一定不正交 xy 平面和yz平面 正交吗 明显不正交

 null space of A transpose  和 collumn space 正交的原因
	Ax=0 null space 的定义 而collum space of A 是由A的每一列vector span 成的 而Ax=0 又可以看成第一列的A乘以x1等于0 依次类推 因此和null space 垂直

Ax=b 判断有无解问题 =b是否在C（A）内

$A^TA$是一个很有用的矩阵  

那么什么情况下这个矩阵可逆呢？
答：当A的collumns为independent
证明：矩阵可逆等于证明只有当x等于0时 $A^TAx=0$
两边同时乘以$x^T$ 得到$x^TA^TAx=0$
等于$(Ax)^TAx=0$
说明Ax=0 则x=0

$N(A^TA)=N(A)$ 或者两者的rank相同

A是否可逆问题及collumn space是否满秩

## Projection
**Dr. Strang 说这节课是最重要的一节课**

Projection 找b在col(A)中的投影(或者说是最近的vector)
**Highly Important**
首先，投影在A上 因此可以先写成对A的缩放 xA
那么b-xA则一定垂直A
$$
A^T(b-xA)=0
$$
解得$x=\frac{A^Tb}{A^TA}$
那么投影可以写成$xA$or $Ax$


Ax=b 无解时 可以寻找A中离b最近的vector 
	b在A上的projection=$A(A^TA)^{-1}A^Tb$
	 
当已知b对A的投影为p 那么对null(A)的投影怎么求呢？
答： Projection of $A^T$=$(I-P)b$ 可以用矢量减法证明

Use projection to find the regression:
 已知一些点(x1,y1) (x2,y2) (x3,y3)
 假设一条线穿过他们 因此可以写成三个方程的方程组然后用矩阵乘法表示Ax=y
	$$
	\begin{bmatrix}
	x_{1} & 1 \\
	x_{2} & 1 \\
	x_{3} &  1
	\end{bmatrix}
	\begin{bmatrix}
	C \\
	D
	\end{bmatrix}
	=
	\begin{bmatrix}
	y_{1} \\
	y_{2} \\
	y_{3}
	\end{bmatrix}$$
	CD为直线的系数 
	那么回归问题可以转化为b在A上的投影得到的直线 
	所以转化为$Ax=Pb=A(A^TA)^{-1}A^Tb$两边同乘$A^T$得到 
	$A^TAx=A^Tb$	
## Orthonormal

$Q^TQ=I$非常重要的特性
	Projection=$Q(Q^TQ)Q^Tb=QQ^Tb$ 

Q为square matrix
$Q^TQ=I;Q^T=Q^{-1}$


当Q为square matrix 时 Projection = I
因为Q满秩 b在A的projection为b本身

因此在求解$A^TAx=A^Tb$时可以转化为$x=Q^Tb$ 

Gram-Schmit
目的是将一堆vectors(或者说是一个矩阵中的各个vector)化为orthonormal vectors 
核心思想是投影 假设两个vector a and b : 可以得到 垂直于a的向量为 b-proj(b on a) 下一步就是通过除以长度得到normalized

# Determinant
四个性质
1. det(I)=1
2. row exchanges 使det 变为负
3. linear combinations ：1. 矩阵中第一列乘上一个a 则行列式也乘上一个a  2.![[Pasted image 20231110195636.png]]
4. 一个矩阵中有equal rows 则行列式为0 （可以由2推出来）
剩下六个性质都可以由这四个公式推出来
很重要的几个：
5. 对角矩阵的行列式等于对角元的乘积 所以矩阵的行列式等于其主元的乘积 
6. elimination 不改变其行列式 
7. 三角矩阵的行列式等于对角矩阵（因为主元定了 通过elimination回到对角矩阵）
8. $\det(A^T)=\det(A)$
9. $\det(AB)=\det(A)\det(B)$

determinant formula: 
可以用行列式linear combination的特性推出来
$$
\begin{bmatrix}
a & b \\
c & d \\
\end{bmatrix}
=
\begin{bmatrix}
a & 0 \\
c & d \\
\end{bmatrix}+
\begin{bmatrix}
0 & b \\
c & d
\end{bmatrix}=
\begin{bmatrix}
a & 0 \\
c & 0 \\
\end{bmatrix}+
\begin{bmatrix}
a & 0 \\
0 & d
\end{bmatrix}+
\begin{bmatrix}
0 & b \\
c & 0
\end{bmatrix}+
\begin{bmatrix}
0 & b \\
0 & d
\end{bmatrix}
=ad-bc
$$
行列式的公式可以看成是对矩阵里的元素的排列组合，例如选第一行的一个数，在排除这一行这一列剩下的矩阵中再选，不过要注意符号。

Cofactor:
将a11提出来，剩下的除开第一行第一列余下的矩阵为c11(注意符号 c11为正 c12.  c21为负)
可以将行列式公式拆开写为a11c11+a12c12+...

### Cramer's Rule

### Volume
行列式的物理意义 体积 面积 

两个向量确定的三角形的面积=
$$
\frac{1}{2}(ad-bc)
$$

# Eigenvectors
Ax parallel to x
将A看成一个function 我们关注的是经过这个方程变换的x且不改变方向 

trace= 所有eigenvalues的和
det = 所有eigenvalues 的积

特征方程 
$\det(A-\lambda I)=0$
由此可见 当λ包含0 则说明矩阵不可逆

eigenvalue 的三种情况 
1. 为实数 且数目为n
2. 为实数 但只有一个
3. 为虚数 当矩阵非常不对称时就可能出现这种情况

## 对角化
假设我的eigenvectors 互不相同 我们将他们写成一个矩阵称为S 
$Ax=\lambda x$ 写成 $AS=S\Lambda$ (因为将S看为一个row的x)

如果再乘一个A
$A^2S=S\Lambda A=S\Lambda^2$
所以A的n次方 其实等于lambda的n次方 
可以用这个判断 $A^n$的极限 

那么什么时候A有不同的eigenvectors 呢
答：所有eigenvalues 不同时

## Powers of A
为什么我们需要eigenvalue呢

对于迭代的系统 可以根据矩阵乘法写成一个公式（具体看Lec 22. Fabonacci）
 $u_{k+1}=Au_{k}$
迭代的可以写作 $u_{n}=A^nu_{0}$
通过特征值分解 可以写成 $A^nS=S\Lambda^n$
而$u_{0}$可以写成多个特征向量的linear combinations $u_{0}=Sc$
 $u_{n}=A^nu_{0}=A^nSc=S\lambda^n c$
可以看出在迭代的过程中 也就是系统dynamic中 看出到底是哪一项对系统的影响最大 因为小于1的eigenvalue最后会趋近于0


## Differential Equation

对于一介差分系统 可以写成 $\frac{du}{dt}=Au$A 代表这个差分系统
这种情况下通解不再是$u_{n}=A^nu_{0}=A^nSc=S\lambda^n c$而是$u_{n}=c_{1}e^{\lambda_{1}}x_{1}+c_{2}e^{\lambda_{2}} x_{2}=Se^\lambda v$ v代表linear combination的系数 代表有多少x1x2参与决定最后的u
注意到$u_{0}=c_{1}x_{1}+c_{2}x_{2}=Sv$ 因此$u_{n}=Se^\lambda v=Se^\lambda S^{-1}u_{0}=e^{A}u_{0}$ 
可能会好奇为什么$e^{A}=Se^\lambda S^{-1}u_{0}$ 从泰勒展开的角度来证明
![[Pasted image 20231113215952.png]]

在power equation（？不确定是不是叫这个名 或者说是离散系统 像fabonacci 数列）在differential equation 中 lambda是否大于1决定了系统最后是否converge中 lambda是否小于0 决定了最后是否converge 等于0 代表了static

### Markov Matrix 
这节更像是一个应用

Markov 的性质 
1. 列的和为1
2. 矩阵元素小于等于1
由这两个性质可以得到关于特征值的两个性质 (具体证明可以看Lec 24 简单来说（1，1，1）必然在N（trans A-I）中 所以 A-I 必然是singular  所以lambda包含1)
 1.特征值必然包含1
 2.其他特征值小于1
因此 Markov 系统说明最后总会有个steady state

Dr. Strang 举了个应用的例子 
 如果将两个城市的人口迁徙用Markov 表示，第一列表示第一个城市迁徙的和留下的人口比列 第二列类似 
 城市人口可以用一个向量表示 
 Dynamic： $u_{k}=A^ku_{0}$
 最后可以得出静态人口数量

### Fourier Series 

Projection 的特点 
对于在n维空间的向量v 可以通过orthonormal 拆分（就像坐标）
$v=x_{1}q_{1}+x_{2}q_{2}+\dots$
由于q （orthonormal ）的特性 可以得到如下等式
$q_{1}^Tv=x_{1}$

傅里叶级数是根据以上性质得到的
傅里叶级数将上面的orthonormal q 用方程来表示 
傅里叶级数的公式：
$f(x)=a_{0}+a_{1}\cos x+b_{1}\sin x+a_{2}\cos{2x}\dots$

那么方程的product怎么写呢？
很简单 方程的coherent：
$f(x)^Tg(x)=\int f(x)g(x) \, dx$ 在傅里叶级数中 x取值0-2pi 

怎么求a1
$f(x)^Tg(x)=\int_{0}^{2\pi} f(x)\cos x \, dx=a_{1}\int_{0}^{2\pi} (\cos x)^2 \, dx=a_{1}\pi$

# Symmetric Matrices 

A=tran A

当A对称时 特征值的两个特性：
1. 特征值都为实数 （A为实数矩阵）
2. 特征向量相互垂直 
Lec 证明了为什么特征值都是实数
特征向量为什么垂直：
![[Pasted image 20231115142214.png]]
## 对称矩阵的分解
$A=S\Lambda S^{-1}$
对称矩阵的分解
$A=Q\Lambda Q^{-1}=Q\Lambda Q^{T}$
$A=\lambda_{1} q_{1}q_{1}^T+\lambda_{2} q_{2}q_{2}^T+\dots$
$qq^T$为projection matrix 

## Positive Definitie Matrices
既然对称矩阵的所有lambda都是实数，那我们接下来想问的问题就是 是否为正？

**建议反复观看Lec 27 不太懂二阶倒数和特征值 的物理意义等问题 看到后面应该有新的理解**
products of eigenvalues = products of pivots = det
number of positive eigenvalues= numbers of positive eigenvectors

In Positive Definitie Matrices: 
all eigenvalues(pivots) are positive
all subdeterminants are positive 
$x^TAx>0$
### $x^TAx$
$Ax$ 线性  $x^TAx$ 为x1x2的二次型
Suppose
$$
A=
\begin{bmatrix}
a & b  \\
b & c
\end{bmatrix}
$$
$x^TAx$ =
$$
\begin{bmatrix}
x_{1} & x_{2}
\end{bmatrix}
\begin{bmatrix}
a & b  \\
b & c
\end{bmatrix}
\begin{bmatrix}
x_{1}  \\
x_{2}
\end{bmatrix}=a^2x_{1}+2bx_{1}x_{2}+c^2x_{2}
$$
即原点为最小值
这里的判断涉及微积分的知识

二次化：将判别式写为平方项
会发现与elimination的紧密联系 

**Matrix of second derivitive** 
判断极大值还是极小值还是鞍点
![[Pasted image 20231115171234.png]]
证明这个矩阵正定

几何意义 

特征向量 表示几何上的最快变化的方向和最慢方向
特征值 表示其具体的值
### Positive Semidefinite Matrix 
半正定矩阵
$x^TAx=0$ 即存在鞍点

detA=0
lambda 包含0

## Complex Matrix 

length 
In real matrices, the length is $\sqrt{q^Tq}$
但是在复数中，$(a+bi)(a+bi)=a^2+b^2+2abi$,得到的依旧是复数
因此在复数中，我们要取共轭
length $q^Hq$ H is called Hermitian 意思是取对称＋共轭
inner product 
$a^Hb$
symmetric
$A^H=A$ 不仅要求对称 同时要求共轭
perpendicular 
$a^Hb=0$
## FFT matix

computation complexity from $n^2$ to $n\log_{2}n$

$$

F(n)=
\begin{bmatrix}
1 & 1 & 1 & \dots. & 1 \\
1 & w & w ^2 & \dots  & w^{n-1} \\
1 & w^2 & w^4 & \dots & w^{2(n-1)} \\
\dots  \\
1 & w^{n-1} & w^{2(n-1)} & \dots & w^{(n-1)^2}
\end{bmatrix}

$$

$w^n=1$
$w=e^{\frac{2\pi}{n}i}=\cos{\frac{2\pi}{n}}+i\sin{\frac{2\pi}{n}}$
当考虑到复频域的性质 相差pi/2互为共轭 
FFT就是利用了这一性质 

例如F(64)的矩阵可以分解为两个32的矩阵加上permutation和共轭转置等操作
![[Pasted image 20231115161655.png]]
因此计算复杂度从64的平方变成了2乘上32的平方(两个F(32)的计算复杂度)加上32(Diagonal matrix 的计算复杂度 其他的I和permutation matrix 不需要计算)

### Rectangular Matrix 
A 当A不为长方形矩阵 
$A^TA$ symmetric positive definite
- 证明 $x^TA^TAx=(Ax)^TAx>0$ because Ax is a vector, and 为了保证不等于0 也就是null space只包含0 也就是 A 满秩 

### Why we need Positive Definite 

## Similar Matrix 
- 存在M $M^{-1}AM=B$ 则AB为similar matrix 

特征值分解 将A分解成对角矩阵
A与$\Lambda$ 为相似矩阵

- Similar Matrix包含相同的lambda
- Similar Matrix 特征向量相差$M^{-1}$倍
证明：
$$
Ax=\lambda x 
$$
$$
M^{-1}AMM^{-1}x=BM^{-1}x=\lambda M^{-1}x
$$

当特征值重复时
矩阵不能对角化 
Similar matrix 包含两类 
一种是对角矩阵 他的similar matrix 只包含自己
另一种是三角矩阵 有多个similar matrix

### Jordan Matrix 
Every quare matrix A is similar to a Jordan matrix J
$$
J=
\begin{bmatrix}
J_{1} &  &  &  \\
&J_{2} &  &  \\
& & J_{3}
\end{bmatrix}
$$
Jordan Matrix 为离对角矩阵”最近的“矩阵

# Singular Value Decomposition
SVD
$A=v\Sigma u^T$

SVD 做的事是将经过矩阵A变换的空间连接起来 
假设u代表变换前的空间的basis v代表变换后的空间的u对应的basis方向 
sigma代表两者之前的scalor缩放 
由于orthognomal $u^{-1}=u^T$

## $A^TA$
$A=v\Sigma u^T$
$A^T=u\Sigma^T v^T$
$A^TA=u\Sigma^T v^Tv\Sigma u^T=u\Sigma^2u^T=Q\Lambda Q^T$

同理$AA^T$也可以分解为$AA^T=v\Sigma^2v^T=Q\Lambda Q^T$

这两个方法可以快速求出u v (或者说特征向量)
但是σ  因为这里算的是平方 （特征值）所以符号问题还得看A是否翻转了basis

相比于特征值分解 奇异值分解不需要矩阵为方阵

## SVD和四个subspaces的关系
$A=v\Sigma u^T$
u1 ... ur orthonormal basis for A row space中 
ur ... un 在A的null space 中
v1 ... vr 在A 的collumn space 中
vm 在 trans(A)的null space 中

# Linear Transformation





