# 一元线性回归模型
## 算法原理
### 问题：通过【发际线的高度】预测【计算机水平】的例子
![[Pasted image 20231023214217.png|200]]
### 一元线性回归（均方误差）
$$f ( x ) = w _ { 1 } x _ { 1 } + b$$
### 二值离散特征【颜值】（好看：1，不好看：0）
$$f ( x ) = w _ { 1 } x _ { 1 } + w _ { 2 } x _ { 2 } + b$$
### 有序的多值离散特征【饭量】（小：1，中：2，大：3）
$$f ( x ) = w _ { 1 } x _ { 1 } + w _ { 2 } x _ { 2 } + w _ { 3 } x _ { 3 } + b$$
### 无序的多值离散特征【肤色】（黄：\[1,0,0]，黑：\[1,0,0]，白：\[1,0,0]）
$$f ( x ) = w _ { 1 } x _ { 1 } + w _ { 2 } x _ { 2 } + w _ { 3 } x _ { 3 } + w _ { 4 } x _ { 4 } + w _ { 5 } x _ { 5 } + w _ { 6 } x _ { 6 } + b$$
## 线性回归的解决策略
### 最小二乘法估计
$$E _ { ( w , b ) } = \sum _ { i = 1 } ^ { m } ( y _ { i } - f ( x _ { i } ) ) ^ { 2 }$$ $$= \sum _ { i = 1 } ^ { m } ( y _ { i } - ( w x _ { i } + b ) ) ^ { 2 }$$ $$= \sum _ { i = 1 } ^ { m } ( y _ { i } - w x _ { i } - b ) ^ { 2 }$$再采用去求得w和b $$arg\min$$
### 极大似然数估计
用途：估计概率分布的参数值
方法：对于离散型（连续型）随机变量X,假设其概率质量函数为P(x;θ)(概率密度
函数为p(x;θ),其中θ为待估计的参数值（可以有多个）。
现有x1,x2,x3,……,xn是来自X的n个独立同分布的样本，它们的联合概率为
$$L ( \theta ) = \prod _ { i = 1 } ^ { n } P ( x _ { i } ; \theta )$$
已知量：xi
未知量：θ
故可以求出θ，L(θ)为样本的似然函数

#### 例题：
现有一批观测样本c1,c2,x3,……,xn,假设其服从某个正态分布X~N(u,σ²),其中u,σ为待估计的参数值，请用极大似然估计法估计u,σ
【解】：
S1：写出随机变量X的概率密度函数
$$p ( x ; k , o ^ { 2 } ) = \frac { 1 } { \sqrt { 2 \pi } e }  ( - \frac { ( x - k ) ^ { 2 } } { 2 o ^ { 2 } } )$$
S2：写出似然函数
$$L ( N , o ^ { 2 } ) = \prod _ { i = 1 } ^ { n } p ( x _ { i j } , u , o ^ { 2 } ) = \prod _ { i = 1 } ^ { n } \frac { 1 } { \sqrt { 2 \pi } o x p } ( - \frac { ( x _ { i } - H ) ^ { 2 } } { 2 g ^ { 2 } } )$$
S3：求出是得L(u,σ²)取得最大值的u,σ

#### 计算技巧lnab=lna+lnb
$$\ln L ( u , o ^ { 2 } )= \sum _ { i = 1 } ^ { n } \ln \frac { 1 } { \sqrt { 2 \pi } o } \operatorname { e x p } ( - \frac { ( x _ { i } - H ) ^ { 2 } } { 2 o ^ { 2 } } )$$

#### 极大似然估计算线性回归
##### S1
$$y = w x + b + \epsilon $$
其中∈为不受控制的随机误差，通常假设其服从均值为0的正态分布ε~N(0,σ2)（高斯提出的，也可以用中心极限定理解释），所以ε的**概率密度函数**为

$$p(\epsilon)= \frac{1}{\sqrt{2 \pi}\sigma}exp(- \frac{\epsilon ^{2}}{2 \sigma ^{2}})$$
将ε用y-(wx+b)等价替代
$$p(y)= \frac{1}{\sqrt{2 \pi}\sigma}exp(- \frac{(y-(wx+b))^{2}}{2 \sigma ^{2}})$$
##### S2

上式变成y~N(wx+b,σ²)，其**似然函数**为：
$$L(w,b)= \prod _{i=1}^{m}p(y_{i})= \prod _{i=1}^{m}\frac{1}{\sqrt{2 \pi}\sigma}exp(- \frac{(y_{i}-(wx_{i}+b))^{2}}{2 \sigma ^{2}})$$
$$\ln L(w,b)= \sum _{i=1}^{m}\ln \frac{1}{\sqrt{2 \pi \sigma}}+ \sum _{i=1}^{m}\ln exp(- \frac{(y_{i}-wx_{i}-b)^{2}}{2 \sigma ^{2}})$$
$$\ln L(w,b)=m \ln \frac{1}{\sqrt{2 \pi}\sigma}- \frac{1}{2 \sigma ^{2}}\sum _{i=1}^{m}(y_{i}-wx_{i}-b)^{2}$$

由于m，b为常数，求最大化ln L（w，b）等价于最小化后半部分
$$(w^{*},b^{*})=arg \min \sum _{(w,b)}^{m}(y_{i}-wx_{i}-b)^{2}$$


$$$$



## 求解w和b
### 思路
求解w和b本质是一个多元函数求极值问题，更是求凸函数求最值问题
推导思路：
1. 证明
	$$E_{(w,b)}= \sum _{i=1}^{m}(y_{i}-wx_{i}-b)^{2}$$是关于w和b的凸函数

2. 用凸函数求最值的思路求出w和b

### 定理
设D ∈ R是非空开凸集，f：D∈R → R，且f(x)在D上二阶连续可微，如果f(x)的Hessian（海塞）矩阵在D上是半正定的，则f（x）是D上的凸函数。
因此，只需证明
$$E_{(w,b)}= \sum _{i=1}^{m}(y_{i}-wx_{i}-b)^{2}$$
的Hessian（海塞）矩阵
![[Pasted image 20231023190420.png|300]]
是半正定，那么E(w,b)就是关于w和b的凸函数。

### 半正定
半正定即为矩阵其特征值均不小于0

### 凸充分性定理：
若f：R → R是凸函数，且f（x）一阶连续可微，则x\*是全局解的充分必要条件
$$\nabla f(x^{*})=0$$
所以
$$\nabla E_{(w,b)}=0$$
的点即为最小值，即为
![[Pasted image 20231023205502.png|200]]

### 继续对w和b求偏导
b偏导：
$$\frac{\partial E_{(w,b)}}{\partial b}$$
$$b= \frac{1}{m}\sum _{i=1}^{m}y_{i}-w \cdot \frac{1}{m}\sum _{i=1}^{m}x_{i}= \overline{y}-w \overline{x}$$
w偏导：
$$\frac{\partial E_{(w,b)}}{\partial w}$$
把b带入得
$$w= \frac{\sum _{i=1}^{m}y_{i}x_{i}- \overline{y}\sum _{i=1}^{m}x_{i}}{\sum _{i=1}^{m}x_{i}^{2}- \overline{x}\sum _{i=1}^{m}x_{i}}$$
$$w= \frac{\sum _{i=1}^{m}y_{i}x_{i}- \overline{x}\sum _{i=1}^{m}y_{i}}{\sum _{i=1}^{m}x_{i}^{2}- \frac{1}{m}(\sum _{i=1}^{m}x_{i})^{2}}= \frac{\sum _{i=1}^{m}y_{i}(x_{i}- \overline{x})}{\sum _{i=1}^{m}x_{i}^{2}- \frac{1}{m}(\sum _{i=1}^{m}x_{i})^{2}}$$

### 拓展：
凸凹集定义，不同于《高等数学》的凸凹，同《最优化基础理论与方法》的凸凹

#### 凸集：
设集合DCR^n， 如果对任意x，y属于D与任意α属于\[0,1] ,有
$$\alpha x + ( 1 - \alpha ) y E D$$
则集合D为凸集。
#### 凸集集合意义：
若两个点属于此集合，则这两点连线上的任意一点均属于此集合。
视图：![[Pasted image 20231023173416.png|100]]

#### 梯度（多元函数的一阶导数）
设n元函数f(x)对自变量x=（x1,x2....,xn）^T的各分量xi的偏导数$$\frac{\partial f(x)}{\partial x_{i}}$$都存在，则称函数f(x)在x处一阶可导，并称向量
![[Pasted image 20231023184331.png|200]]

为函数f(x)在x出的一阶可导或梯度。

#### Hessian（海塞）矩阵（多元函数的二阶导数）
设n元函数f（x）对自变量x=（x1,x2,...,xn）^T的各分量xi的二阶偏导数
$$\frac{\partial ^{2}f(x)}{\partial x_{i}\partial x_{j}}$$
(i=1,2,...,n;j=1,2...,n)都存在，称函数f(x)二阶可导数，并称矩阵
![[Pasted image 20231023185140.png|300]]


## BTW：by the way
### 机器学习三要素：
1. 模型：根据具体问题，确定假设空间
2. 策略：根据评价标准，确定选取最优模型的策略（通常会产出一个"loss函数")
3. 算法：求解loss函数，确定最优模型

### 一元线性回归
- 模型：
	- 具体问题：发际线高度预测计算机水平
	- 假设空间：经验和数据形态是直线空间
	- 

-  策略：
	- 找到那条直线：
	- 均方误差最小（最小二乘法思路），得出loss函数
	- 极大似然数，误差呈正态分布，得出loss函数

- 算法：（一元为闭式解）
	- 求凸函数

拓展：大多数非线性都是非闭式解

## Prob
![[Pasted image 20231023171049.png]]
exp = e^
## 拓展阅读：
靳志辉.《正态分布的前世今生》
王燕军.《最优化基础理论与方法》



# 多元线性回归
## 由最小二乘法导出loss函数Eω^
#### S1
将ω和b组合称ω^：
$$f(x_{i})=w^{T}x_{i}+b$$
ω∈R^d，x∈R^i
$$f(x_{i})=(w_{1}w_{2}\cdots w_{d}) \begin{pmatrix} {x_{i1}} \\ {x_{i2}} \\ {\cdots} \\ {x_{id}} \end{pmatrix}+b $$
$$f(x_{i})=w_{1}x_{i1}+w_{2}x_{i2}+ \ldots +w_{d}x_{id}+b$$
将b扩充进入ω中，且在x列向量中加入1，则ω∈R^d+1，x∈R^id
$$f(x_{i})=w_{1}x_{i1}+w_{2}x_{i2}+...+w_{d}x_{id}+w_{d+1}\cdot 1$$

#### S2
新式子
$$f(x_{i})=(w_{1}w_{2}\cdots w_{d+1}) \begin{pmatrix} {x_{i1}} \\ {x_{i2}} \\ {\cdots} \\ {x_{id}} \\ {1} \end{pmatrix}  +b $$
$$f(\widehat{x}_{i})= \widehat{w}^{T}\widehat{x}_{i}$$
#### S3
由最小二乘法得到：
$$E_{ \widehat{w}}= \sum _{i=1}^{m}(y_{i}-f(\widehat{x}_{i}))^{2}= \sum _{i=1}^{m}(y_{i}- \widehat{w}^{T}\widehat{x}_{i})^{2}$$
再将其向量化，平方最换为行列向量相乘a²+b²=\[a,b]列\[a,b]
$$E_{ \widehat{w}}= (y_{1}- \widehat{w}^{T}\widehat{x}_{1}y_{2}- \widehat{w}^{T}\widehat{x}_{2}\cdots y_{m}- \widehat{w}^{T}\widehat{x}_{m}) \begin{pmatrix} \ (y_{1}- \widehat{w}^{T}\widehat{x}_{1})^{2} \\ (y_{2}- \widehat{w}^{T}\widehat{x}_{3})^{2} \\ …… \\ (y_{m}- \widehat{w}^{T}\widehat{x}_{m})^{2} \end{pmatrix}$$


## 求解ω^
$$\widehat{w}$$
