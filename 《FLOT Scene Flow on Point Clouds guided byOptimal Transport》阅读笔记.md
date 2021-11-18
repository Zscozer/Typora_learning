# 《FLOT: Scene Flow on Point Clouds guided by
Optimal Transport》阅读笔记

此篇论文是weakly那篇的比较论文

## 1.Abstract

​	此片论文是处理两帧点云数据匹配问题的。主要是利用最佳传输（Optimal Transport）来做点云匹配。

## 2.Introduction

​	FLOT方法区别于以前的方法，只需要点云的三维坐标信息即可，而且FLOT是直接使用卷积核对点云进行操作。

​	FLOT方法完成场景流估计的步骤：第一步，找出输入点云中点的潜在对应关系，第二步，利用这些对应关系来估计流。

​	FLOT方法将用在图匹配中的最佳传输方法应用于点云中，去找出点之间的潜在对应联系

​	具体步骤：

​	第一步，以连续两帧点云作为输入，使用卷积提取点云特征，并将这些特征用于计算传输代价（transport cost），两点之间的代价暗示了他们之间的对应关系。

​	第二步，利用这些潜在对应关系，通过线性插值（linear interpolation）去得到场景流估计。

​	以上步骤可以通过残差网络进行优化

## 3. Related Work

### 点云上的深度场景流估计

​	FLOT提取场景流的方法是基于pointnet++（读一下）,且FLOT中的optimal transport模块是独立于卷积模块的，即其他的卷积方法可以替代pointnet++中的卷积方法

​	FLOT仍然用的是全监督。

### 基于最优传输的图匹配

​	图匹配过程简述：使用图拉普拉斯算子（Laplacian）将图映射到表示图结构的多维高斯分布，然后，这些分布之间的瓦瑟斯坦距离（Wasserstein distance ）被用来衡量图的相似性，并允许人们匹配图之间的节点。



​	FLOT中运用的图匹配思想：FLOT采用Wasserstein distance。通过将每个点连接到其最近邻点的方式来构造点云的图结构，然后提出了一种方式来训练网络：提取每个点的深度特征，并使用这些特征来匹配最佳传输模块中的点云之间的点。

## 4. Method

### step1:寻找点之间的潜在联系

<img src="D:\typora\Typora\document\picture\FLOTbackbone.png" style="zoom:80%;" />

主要是OT模块，他的作用是通过p、q提取特征计算出两帧点云中每个点对应的信息，通过此可计算出场景流f^,然后再通过h进行提炼获得f_est



p + f = Pq----- p:点云p的信息

​						f:场景流f

​						P:置换矩阵

​						q:点云q的信息



dtw