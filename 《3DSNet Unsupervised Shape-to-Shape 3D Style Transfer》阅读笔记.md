# 《3DSNet: Unsupervised Shape-to-Shape 3D Style Transfer》阅读笔记

​	此篇论文围绕两个任务三维重建和三维风格迁移

## 网络结构

![3dsnet](D:\typora\Typora\document\picture\3dsnet.png)

### Encoder

​	使用pointnet对点云特征进行提取，将x1,x2映射到内容空间和风格空间（潜空间）

### Decoder

​	实现了三维重建，如何将风格和内容叠加呢，使用了Adainnorm

#### 	AdaNorm

##### 	学习AdaNorm之前先学习InNorm和Conditional InNorm

<img src="D:\typora\Typora\document\picture\In and Conditional InNorm.png" alt="In and Conditional InNorm" style="zoom: 50%;" />

​	其中r和B代表旋转和平移变量

##### 	随后学习AdaInNorm

​	

<img src="D:\typora\Typora\document\picture\adain normalization.png" alt="adain normalization" style="zoom:50%;" />  

​	Adain、In实际上是对内容编码x和风格编码y做了一个对齐，式子中 将x做了正太分布的标准化，然后对齐y空间，这样内容就和风格叠加了,但是adain中没有r、B，说明Adain的仿射参数是不需要进行学习的，AdaIn接受两个信息源：内容输入x和风格输入y，将x的通道级(channel-wise)均值和标准差匹配到y的通道级均值和标准差上



Adain在feature map层面上通过改变特征的数据分布来实现风格迁移，这样做计算开销和存储开销都较小，且易实现



#### 那AdaNorm呢

 本文中提到的AdaNorm（自适应归一化），在本文中 是通过比较InNorm和BN来确定那种Norm是合适的，通过AtlasNet和MeshFlow上的应用来选择

#### AtlasNet and MeshNet

作者在decoder部分是采用了他们两个网络的decoder结构比较，采用3DSNet+AtlasNet/Meshflow decoder

AtlasNet

通过输入二维图像或者点云生成3d模型  (e)3d打印出的结果 在本文中我们只需要它的decoder

![AtlasNet](D:\typora\Typora\document\picture\AtlasNet.png)

![AtlasNet-structure](D:\typora\Typora\document\picture\AtlasNet-structure.png)

这个AtlasNet实际上就是比原来这个latent shape输入多了一个单位正方形[0,1]^2，将这个单位正方形上均匀采样的二维点作为附加输入，并使用它在曲面上生成一个点，因此输出是平面的连续图像



Meshflow

![image-20211020161443776](C:\Users\ZHG\AppData\Roaming\Typora\typora-user-images\image-20211020161443776.png)

​	Neural Mesh Flow由三个变形块组成，它们基于目标形状MT的形状嵌入z在球面网格顶点上进行逐点流动。时间实例0< T1< T2<  T表示通过第一个变形块将球形网格变形为粗椅表示Mp0。进一步的变形块进行细化，生成细化的网格Mp1, Mp2。

​	Meshflow步骤：1.首先，对目标形状MT进行编码，从其表面均匀采样N个点，并将它们输入PointNet编码器，得到尺寸为k的全局形状嵌入z。

2.以形状嵌入z为条件，Node block将模板球的顶点向目标形状微分流动

3.Instance Normalization对NODE输出进行非均匀缩放，以简化跨类别训练。

4.细化流程提供了质量的逐步改进

在3dsnet中，这个z就有内容编码代替了，Ms就有风格编码代替了



### Discriminator

​	基于pointnet的判别器模型



# 度量指标

### 3D-LPIPS

利用PointNet对两类点云做特征提取，做归一化处理，然后计算他们之间的欧式距离得到的度量

### STS(Style Transfer Score)

<img src="D:\typora\Typora\document\picture\STS.png" alt="STS" style="zoom: 67%;" />

# 疑问

1.为什么要用到循环一致性损失

2。Munit还有3DSNet-M

3.chamfer distance



