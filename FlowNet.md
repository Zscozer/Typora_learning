# FlowNet

## 1.光流

​	光流：视频图像的一帧中的代表同一对象(物体)像素点移动到下一帧的移动量，使用二维向量表示。

## 2.光流可视化

​	光流场是图片中每个像素都有一个x方向和y方向的位移，所以在上面那些光流计算结束后得到的光流flow是个和原来图像大小相等的双通道图像。
​	不同颜色表示不同的运动方向，深浅表示运动的速度。

![img](https://img-blog.csdn.net/20170504220106558?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	讲x和y转为极坐标，夹角(actan2(y,x))代表方向，极径(x和y的平方和开根号)代表位移大小，刚好用一下hsv的图像表示。上图的光流可以看到，红色的人在往右边动，那个蓝色的东西在往左上动
![这里写图片描述](https://img-blog.csdn.net/20180413155731994?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTMwMTA4ODk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 3.网络结构

​	由于卷积神经网络中一层有一层pooling后特征图越来越小，而最终预测的光流是要和原图大小相同，所以还需要放大(unconv)，所以网络整体上分为两个部分，一个缩小还有一个放大。（有一个缩小处理层，一个放大层）

### (1)缩小（卷积）（编码）部分

​	第一种缩小(卷积)方案是最朴素的方法的，就是将这一对图片的通道concat(合并)起来，网络结构图中可以看到输入的data层的channel是6.

![img](https://img-blog.csdn.net/20170504221123770?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	第二种方案，将一对图片分开处理，分别进入卷积层提取各自的特征，然后再对它们的特征进行匹配，寻找它们之间的联系，我们称之为 FlowNetCorr

![img](https://img-blog.csdn.net/20170504221450227?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	***\*黄色箭头\****所示即为比较两个特征图的操作。我们举例说明，设两个特征图分别为f1和f2，它们都为w*h*c的维度，即宽度为w，高度为h，通道数为c，比如以x1为中心的一块和以x2为中心的一块，它们之间的联系用以下公式计算，假设块长为K= 2k+1，大小为 K*K

![img](https://img-blog.csdn.net/20170504222007723?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	这一公式与神经网络的卷积操作是一样的，*<u>只不过普通的卷积是与filter（卷积核）进行卷积，而这个是两个块进行卷积，</u>*所以它没有可以训练的权重。

### (2)放大（解码）部分

![img](https://img-blog.csdn.net/20171007200233349)

​	这里的放大部分主要由逆卷积层（upconvolutional layer）组成，而上卷积层又由逆池化层（unpooling，与pooling操作相反，用来扩大feature map）和一个卷积层组成。

​	<u>为了更好地融合来自不同层的语义信息，每一层的输入除了上一层的输出外，还有利用上一层输出预测的“光流”和来自编码器对应层的特征。在拼接的过程中会使用上采样使所有的特征空间分辨率一致。</u>

​    我们对 feature maps 执行逆卷积操作（绿色箭头 deconv5），并且把它和之前收缩部分对应的feature map（灰色箭头 conv5_1）以及一个上采样的的光流预测（红色箭头 flow5）链接起来。每一步提升两倍的分辨率，重复四次，预测出来的光流的分辨率依然比输入图片的分辨率要小四倍。论文中说在这个分辨率时再接着进行双线性上采样的 refinement 已经没有显著的提高，所以采用优化方式：the variational approach。



## 4.数据集

### (1)Middlebury数据集

​	用于训练的图片对只有8对，从图片对中提取出的，用于训练光流的 ground truth 用四种不同的技术生成，位移很小，通常小于10个像素。

### (2)Kitti数据集

​	有194个用于训练的图片对，但只有一种特殊的动作类型，并且位移很大，视频使用一个摄像头和 ground truth 由3D激光扫描器得出，远距离的物体，如天空没法被捕捉，导致他的光流 ground truth 比较稀疏。

### (3)Mpi sintel数据集

​	是从人工生成的动画sintel中提取训练需要的光流 ground truth，是目前最大的数据集，每一个版本都包含1041个可一用来训练的图片对，提供的gt十分密集，大幅度，小幅度的运动都包含。
sintel数据集包括两种版本：
​	sintel final：包括运动模糊和一些环境氛围特效，如雾等。

​    sintel clean：没有上述final的特效。

### (4)Flying Chairs数据集

​	用于训练大规模的cnns，sintel的dataset依然不够大，所以作者他们自己弄出来一个flying chairs数据集。







# 5.Reference

[(12条消息) FlowNet 论文笔记_Reborn的博客-CSDN博客_flownet](https://blog.csdn.net/zjc8888888888/article/details/78171018)

[FlowNet和它的升级版 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/58509142)