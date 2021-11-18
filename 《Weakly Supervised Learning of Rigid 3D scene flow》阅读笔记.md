# 《Weakly Supervised Learning of Rigid 3D scene flow》阅读笔记

## 1.Abstract

​	这篇论文提出了一种场景流估计的方法，其框架主要在**object-level**(物体级别)上进行推理学习的。因为不是以点为最小单位，所以该模型在标签不全的弱监督学习上效果很好。

​	在弱监督层面，模型最后的输出结果，输出了每个点的流（低层次上），刚体级别的对整个景的理解（高层次上）。

​	同时，论文还提出了一种测试时间优化器优化了刚体场景流预测



### 知乎评价：

​	**刚性3D场景流弱监督学习方法。提出一种数据驱动的场景流估计算法，通过将流整合到更高层次的场景抽象中，以多刚体运动的形式来放松密集监督，核心在于一个深度架构，利用刚性场景流问题的几何形状将归纳偏置引入到网络中，考虑3D场景流与其他3D任务相结合，在对象层面进行推理，结果是一个先进的流估计网络，还输出了一个简洁的动态场景表示。进一步提出了一个测试时的优化方法来完善预测的刚性场景流。该方法由一个新颖的、灵活的、场景流动骨干支持，可用于解决各种任务。在四个不同的自动驾驶数据集上展示了方法的有效性和泛化能力。**



## 3.Method

​	连续两帧三维场景X，Y表示两个点云。存在一个V属于x存在的域，使得X+V = Y



## 4.NetWork

![img](https://3dsceneflow.github.io/assets/network_architecture.jpg)

将连续两帧点云传入模型，分别进入backbone F和G， 两个图像出来后分成两步，第一进入 bsh(background segmentation head) 、第二合成为w。



F和G所代表的是一种结构类似于U-Net的场景分割网络，将输入的两帧图像进行前景、背景的分割

W代表将F和G的结果做一个concat（合并），卷积，再做放大（参考一下flownet做出光流估计）就可以得到场景流估计

Scene flow head:



ego-motion head: 

计算两帧点云对应关系用到了Sinkhorn算法，假定一个相似性矩阵M来表示结果，用Sinkhorn算法来逼近解出最优M

计算出的addintity matrix M(相似度矩阵m)，这个M代表的是 X与Y中点的对应程度（相似性）。

得到两帧中点的对应关系后，通过Kabsch算法可以恢复自运动过程，即做出自运动的估计。



网络的输出：（1）自运动和聚类的转换参数（transformation parameters）

​						（2）背景的掩膜（masks）和点层面的无约束场景流

​						对应到三层输出：

​						Instance clustering: 聚类的转换参数 {Tk}  

​						Scene flow head:  Scene flow

​						Ego-motion head:自运动转换参数T_ego

​						

​						masks在scene flow 、ego-motion中已经实现

网络的应用过程



## 5.Datasets

### SemanticKITTI

​	![img](https://img-blog.csdnimg.cn/20200403120026469.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNzMxMzMy,size_16,color_FFFFFF,t_70)

​	这个数据集是用于训练弱监督网络的。而且这个数据集是物体级别的，数据集中不区分有人开的车和车，但区分车和单车（摩托车）；区分移动和不移动的任何车。总的来说，数据集分为28类，其中6类被指定归属于移动或非移动。一个离群类包含了因为反光或其他原因造成的错误雷达测量。

### stereoKITTI

​	

## 疑问

1.如何提取自运动信息（根据背景的图像配准？）

​	答：看了FLOT，我发现上图经过Ego-motion head 后应该要输出背景的匹配（点云匹配）

2.聚类应该是将前景的小车以每个为单位聚类出来，而场景流信息虽然是前景的，不需要聚类出每个小车，为什么还要聚类

3.聚类出的小车（object级）、场景流、自运动信息为什么要结合在一起，这个object-level rigid transformation（物体级别的刚性变换）是什么意思

4.此篇论文想要实现的是不是从物体级别做场景流估计然后再泛化回每个点的场景流估计？

​	答：没错

**5.在源码ego-motion head 中使用了一个Sinkhorn迭代模块，这个是干嘛的

​	答：Sinkhorn算法实际上是最优传输的一种算法,运用的最优传输是为了让两帧背景的点云进行匹配





## 代码（model）

```python
# Ego-motion head
class EgoMotionHead(nn.Module):
    """
    Class defining EgoMotionHead
    """

    def __init__(self, add_slack=True, sinkhorn_iter=5):
        nn.Module.__init__(self)

        self.slack = add_slack
        self.sinkhorn_iter = sinkhorn_iter

        # Affinity parameters
        self.beta = torch.nn.Parameter(torch.tensor(-5.0))
        self.alpha = torch.nn.Parameter(torch.tensor(-5.0))

        self.softplus = torch.nn.Softplus()


    def compute_rigid_transform(self, xyz_s, xyz_t, weights):
        """Compute rigid transforms between two point sets

        Args:
            a (torch.Tensor): (B, M, 3) points
            b (torch.Tensor): (B, N, 3) points
            weights (torch.Tensor): (B, M)

        Returns:
            Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
        """

        weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
        centroid_s = torch.sum(xyz_s * weights_normalized, dim=1)
        centroid_t = torch.sum(xyz_t * weights_normalized, dim=1)
        s_centered = xyz_s - centroid_s[:, None, :]
        t_centered = xyz_t - centroid_t[:, None, :]
        cov = s_centered.transpose(-2, -1) @ (t_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_s[:, :, None] + centroid_t[:, :, None]

        transform = torch.cat((rot_mat, translation), dim=2)

        return transform

    def sinkhorn(self, log_alpha, n_iters=5, slack=True):
        """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column 			sum to <=1
        Args:
            log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
            n_iters (int): Number of normalization iterations
            slack (bool): Whether to include slack row and column
            eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.
        Returns:
            log(perm_matrix): Doubly stochastic matrix (B, J, K)
        Modified from original source taken from:
            Learning Latent Permutations with Gumbel-Sinkhorn Networks
            https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
        """

        # Sinkhorn iterations

        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)


        log_alpha = log_alpha_padded[:, :-1, :-1]

        return log_alpha


    def forward(self, score_matrix, mask, xyz_s, xyz_t):

        affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)

         # Compute weighted coordinates
        log_perm_matrix = self.sinkhorn(affinity, n_iters=self.sinkhorn_iter, slack=self.slack)

        perm_matrix = torch.exp(log_perm_matrix) * mask
        weighted_t = perm_matrix @ xyz_t / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

        # Compute transform and transform points
        #transform = self.compute_rigid_transform(xyz_s, weighted_t, weights=torch.sum(perm_matrix, dim=2))
        R_est, t_est, _, _ = kabsch_transformation_estimation(xyz_s, weighted_t, weights=torch.sum(perm_matrix, dim=2))
        return R_est, t_est, perm_matrix

```
