# 基于紧凑混合网络的视网膜血管自动分割
# Automatic segmentation of retinal vessel via compact mixed network（CMNet）

## INTRODUCTION（介绍）
摘  要 ：针对视网膜血管分割困难及时间复杂度高等问题, 本文提出一种可以兼顾分割速度和准确度, 同时结构非对称的视网膜血管分割模型, 即紧凑混合网络 (Compact Mixed Network, CMNet). 首先, 由于可变形卷积能够提取复杂多变的血管结构, 并且混合深度卷积中的大核在增大感受野的同时能够改善分割质量, 本文在此基础上提出一种轻量级混合瓶颈模块; 其次, 采用自适应层融合方法进一步提高了模型的空间映射能力; 最后, 对血管分割性能进行了定量和定性分析. 算法在 DRIVE、 CHASE_DB1 和 HRF 三个基准数据集上的 AUC 指标分别为 0.9840、0.9879 和 0.9853. 以上结果表明, 提出的模型能够得到高精度的分割结果. 此外, 在输入分辨率为 512×512 下, 模型在单张 V100 显卡上帧率可达 33 FPS, 进一步表明该模型适用于临床快速部署.

Abstract: To address the difficulty and high time-complexity of retinal vessel segmentation, an asymmetric model called CMNet(Compact Mixed Network) is proposed, which is capable of achieving trade-off between speed and accuracy. Firstly, considering the ability of deformable convolution to extract complex and variable vascular structures and large
kernel in mixed depthwise convolution can further improve segmentation quality while increasing the receptive field, we propose a lightweight mixed bottleneck module. Then, an adaptive feature layer fusion is proposed to further improve the spatial mapping capability of the model. Finally, the vessel segmentation performance are analyzed quantitatively and qualitatively. The AUC metrics are 0.9840, 0.9879 and 0.9853 for DRIVE, CHASE_DB1 and HRF benchmark datasets, respectively, indicating that the proposed algorithm is able to obtain highly accurate segmentation results. Furthermore, with an input resolution of 512×512, the model achieves a frame rate of 33 FPS on a single V100 GPU, which further indicates its suitability for rapid clinical deployment.


## PAY ATTENTION （需要注意的是）
现已被控制与决策期刊接收！
It has been accepted by the Journal of Control and Decision.

## Main contribution（主要贡献）
The new version has achieved good results on multiple datasets.

## Author team（作者团队）
Ling Luo, Dingyu Xue，Xinglong Feng



