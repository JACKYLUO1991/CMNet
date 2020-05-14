# CMNet
## CHINESE INTRODUCTION
摘 要 ： 目的 视网膜血管自动分割技术已经成为临床医学筛选和眼病诊断的重要工具。 为了追求准确性，模型通常会
忽略实时性，从而难以满足临床需求。 此外， 受到低对比度和视网膜病变的影响， 细粒度血管的精准分割面临着巨大挑战。
考虑到以上几点，本文提出了一种新颖的结合速度和准确性的非对称性视网膜血管分割方法。 方法 考虑到可变形卷积能
够提取复杂多变的血管结构特征，混合深度可分离卷积中的大核在增大感受野的同时能够提高分割精度。结合两者的优势，
本文提出了一种混合瓶颈模块， 并将其广泛应用于模型中。 此外，本文还提出了一种自适应层级融合方法以进一步提高模
型的空间映射能力。 结果 使用多种评估指标对训练结果进行综合评定，模型在 3 个数据集上与最新的方法进行了比较。
在 DRIVE 数据集上精度和 AUC（area under curve） 分别达到 0.9662 和 0.9840；在 CHASE _DB1 数据集上， 相比于
Octave-UNet 模型， 灵敏度提高了 2.52%； 在 HRF 数据集上， 相比于 Octave-UNet 模型， 在灵敏度、 F1（F1 score） 和 AUC
指标上分别提高了 0.18%、 0.01%和 0.02%。 另外， 在输入长宽都为 512 的情况下模型参数仅有 0.47 M， 同时在单卡 Tesla
V100 GPU 下帧率可以达到 33 FPS。 以上实验结果表明所提出的算法相对于目前主流方法来说既高效又有效。 结论 本文
针对视网膜血管区域分割难度大及时间复杂度较高的问题， 提出了一种紧凑且高效的非对称混合网络模型和自适应层级融
合机制。 结果表明该模型具有良好的性能和极高的应用价值。


## Pay Attention
Once the paper is accepted, I will disclose all the training codes and detailed training procedures.

## Weights and training details
Baidu web disk：https://pan.baidu.com/s/11-7SyBmIeue-dmPQa9P3aA  
password：u8q3

## Major introduction！！！！
The new version has achieved good results on multiple datasets.
