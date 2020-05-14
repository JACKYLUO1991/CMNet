import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss
    """
    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce_out = F.binary_cross_entropy(prediction, target)
    dice_out = dice_loss(prediction, target)

    loss = bce_weight * bce_out + (1 - bce_weight) * dice_out

    return loss


# def cross_entropy(logits, labels):
#     return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

# class EdgeAwareLoss(nn.Module):
#     '''
#     A new novel edge aware segmention loss function
#     Paper: Pyramid Feature Attention Network for Saliency detection
#     https://github.com/CaitinZhao/cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection
#     '''
#
#     def __init__(self):
#         super(EdgeAwareLoss, self).__init__()
#         # out_channel, in_channel, height, weight
#         laplace_operator = torch.FloatTensor(
#             [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view([1, 1, 3, 3])
#         self.laplace = nn.Parameter(data=laplace_operator, requires_grad=False)
#
#     def forward(self, prediction, target, ratio=.5):
#         # Prediction parameters need to be added after sigmoid function
#         prediction_edge = self.__laplace_func(prediction)
#         target_edge = self.__laplace_func(target)
#         # Loss function
#         edge_loss = cross_entropy(
#             prediction_edge, target_edge)
#         segmentation_loss = dice_loss(prediction, target)
#
#         return (1 - ratio) * edge_loss + ratio * segmentation_loss
#
#     def __laplace_func(self, x):
#         x = F.conv2d(x, self.laplace, stride=1, padding=1)
#         x = torch.abs(torch.tanh(x))
#         return x
