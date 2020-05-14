import torch
import torch.nn as nn
import numpy as np


def f_score(pr, gt, beta=1, eps=1e-7, threshold=.5):
    """dice score(also referred to as F1-score)"""
    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class FscoreMetric(nn.Module):
    __name__ = 'f-score'

    def __init__(self, beta=1, eps=1e-7, threshold=.5):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.beta = beta

    def forward(self, y_pr, y_gt):
        return f_score(y_pr, y_gt, self.beta, self.eps, self.threshold)


def calculate_Accuracy(confusion):
    # Measure metric
    confusion = np.asarray(confusion)
    pos = np.sum(confusion, 1).astype(np.float32)  # 1 for row
    res = np.sum(confusion, 0).astype(np.float32)  # 0 for coloum
    tp = np.diag(confusion).astype(np.float32)
    IU = tp / (pos + res - tp)

    Acc = np.sum(tp) / np.sum(confusion)
    Se = confusion[1][1] / (confusion[1][1] + confusion[0][1])  # tp / (tp + fn)
    Sp = confusion[0][0] / (confusion[0][0] + confusion[1][0])  # tn / (tn + fp)
    PPV = confusion[1][1] / (confusion[1][1] + confusion[1][0])  # tp / (tp + fp)

    # is the harmonic mean of precision and sensitivity
    F1 = (2 * PPV * Se) / (PPV + Se)

    return F1, Acc, Se, Sp, IU
