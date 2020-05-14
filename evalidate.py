import os
import argparse
import numpy as np

import torch

from models.mdmnet import MDMNet
from utils.data_loader import get_loader
from utils.metrics import calculate_Accuracy
from utils.saver import load_pretrained
from utils.losses import calc_loss
from utils.utils import AverageMeter
from progress.bar import Bar

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


# import scikitplot as skplt


def eval(net, test_loader, device):
    '''
    Parameter:
        model: model after loading weights
        device: gpu or cpu?
    '''
    net.eval().to(device)

    bar = Bar('Processing validate', max=len(test_loader))
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    se_ = AverageMeter()
    sp_ = AverageMeter()
    auc_ = AverageMeter()
    f1_ = AverageMeter()

    draw_curve = True

    preds_prob_list = []
    preds_list = []
    gts_list = []

    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            pred_prob, d1_probs, d2_probs, d3_probs, d4_probs = net(data)  # after sigmoid function

            validate_loss = calc_loss(pred_prob, label, bce_weight=.5)
            # validate_loss += calc_loss(d2_probs, label, bce_weight=.5)
            # validate_loss += calc_loss(d3_probs, label, bce_weight=.5)
            # validate_loss += calc_loss(d4_probs, label, bce_weight=.5)
            loss_.update(validate_loss.item(), data.size(0))

            preds = torch.gt(pred_prob, .5).float()

            # Convert to numpy format
            preds = preds.cpu().data.numpy()[:, 0]
            label = label.cpu().data.numpy()[:, 0]
            pred_prob = pred_prob.cpu().data.numpy()[:, 0]

            pred_prob = pred_prob.reshape([-1])
            gt = label.reshape([-1])
            preds = preds.reshape([-1])
            CM = confusion_matrix(preds, gt)
            F1, Acc, Se, Sp, _ = calculate_Accuracy(CM)
            Auc = roc_auc_score(gt, pred_prob)

            if draw_curve:
                preds_prob_list.append(pred_prob)
                preds_list.append(preds)
                gts_list.append(gt)

            acc_.update(Acc, data.size(0))
            se_.update(Se, data.size(0))
            sp_.update(Sp, data.size(0))
            auc_.update(Auc, data.size(0))
            f1_.update(F1, data.size(0))

            bar.suffix = '{batch}/{size}) | Loss: {loss:.3f}'.format(
                batch=i + 1,
                size=len(test_loader),
                loss=loss_.avg
            )
            bar.next()
        bar.finish()

    print('Acc: %s  |  F1: %s |  Se: %s |  Sp: %s |  Auc: %s' % (
        str(acc_.avg), str(f1_.avg), str(se_.avg), str(sp_.avg), str(auc_.avg)))

    if draw_curve:
        # https://github.com/RanSuLab/DUNet-retinal-vessel-detection
        _preds_prob = np.asarray(preds_prob_list).reshape(-1)
        _preds = np.asarray(preds_list).reshape(-1)
        _gts = np.asarray(gts_list).reshape(-1)

        # Area under the ROC curve
        fpr, tpr, thresholds = roc_curve(_gts, _preds_prob)
        auc_roc = roc_auc_score(_gts, _preds_prob)
        plt.figure()
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.plot(fpr, tpr, 'darkorange', label='(AUC = %0.4f)' % auc_roc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC Curve', fontsize=14)
        plt.xlabel("FPR (False Positive Rate)", fontsize=14)
        plt.ylabel("TPR (True Positive Rate)", fontsize=14)
        plt.legend(loc="lower right")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # skplt.metrics.plot_roc(_gts, _preds_prob)
        plt.savefig("ROC.png")

        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(_gts, _preds_prob)
        precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        auc_prec_rec = np.trapz(precision, recall)
        plt.figure()
        plt.plot(recall, precision, 'darkorange', label='Area Under the Curve (AUC = %0.4f)' % auc_prec_rec)
        plt.title('Precision - Recall curve', fontsize=14)
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend(loc="lower right")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig("Precision_Recall.png")

    # according to f1 score
    return f1_.avg, loss_.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MDMNet Val Pipline')
    parser.add_argument('--data_path', type=str, default='./dataset/DRIVE',
                        help='dir of the all img')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the num of img in a batch')
    parser.add_argument('--resize', type=tuple, default=(512, 512),
                        help='the train img size')
    parser.add_argument('--dataset', type=str, default='DRIVE',
                        help="dataset's name")
    parser.add_argument('--n_class', type=int, default=1,
                        help='the channel of out img, decide the num of class')
    parser.add_argument('--gpu_avaiable', type=str, default='0',
                        help='the gpu used')
    parser.add_argument('--checkpoints', type=str,
                        default='./weights/DRIVE1/model_best.pth', help="weight's path")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading test data
    test_loader = get_loader(
        args.data_path, args.resize, args.batch_size, mode='test', dataset_name=args.dataset)

    # Load model
    net = MDMNet(input_size=args.resize, n_classes=args.n_class)
    net, _ = load_pretrained(net, args.checkpoints)

    eval(net, test_loader, device)
