import os
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TTF

from models.hybridnet import HybridNet
from utils.data_loader import get_loader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from utils.metrics import calculate_Accuracy
from utils.saver import load_pretrained

import warnings
warnings.filterwarnings("ignore")


def eval(model, data_loader, device):
    '''
    Parameter:
        model: model after loading weights
        device: gpu or cpu?
    '''
    model.eval().to(device)

    ACC = []
    SE = []
    SP = []
    AUC = []
    IOU = []
    F1 = []

    with torch.no_grad():

        for i, (data, label) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            label = label.to(device)

            _, pred_prob, _, _, _ = net(data)  # after sigmoid function
            preds = torch.gt(pred_prob, .5).float()

            # Convert to numpy format
            preds = preds.cpu().data.numpy()
            label = label.cpu().data.numpy()
            pred_prob = pred_prob.cpu().data.numpy()

            pred_prob = pred_prob.reshape([-1])
            gt = label.reshape([-1])
            preds = preds.reshape([-1])
            CM = confusion_matrix(preds, gt)
            f1, Acc, Se, Sp, Iou = calculate_Accuracy(CM)
            F1.append(f1)
            ACC.append(Acc)
            SE.append(Se)
            SP.append(Sp)
            IOU.append(Iou[1])

            Auc = roc_auc_score(gt, pred_prob)
            AUC.append(Auc)

    print('Acc: %s  |  F1: %s |  Se: %s |  Sp: %s |  Auc: %s  | Iou: %s' % (str(np.mean(np.stack(ACC))), str(np.mean(np.stack(F1))), str(np.mean(np.stack(
        SE))), str(np.mean(np.stack(SP))), str(np.mean(np.stack(AUC))), str(np.mean(np.stack(IOU)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CEDenseNetAttention Testing Demo')
    parser.add_argument('--data_path', type=str, default='./dataset/DRIVE',
                        help='dir of the all img')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the num of img in a batch')
    # parser.add_argument('--crop_size', type=tuple, default=(544, 544),
    #                     help='the crop img size')
    parser.add_argument('--resize', type=tuple, default=(512, 512),
                        help='the train img size')
    parser.add_argument('--n_class', type=int, default=1,
                        help='the channel of out img, decide the num of class')
    parser.add_argument('--gpu_avaiable', type=str, default='0',
                        help='the gpu used')
    parser.add_argument('--checkpoints', type=str,
                        default='./weights/model_best.pth', help="weight's path")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading test data
    # test_loader = get_loader(
    #     args.data_path, args.crop_size, args.resize, args.batch_size, mode='test')
    test_loader = get_loader(
        args.data_path, None, args.resize, args.batch_size, mode='test')

    # Load model
    net = HybridNet(input_size=args.resize[0], n_classes=args.n_class)
    net, _ = load_pretrained(net, args.checkpoints)

    eval(net, test_loader, device)
