import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn.functional as F

from models.hybridnet import HybridNet
from utils.data_loader import get_loader
from utils.meter import AverageMeter
from utils.losses import calc_loss
from utils.saver import save_checkpoint, load_pretrained
from progress.bar import Bar


import warnings
warnings.filterwarnings("ignore")


def train(train_loader, net, optimizer, epoch, device):
    """Train every epoch"""
    net.train()

    bar = Bar('Processing train', max=len(train_loader))
    loss_ = AverageMeter()

    for i, (img, gt) in enumerate(train_loader):
        img = img.to(device)
        gt = gt.to(device)

        # segmentation result
        sr_probs, d1_probs, d2_probs, d3_probs, d4_probs = net(img)

        # Total loss
        loss = calc_loss(sr_probs, gt, bce_weight=.5)
        loss += calc_loss(d1_probs, gt, bce_weight=.5)
        loss += calc_loss(d2_probs, gt, bce_weight=.5)
        loss += calc_loss(d3_probs, gt, bce_weight=.5)
        loss += calc_loss(d4_probs, gt, bce_weight=.5)
        loss_.update(loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.suffix = '(Epoch:{epoch} - {batch}/{size}) | Loss: {loss:.3f}'.format(
            batch=i + 1,
            size=len(train_loader),
            epoch=epoch,
            loss=loss_.avg
        )
        bar.next()
    bar.finish()

    return loss_.avg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='CEDenseNetAttention Training Demo')
    parser.add_argument('--epochs', type=int, default=300,
                        help='the epochs of this run')
    parser.add_argument('--n_class', type=int, default=1,
                        help='the channel of out img, decide the num of class')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--data_path', type=str, default='./dataset/DRIVE',
                        help='dir of the all img')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='the num of img in a batch')
    # parser.add_argument('--crop_size', type=tuple, default=(544, 544),
    #                     help='the crop img size')
    parser.add_argument('--resize', type=tuple, default=(512, 512),
                        help='the train img size')
    parser.add_argument('--gpu_avaiable', type=str, default='0',
                        help='the gpu used')
    parser.add_argument('--checkpoints', type=str,
                        default='./weights', help="weight's path")
    parser.add_argument('--early_stopping', type=int,
                        default=20, help="early stopping parameter")
    parser.add_argument('--pretrained', type=str,
                        default='./weights/net-epoch-51-0.7895.pth', help="pretrained model path")
    parser.add_argument('--resume', dest='resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

    # Data iteration generation
    # train_loader = get_loader(
    #     args.data_path, args.crop_size, args.resize, args.batch_size, shuffle=True
    # )
    train_loader = get_loader(
        args.data_path, None, args.resize, args.batch_size, shuffle=True
    )

    # Model given
    net = HybridNet(input_size=args.resize[0], n_classes=args.n_class)

    if args.resume:
        net, _ = load_pretrained(net, args.pretrained)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    best_train_loss = np.inf
    no_optimize = 0

    # Training
    for epoch in range(args.epochs):
        train_loss = train(train_loader, net, optimizer, epoch, device)

        if train_loss >= best_train_loss:
            no_optimize += 1
        else:
            no_optimize = 0
            best_train_loss = train_loss

            filename = []
            filename.append(os.path.join(args.checkpoints,
                                         'net-epoch-%s-%s.pth' % (epoch + 1, round(best_train_loss, 4))))
            filename.append(os.path.join(args.checkpoints, 'model_best.pth'))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
            }, True, filename)

        if no_optimize > args.early_stopping:
            print("Early Stopping...")
            break

    print("Training Done...")
