import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim

from models.mdmnet import MDMNet
from utils.data_loader import get_loader
from utils.utils import AverageMeter
from utils.losses import calc_loss
from utils.utils import reproducible
from utils.saver import save_checkpoint, load_pretrained
from progress.bar import Bar
from torch.utils.tensorboard import SummaryWriter
from utils.saver import rm_mkdir

from evalidate import eval

from torch.optim import lr_scheduler


def train(train_loader, net, optimizer, epoch, device):
    """Train every epoch"""
    net.train()

    bar = Bar('Processing train', max=len(train_loader))
    loss_ = AverageMeter()

    for i, (img, gt) in enumerate(train_loader):
        img = img.to(device)
        gt = gt.to(device)

        # segmentation result
        d_probs, d1_probs, d2_probs, d3_probs, d4_probs = net(img)

        loss = calc_loss(d_probs, gt, bce_weight=.5)
        # loss += calc_loss(d2_probs, gt, bce_weight=.5)
        # loss += calc_loss(d3_probs, gt, bce_weight=.5)
        # loss += calc_loss(d4_probs, gt, bce_weight=.5)
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
        description='MDMNet Training Pipline')
    parser.add_argument('--epochs', type=int, default=500,
                        help='the epochs of this run')
    parser.add_argument('--n_class', type=int, default=1,
                        help='the channel of out img, decide the num of class')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--data_path', type=str, default='./dataset/DRIVE',
                        help='dir of the all img')
    parser.add_argument('--dataset', type=str, default='DRIVE',
                        help="dataset's name")
    parser.add_argument('--batch_size', type=int, default=4,
                        help='the num of img in a batch')
    parser.add_argument('--resize', type=tuple, default=(512, 512),
                        help='the train img size')
    parser.add_argument('--gpu_avaiable', type=str, default='0',
                        help='the gpu used')
    parser.add_argument('--checkpoints', type=str,
                        default='./weights/DRIVE_scratch', help="weight's path")
    # parser.add_argument('--early_stopping', type=int,
    #                     default=30, help="early stopping parameter")
    parser.add_argument('--pretrained', type=str,
                        default='./weights/DRIVE_scratch/model_best.pth', help="pretrained model path")
    parser.add_argument('--resume', dest='resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    rm_mkdir(args.checkpoints)

    # Data iteration generation
    train_loader = get_loader(
        args.data_path, args.resize, args.batch_size, shuffle=True, dataset_name=args.dataset, num_workers=4
    )
    val_loader = get_loader(
        args.data_path, args.resize, args.batch_size, shuffle=False, dataset_name=args.dataset, mode='test',
        num_workers=4
    )

    net = MDMNet(input_size=args.resize, n_classes=args.n_class)

    if args.resume:
        net, _ = load_pretrained(net, args.pretrained)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    best_f1 = 0
    # no_optimize = 0

    writer = SummaryWriter(os.path.join(args.checkpoints, 'logs'))
    for epoch in range(args.epochs):
        train_loss = train(train_loader, net, optimizer, epoch, device)
        f1_score, val_loss = eval(net, val_loader, device)
        writer.add_scalar('Loss/train_loss', train_loss, epoch + 1)
        writer.add_scalar('Loss/validate_loss', val_loss, epoch + 1)
        writer.add_scalar('Loss/F1-score', f1_score, epoch + 1)
        # if val_loss >= lower_loss:
        #     no_optimize += 1
        # else:
        #     no_optimize = 0
        scheduler.step(val_loss)

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('Loss/learning_rate', lr, epoch + 1)

        if f1_score > best_f1:
            best_f1 = f1_score

            filename = []
            filename.append(os.path.join(args.checkpoints,
                                         'net-epoch-%s-%s.pth' % (epoch + 1, round(best_f1, 4))))
            filename.append(os.path.join(args.checkpoints, 'model_best.pth'))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
            }, True, filename)

        # if no_optimize > args.early_stopping:
        #     print("Early Stopping...")
        #     break

    print("Training Done...")
