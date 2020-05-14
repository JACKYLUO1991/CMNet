import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import time
import torch
import torchvision.transforms.functional as TF

from models.mdmnet import MDMNet
from utils.data_loader import get_loader
from utils.saver import rm_mkdir, load_pretrained


def run_test(model, data_loader, device, pred_path):
    '''
    Parameter:
        model: model after loading weights
        device: gpu or cpu?
        pred_path: image storage path | whether save prob image or binary image
    '''
    model.eval().to(device)
    rm_mkdir(pred_path)

    with torch.no_grad():
        batch_times = []
        
        prob_images_so_far = 0
        bin_images_so_far = 0

        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            # label = label.to(device)

            # torch.cuda.synchronize()
            start_time = time.perf_counter()
            _, _, _, pred_prob, _  = net(data)  # after sigmoid function
            # torch.cuda.synchronize()
            inf_time = time.perf_counter() - start_time

            preds = torch.gt(pred_prob, .5).float()

            print('Batch {}/{} inference time per image: {:.5f}s'.format(i +
                                                                         1, len(test_loader), inf_time))
            batch_times.append(inf_time)

            if pred_path.endswith('binary'):
                for j in range(data.size()[0]):
                    bin_images_so_far += 1
                    preds_img = TF.to_pil_image(preds.cpu().data[j])
                    pred_name = '{}_vessel_binary.png'.format(
                        str(bin_images_so_far).zfill(2))
                    preds_img.save(osp.join(pred_path, pred_name))

            elif pred_path.endswith('prob'):
                for j in range(data.size()[0]):
                    prob_images_so_far += 1
                    preds_img = TF.to_pil_image(pred_prob.cpu().data[j])
                    pred_name = '{}_vessel_prob.png'.format(
                        str(prob_images_so_far).zfill(2))
                    preds_img.save(osp.join(pred_path, pred_name))
            else:
                raise Exception('Path cannot be resolved...')
            
    # ignore first batch for warm up
    batch_avg = np.mean(batch_times[1:])
    print()
    print('Mean inference time per image: {:.5f}s'.format(batch_avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MDMNet Testing Demo')
    parser.add_argument('--data_path', type=str, default='./dataset/DRIVE',
                        help='dir of the all img')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the num of img in a batch')
    parser.add_argument('--dataset', type=str, default='DRIVE',
                        help="dataset's name")
    parser.add_argument('--resize', type=tuple, default=(512, 512),
                        help='the train img size')
    parser.add_argument('--n_class', type=int, default=1,
                        help='the channel of out img, decide the num of class')
    parser.add_argument('--gpu_avaiable', type=str, default='0',
                        help='the gpu used')
    parser.add_argument('--checkpoints', type=str,
                        default='./weights/DRIVE1/model_best.pth', help="weight's path")
    parser.add_argument('--save_path', type=str, required=True,
                        choices=['./results/prob', './results/binary'], help="weight's path")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading test data
    test_loader = get_loader(
        args.data_path, args.resize, args.batch_size, mode='test', dataset_name=args.dataset)

    # Load model
    net = MDMNet(input_size=args.resize, n_classes=args.n_class)

    # Load weights
    net, _ = load_pretrained(net, args.checkpoints)

    run_test(net, test_loader, device, args.save_path)
