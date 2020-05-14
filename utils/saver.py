import os
import shutil
import torch


def save_checkpoint(state, is_best, filename):
    if isinstance(filename, list) and len(filename) == 2:
        torch.save(state, filename[0])
        if is_best:
            shutil.copyfile(filename[0], filename[1])
    else:
        raise Exception("save deploy error, please check files...")


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        if torch.cuda.is_available():
            checkpoint = torch.load(fname)
        else:
            checkpoint = torch.load(fname, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model, checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)
