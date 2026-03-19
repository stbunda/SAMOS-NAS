import os
import sys
import logging
import shutil
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix as cm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class Cutout(object):

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1),
                                    shear=0.1)
        ]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1),
                                    shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform


def get_data(dataset, data_path, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path,
                        train=True,
                        download=True,
                        transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.train_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation:  # append validation data
        ret.append(
            dset_cls(root=data_path,
                     train=False,
                     download=True,
                     transform=val_transform))

    return ret


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters()
        if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class LoggerBuffer:

    def __init__(self, name, path, headers, screen_interval=1):
        self.logger = self.get_logger(path)
        self.history = []
        self.headers = headers
        self.screen_interval = screen_interval

    def get_logger(self, file_path):
        """ Make python logger """
        # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
        logger = logging.getLogger('darts')
        logger.setLevel(logging.DEBUG)

        # set log level
        msg_fmt = '[%(levelname)s] %(asctime)s, %(message)s'
        time_fmt = '%Y-%m-%d_%H-%M-%S %p'
        formatter = logging.Formatter(msg_fmt, time_fmt)

        file_handler = logging.FileHandler(file_path, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        # to avoid duplicated logging info in PyTorch >1.9
        if len(logger.root.handlers) == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.root.addHandler(stream_handler)
        # to avoid duplicated logging info in PyTorch >1.8
        for handler in logger.root.handlers:
            handler.setLevel(logging.WARNING)

        return logger

    def clean(self):
        self.history = []

    def update(self, msg):
        # get the iteration
        n = msg.pop('Iter')
        self.history.append(msg)

        # header expansion
        novel_heads = [k for k in msg if k not in self.headers]
        if len(novel_heads) > 0:
            self.logger.warning(
                'Items {} are not defined.'.format(novel_heads))

        # missing items
        missing_heads = [k for k in self.headers if k not in msg]
        if len(missing_heads) > 0:
            self.logger.warning('Items {} are missing.'.format(missing_heads))

        if self.screen_intvl != 1:
            doc_msg = ['Iter: {:5d}'.format(n)]
            for k, fmt in self.headers.items():
                v = self.history[-1][k]
                doc_msg.append(('{}: {' + fmt + '}').format(k, v))
            doc_msg = ', '.join(doc_msg)
            self.logger.debug(doc_msg)
        '''
        construct message to show on screen every `self.screen_intvl` iters
        '''
        if n % self.screen_intvl == 0:
            screen_msg = ['Iter: {:5d}'.format(n)]
            for k, fmt in self.headers.items():
                vals = [
                    msg[k] for msg in self.history[-self.screen_intvl:]
                    if k in msg
                ]
                v = sum(vals) / len(vals)
                screen_msg.append(('{}: {' + fmt + '}').format(k, v))

            screen_msg = ', '.join(screen_msg)
            self.logger.info(screen_msg)


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def confusion_matrix(output, target, classes):
    pred = np.argmax(output, axis=1)
    
    cf_matrix = cm(pred, target, labels=np.arange(0, len(classes)))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


def accuracy(output, target, topk=1):
    batch_size = target.size(0)
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0)
    return correct_k.mul_(1.0 / batch_size)


def evaluate(output, target, metrics, topk=1):
    """ Computes the accuracy@k for the specified values of k """
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    # Calculate TP, FP, FN, TN
    TP = torch.sum((pred == 1) & (target == 1)).item()
    FP = torch.sum((pred == 1) & (target == 0)).item()
    FN = torch.sum((pred == 0) & (target == 1)).item()
    TN = torch.sum((pred == 0) & (target == 0)).item()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0)
    res = correct_k.mul_(1.0 / batch_size)
    print('Original Accuracy: ', res)

    # Calculate metrics
    output_metrics = dict.fromkeys(metrics)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (
        precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP +
                                                   FN) > 0 else 0.0

    if 'accuracy' in metrics:
        output_metrics['accuracy'] = accuracy
    if 'precision' in metrics:
        output_metrics['precision'] = precision
    if 'recall' in metrics:
        output_metrics['recall'] = recall
    if 'f1' in metrics:
        output_metrics['f1'] = f1

    print(output_metrics)

    return output_metrics


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def build_dataloader(cfg):
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]

    else:
        ...
