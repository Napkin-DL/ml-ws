import logging
import os
import shutil
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import horovod.torch as hvd
import smdist_util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

best_acc1 = 0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _get_train_data_loader(args, **kwargs):
    logger.info("Get train data loader")
    logger.info("batch_size: {}".format(args.batch_size))
    transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.RandomAffine(
            0, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # download the dataset
    # this will not only download data to ./mnist folder, but also load and transform (normalize) them
    dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'training_set'),  # 샘플 dataset 위치
                                   transform=transform)
    train_sampler = smdist_util.dis_data(
        dataset, args.horovod) if args.multigpus_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs), train_sampler


def _get_test_data_loader(args, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(root=os.path.join(args.data_dir, 'test_set'),
                             transform=transforms.Compose([
                                 transforms.Resize((args.resize, args.resize)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(gpu, args):
    global best_acc1
    train_start = time.time()

    # model = Net()
    # args.model_name = None
    args.model_name = 'resnet50'
    model = smdist_util.torch_model(args.model_name, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    model, args = smdist_util.dist_setting(
        gpu, model, criterion, args)
#     criterion = criterion.cuda(args.gpu)

    cudnn.benchmark = True

    lr = args.lr
    if args.horovod:
        lr = (args.lr * hvd.size())
    elif args.apex:
        lr = args.lr*float(args.batch_size*args.world_size)/256.

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.apex:
        model, optimizer, args.crop_size, args.val_size = smdist_util.apex_init(
            model, optimizer, args)
    elif args.horovod:
        optimizer = smdist_util.horovod_optimizer(model, optimizer, args)

    if args.resume:
        model = smdist_util.resume(args, model)

    train_loader, train_sampler = _get_train_data_loader(args, **args.kwargs)
    test_loader = _get_test_data_loader(args, **args.kwargs)

    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.info("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    for epoch in range(1, args.epochs + 1):
        end = time.time()
        if not args.apex and args.multigpus_distributed:
            train_sampler.set_epoch(epoch)

        smdist_util.adjust_learning_rate(optimizer, epoch, args)

        if args.apex:
            # train for one epoch
            smdist_util.apex_train(train_loader, model, criterion,
                                   optimizer, epoch, args)
            smdist_util.gpu_measure(args)
            # evaluate on validation set
            acc1 = smdist_util.apex_validate(
                test_loader, model, criterion, args)
        else:
            # train for one epoch
            smdist_util.train(train_loader, model, criterion,
                              optimizer, epoch, args)
            smdist_util.gpu_measure(args)
            # evaluate on validation set
            acc1 = smdist_util.validate(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multigpus_distributed or (args.multigpus_distributed and args.rank % args.num_gpus == 0):
            save_model({
                'epoch': epoch + 1,
                'model_name': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.model_dir)


def save_model(state, is_best, model_dir):
    logger.info("Saving the model.")
    filename = os.path.join(model_dir, 'checkpoint.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))


def main():
    args = smdist_util.parser_args()
    smdist_util.dist_init(train, args)


if __name__ == '__main__':
    main()
