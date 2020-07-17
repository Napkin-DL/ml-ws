import smdebug.pytorch as smd
import argparse
import json
import logging
import os
import sys
import time
from os.path import join

import boto3
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

import GPUtil as GPU
import sagemaker_containers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# SageMaker Debugger: Import the package
# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py


class Net(nn.Module):
    def __init__(self, kernel_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=kernel_size)
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


def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    logger.info("batch_size: {}".format(batch_size))
    dataset = datasets.CIFAR10(training_dir, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), download=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"])) if is_distributed else None

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), download=False),
        batch_size=test_batch_size, shuffle=False, **kwargs)


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args, model, tracker=None):
    train_start = time.time()
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    multi_gpus = args.num_gpus > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    logger.debug("multi_gpus training - {}".format(multi_gpus))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    local_rank = args.local_rank

    if is_distributed or multi_gpus:
        # Initialize the distributed environment.
        world_size = len(args.hosts) * args.num_gpus
        os.environ['WORLD_SIZE'] = str(world_size)
        host_num = args.hosts.index(args.current_host)
        host_rank = args.num_gpus * host_num + local_rank
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend,
                                rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    if multi_gpus:
        # Establish Local Rank and set device on this node
        dp_device_ids = [local_rank]
        torch.cuda.set_device(local_rank)
        print("cuda.set_device(local_rank) : {}".format(local_rank))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(
        args.batch_size, args.data_dir, is_distributed or multi_gpus, **kwargs)
    test_loader = _get_test_data_loader(
        args.test_batch_size, args.data_dir, **kwargs)

    logger.info("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.info("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    if is_distributed and multi_gpus:
        # multi-machine multi-gpu case

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss().to(device)

    hook = smd.Hook.create_from_json_file()
    hook.register_module(model.module)
    hook.register_loss(criterion)  # nn.Module
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.cuda(local_rank, non_blocking=True), target.cuda(
                local_rank, non_blocking=True)
            optimizer.zero_grad()
            try:
                output = model(data)
                logger.info('HOST_NUM: {} | Allocated: {} GB | Cached: {} GB'.format(str(local_rank), round(
                    torch.cuda.memory_allocated(local_rank) / 1024**3, 1), round(torch.cuda.memory_cached(local_rank) / 1024**3, 1)))
                loss = criterion(output, target)
                loss.backward()
                if is_distributed and not use_cuda:
                    # average gradients manually for multi-machine cpu case only
                    _average_gradients(model)
                optimizer.step()
            except Exception as e:
                print("e: {}".format(e))
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f};'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
                hook.save_scalar('train_loss', loss.item(), sm_metric=True)

        test(model, test_loader, local_rank, tracker)
        # measure time intervals from train start / epoch start to now
        t_time = '{:0.3f}'.format(time.time() - train_start)
        e_time = '{:0.3f}'.format(time.time() - epoch_start)
        gpu_measure(local_rank, t_time, e_time)

    save_model(model, args.model_dir)


def test(model, test_loader, local_rank, tracker=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(local_rank, non_blocking=True), target.cuda(
                local_rank, non_blocking=True)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('Test Average loss: {:.4f}, Test Accuracy: {:.0f}%;\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))


def gpu_measure(local_rank, t_time=None, e_time=None):
    GPUs = GPU.getGPUs()
    m_gpu = GPUs[local_rank]
    logger.info("Total_time: {0}, Epoch_time: {1}, GPU_NUM : {2}, GPU RAM Free: {3:.0f}MB | Used: {4:.0f}MB | Util {5:3.0f}% | Total {6:.0f}MB".format(
        t_time, e_time, local_rank, m_gpu.memoryFree, m_gpu.memoryUsed, m_gpu.memoryUtil * 100, m_gpu.memoryTotal))


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     hidden_channels = int(os.environ.get('hidden_channels', '5'))
    kernel_size = int(os.environ.get('kernel_size', '5'))
#     dropout = float(os.environ.get('dropout', '0.5'))
    model = torch.nn.DataParallel(Net(kernel_size))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
