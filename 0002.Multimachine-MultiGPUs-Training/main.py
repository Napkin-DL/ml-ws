import argparse
import json
import logging
import os
import sys
from os.path import join

import torch
import torch.multiprocessing as mp

import train_model


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


if 'SAGEMAKER_METRICS_DIRECTORY' in os.environ:
    log_file_handler = logging.FileHandler(
        join(os.environ['SAGEMAKER_METRICS_DIRECTORY'], "metrics.json"))
    log_file_handler.setFormatter(
        "{'time':'%(asctime)s', 'name': '%(name)s', \
    'level': '%(levelname)s', 'message': '%(message)s'}"
    )
    logger.addHandler(log_file_handler)


def main():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default="sgd",
                        help='optimizer for training.')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--kernel_size', type=int, default=5, metavar='KERNEL',
                        help='conv2d filter kernel size (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--hidden_channels', type=int, default=10,
                        help='number of channels in hidden conv layer')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str,
                        default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int,
                        default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    mp.set_start_method('spawn')

    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    model = train_model.Net(args.kernel_size)

    args.local_rank = 0
    if args.num_gpus > 1:
        processes = []

        for local_rank in range(args.num_gpus):
            model = model.cuda(local_rank)
            model.share_memory()
            args.local_rank = local_rank

            p = mp.Process(target=train_model.train, args=(args, model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        train_model.train(args, model)


if __name__ == '__main__':
    main()
