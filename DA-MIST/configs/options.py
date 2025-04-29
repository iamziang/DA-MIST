import argparse
from random import seed
import os

def parse_args():
    descript = 'Pytorch Implementation of UR-DMU'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--output_path', type = str, default = 'outputs/')
    parser.add_argument('--root_src', type = str, default = 'features/src/')
    parser.add_argument('--root_full', type = str, default = 'features/full/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    parser.add_argument('--modal', type = str, default = 'rgb',choices = ["rgb,flow,both"])  # 使用哪些特征的模态
    parser.add_argument('--weight_path', type = str, default = 'weights/')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--num_iters', type = int, default = 3000)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--num_segments', type = int, default = 32)
    parser.add_argument('--seed', type = int, default = 3407, help = 'random seed (-1 for no manual seed)')
    parser.add_argument('--model_file', type = str, default = "trans_{}.pkl".format(seed), help = 'the path of pre-trained model file')
    parser.add_argument('--debug', action = 'store_true')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.weight_path):
        os.makedirs(args.weight_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
