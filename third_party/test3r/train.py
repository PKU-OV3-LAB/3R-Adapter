import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from dust3r.training import get_args_parser, train

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
