# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=1)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=60)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-nsNCM', '--num_support_NCM',
                        type=int,
                        help='number of samples per stage to use as query for training, default=5',
                        default=5)
    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--lr', default = 0.01, type = int)
    parser.add_argument('--max_size', default = 2000, type = int)
    parser.add_argument('--total_cls', default = 100, type = int)
    parser.add_argument('--stage',default = 5, type = int)
    parser.add_argument('--class_per_stage',default = 20, type = int)
    parser.add_argument('--edge',default = 10000, type = int)
    parser.add_argument('--NCM_batch',default = 1024, type = int)
    parser.add_argument('--Data_file',default = 'train_meta', type = str)
    parser.add_argument('--Bias_epoch',default = 30, type = int)
    parser.add_argument('--lossF',default = 'NCM', type = str)
    parser.add_argument('--pushR',default = 0.1, type = float)
    parser.add_argument('--pillR',default = 0.1, type = float)
    parser.add_argument('--centerR',default = 0.1, type = float)
    parser.add_argument('--mix',
                        action='store_true',help='enables mix')
    return parser
