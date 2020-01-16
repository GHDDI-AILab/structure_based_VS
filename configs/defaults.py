# coding: utf-8
"""
default config setup
Author: Junfeng Wu (junfeng.wu@ghddi.org)
TODO: change argparser to config files
"""


import logging
import argparse


def default_arg():
    """
    basic setups
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-device', default=-1, type=int, help='device, -1=CPU, >=0=GPU')
    argparser.add_argument('-model_file', default=None, type=str)
    argparser.add_argument('-model_ver', default='4v1', type=str)
    argparser.add_argument('-class_num', default=2, type=int)
    argparser.add_argument('-optimizer', default='adam')
    argparser.add_argument('-batchsize', default=256, type=int)
    argparser.add_argument('-batchsize_min', default=256, type=int)
    argparser.add_argument('-batchsize_test', default=256, type=int)
    argparser.add_argument('-max_epoch_num', default=10, type=int)
    argparser.add_argument('-save_root_folder', default='results')
    argparser.add_argument('-train_loader_worker', default=1, type=int)
    argparser.add_argument('-test_loader_worker', default=1, type=int)
    argparser.add_argument('-use_multiprocessing', default=False, type=str)
    argparser.add_argument('-dataset_path', default='/data01/jfwu/code/data/SBVS')
    argparser.add_argument('-dataset', default='trueinactive', type=str)
    argparser.add_argument('-logger', default='model4v1', type=str)
    arg = argparser.parse_args()
    return arg
