# coding: utf-8
"""
Author: Junfeng Wu (junfeng.wu@ghddi.org)
"""
import logging
import logging.handlers
import time, sys
import os


def init_logger(args):
    if not args.logger:
        name = 'temp'
        folder = ''
        file_name = name
    else:
        name = args.logger_filename
        folder = args.logger_folder
        file_name = name + time.strftime("_%y%h%d_%Hh%Mm%Ss", time.localtime())

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    if not os.path.exists('./logs/' + folder):
        os.makedirs('./logs/' + folder)

    fn = './logs/' + folder + '/' + file_name + '.log'
    fh = logging.FileHandler(fn, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)

    # save all params into logs first
    for k, v in vars(args).items():
        logger.info('%s: %s' %(k, str(v)))

    return logger
