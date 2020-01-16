# coding: utf-8
"""
Author: Junfeng Wu (junfeng.wu@ghddi.org)
"""
import logging
import logging.handlers
import time, sys


def setup_logger(name='SBVS'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    fn = 'logs/' + name + time.strftime("_%y%h%d_%Hh%Mm%Ss.log", time.localtime())
    fh = logging.FileHandler(fn, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(plain_formatter)
    logger.addHandler(fh)
    return logger
