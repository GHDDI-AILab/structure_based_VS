# coding: utf-8
"""
Author: Junfeng Wu (junfeng.wu@ghddi.org)
"""
import numpy as np


def enrichment_factor(scoresToAuc, ef_denominator = None):
    # k denotes percentage
    scores = np.array(scoresToAuc)
    if not ef_denominator:
        ef_denominator = scores[:, 0].sum() / len(scores)
    ef = np.cumsum(scores[:, 0]) / (np.arange(len(scores)) + 1) / ef_denominator
    return ef