import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.special import softmax

def metrics_1(scorematrix, groundtruths, sample_weight=None):
    # input: numpy array
    # AUC, AuPRef, accuracy
    AUROC = roc_auc_score(groundtruths, softmax(scorematrix, axis=1)[:, 1])
    
    sample_num, class_num = scorematrix.shape
    auprs = []
    for i in range(class_num):
        p, r, th = precision_recall_curve(groundtruths, scorematrix[:,i], pos_label=i, sample_weight=sample_weight)
        aupr = auc(r, p)
        auprs.append(aupr)
    aupr_hmean = scipy.stats.hmean(auprs)

    scores = scorematrix[:, 1]
    # k denotes percentage
    gt_scores = [(groundtruths[ii], scores[ii]) for ii in range(len(groundtruths))]
    gt_scores = np.array(sorted(gt_scores, key=lambda x: x[1], reverse=True))

    ef_denominator = gt_scores[:, 0].sum() / len(scores)
    hits = np.cumsum(gt_scores[:, 0])
    ef = hits / (np.arange(len(gt_scores)) + 1) / ef_denominator
    ef1  = ef[int(len(scores) * 1 / 100)]
    ef2  = ef[int(len(scores) * 2 / 100)]
    ef5  = ef[int(len(scores) * 5 / 100)]
    ef10 = ef[int(len(scores) * 10 / 100)]
    ef20 = ef[int(len(scores) * 20 / 100)]

    pred = scorematrix.argmax(axis=1)
    acc = (pred == groundtruths).sum() / len(groundtruths)

    metric_value_dict = {
        'AUROC': AUROC,
        'AUPR': aupr_hmean,
        'ACC': acc,
        'EF@1%':  ef1,
        'EF@2%':  ef2,
        'EF@5%':  ef5,
        'EF@10%': ef10,
        'EF@20%': ef20,
    }
    return metric_value_dict

