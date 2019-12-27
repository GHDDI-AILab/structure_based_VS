import numpy as np


def get_annealing_dropout(current_epoch, anneal_dropouts, max_epoch=None):
    """
    Dropout annealing. The annealing curve is piece-wise linear, defined by `anneal_dropouts`.
    The `anneal_dropouts` is a (n, 2) shaped np.ndarray or an equal list, each row of `anneal_dropouts` is of
    format (dropout, idx), in which idx is either integer (unscaled) or float <= 1.0 (scaled), in the latter case
    the `max_epoch` must be specified.
    :param current_epoch:
    :param anneal_dropouts:
    :param max_epoch:
    :return:
    """
    if isinstance(anneal_dropouts, list):
        anneal_dropouts = np.array(anneal_dropouts)
    if np.all(anneal_dropouts[:, 1] <= 1.0):
        if max_epoch is None:
            raise ValueError('max_epoch must be specified if scaled anneal_dropouts is used')
        anneal_dropouts[:, 1] *= max_epoch
    n = anneal_dropouts.shape[0]
    idx = n
    for i in range(n):
        if current_epoch < anneal_dropouts[i, 1]:
            idx = i
            break
    if idx == n:
        dropout = anneal_dropouts[-1, 0]
    else:
        p1, p2 = anneal_dropouts[idx-1, 0], anneal_dropouts[idx, 0]
        x1, x2 = anneal_dropouts[idx-1, 1], anneal_dropouts[idx, 1]
        x = current_epoch
        dropout = (x - x1) / (x2 - x1) * (p2 - p1) + p1
    return dropout
