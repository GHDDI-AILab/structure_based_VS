
import yaml
import numpy as np

def setup_cfg(config_file, logger=None):
    # load yaml config
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    # logger / print config
    config_dumps = yaml.dump(cfg, indent=4)
    if logger:
        logger.info('=========================== CONFIGURATIONS ===========================')
        for dump_line in config_dumps.split('\n'):
            logger.info(dump_line)
    else:
        print(config_dumps)
    # return config obj
    return dict2obj(cfg)

def dict2obj(d):

    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
           d = [dict2obj(x) for x in d]

    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
           return d

    # declaring a class
    class C:
        pass

    # constructor of the class passed to obj
    obj = C()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])

    return obj

def split_trainval(data, args, shuffle=False):
    if type(data) is list:
        len_ = len(data[0])
        # make sure every element in the list contains some number of elements
        for i in range(1, len(data)):
            assert len(data[i]) == len_
    else:
        # len_ = number of samples
        len_ = len(data[0])
    
    # data[0] will always be unique_id for loo if loo applicable
    unique_ids = np.unique(data[0])

    if args.loo is True:
        # LOO
        for i in range(len(unique_ids)):
            train_idx = np.where(data[0] != unique_ids[i])[0]
            val_idx = np.where(data[0] == unique_ids[i])[0]
            yield train_idx, val_idx
    elif args.kfold > 0:
        # K-Fold
        k = args.kfold
        idx = np.arange(len_)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(k):
            val_idx = np.zeros_like(idx, dtype=bool)
            val_idx[int(len_ * i / k):int(len_ * (i+1) / k)] = True
            train_idx = idx[np.logical_not(val_idx)]
            val_idx = idx[val_idx]
            yield train_idx, val_idx
    else:
        idx = np.arange(len_)
        # use all data for training
        yield idx, None

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