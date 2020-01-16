import os
import multiprocessing
import threading, psutil, queue
from dandelion.util import gpickle
from functools import partial

from .batch_data_loader import *
# from batch_data_loader_GAT import *

import time


def feed_sample_batch(batch_data_loader, data,
                      data_queue=None, max_epoch=200,
                      batch_size=64, batch_size_min=None,
                      shuffle=False):
    me_process = psutil.Process(os.getpid())
    # Xs, Y, adj_matrix_is = data
    sample_num = len(data[0])
    if batch_size_min is None:
        batch_size_min = batch_size
    else:
        batch_size_min = min([batch_size_min, batch_size])

    for epoch in range(max_epoch):
        if me_process.parent() is None:
            raise RuntimeError('Parent process is dead, exiting')
        if shuffle:
            index = np.random.choice(range(sample_num), size=sample_num, replace=False)
        else:
            index = np.arange(sample_num)
        index = list(index)
        end_idx = 0
        while end_idx < sample_num:
            current_batch_size = np.random.randint(batch_size_min, batch_size + 1)
            start_idx = end_idx
            end_idx = min(start_idx + current_batch_size, sample_num)
            batch_sample_idx = index[start_idx:end_idx]
            batch_data = batch_data_loader(data, batch_sample_idx)
            # batch_data = [ligs[i] for i in batch_sample_idx]
            data_queue.put(batch_data)


def start_data_loader(batch_data_loader, data_queue, data,
                      batch_size=128, batch_size_min=None,
                      max_epoch_num=1, worker_num=1,
                      use_multiprocessing=False, name=None,
                      shuffle=False):
    if use_multiprocessing:
        worker_container = multiprocessing.Process
    else:
        worker_container = threading.Thread

    sample_num = len(data[1])
    Xs, Y, adj_matrix_is = data

    # print('create data processes...')
    startidx, endidx, idxstep = 0, 0, sample_num // worker_num
    workers = []
    for i in range(worker_num):
        # startidx and endidx marks the separation for different workers
        startidx = i * idxstep
        if i == worker_num - 1:
            endidx = sample_num
        else:
            endidx = startidx + idxstep
        data_proc = worker_container(target=feed_sample_batch,
                                     args=(batch_data_loader,
                                           (Xs[startidx:endidx], Y[startidx:endidx], adj_matrix_is[startidx:endidx]),
                                           data_queue, max_epoch_num, batch_size,
                                           batch_size_min, shuffle),
                                     name='%s_thread_%d' % (name, i))
        data_proc.daemon = True
        data_proc.start()
        workers.append(data_proc)
    return workers


# TESTING CODE
if __name__ == '__main__':
    Xs, adj_matrix_is, Y = gpickle.load('prep_task/processed_data/1a1e_cm4_nm6_X_adji.gpkl')

    data_container = queue.Queue
    queue_size = 3
    worker = 1
    data_queue = data_container(queue_size * worker)
    t0 = time.time()
    batch_data_loader = partial(batch_data_loader_type)
    workers = start_data_loader(batch_data_loader, data_queue, (Xs, Y, adj_matrix_is),
                                batch_size=256, batch_size_min=256,
                                max_epoch_num=1, worker_num=1,
                                use_multiprocessing=False, name='codetest',
                                shuffle=False)
    print(len(data_queue.get()))
    t1 = time.time()
    print('done\n')
    print('dt: %.2f' %(t1-t0))