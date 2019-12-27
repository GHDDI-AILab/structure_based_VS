# import multiprocessing
# import queue
# import logging
# from dandelion.util import gpickle

from configs.defaults import default_arg
from utils import *
from model import *
from dataloader import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


def main(arg, logger):
    # =================================== configs ===================================
    use_multiprocessing = arg.use_multiprocessing
    train_loader_worker = arg.train_loader_worker
    test_loader_worker = arg.test_loader_worker
    batch_size = arg.batchsize
    batch_size_min = arg.batchsize_min
    batch_size_test = arg.batchsize_test
    max_epoch_num = arg.max_epoch_num

    anneal_dropouts = [(0.1, 0), (0.1, 0.5)]

    atom_dict, proteins, data_for_training = preparation(arg)
    Xs, Ys, adj_matrix_is = data_for_training

    device = arg.device
    if device < 0:
        logger.info('Using CPU for training')
        device = torch.device('cpu')
    else:
        logger.info('Using CUDA%d for training' % device)
        device = torch.device('cuda:%d' % device)

    # =================================== logging: basic config ===================================
    logger.info("Basic Configs")
    for k, v in vars(arg).items():
        logger.info('%s: %s' %(k, str(v)))

    # =================================== training ===================================
    logger.info("======================== Start Training ========================")
    logger.info("Number of Proteins: %d" % len(proteins))
    for j in range(len(proteins)):
        train_cul_time_protein = 0
        # =================================== separate train/test ===================================
        # Xs, Y, adj_matrix_is
        trainset = [[], [], []]
        for jj in range(len(proteins)):
            if jj == j:
                testset = [Xs[j], Ys[j], adj_matrix_is[j]]
            else:
                trainset[0] = trainset[0] + Xs[jj]
                trainset[1] = np.concatenate([trainset[1], Ys[jj]]).astype(np.int64)
                trainset[2] = trainset[2] + adj_matrix_is[jj]
        trainset[2] = [ts.astype(np.int64) for ts in trainset[2]]
        test_sample_num = len(testset[0])

        # =================================== dataloader & model ===================================
        if use_multiprocessing:
            data_container = multiprocessing.Queue
        else:
            data_container = queue.Queue
        queue_size = 3
        train_data_queue = data_container(queue_size * train_loader_worker)
        test_data_queue = data_container(queue_size * test_loader_worker)

        batch_data_loader = partial(batch_data_loader_type)
        test_data_workers = start_data_loader(batch_data_loader, test_data_queue,
                                              testset,
                                              batch_size=batch_size_test, max_epoch_num=max_epoch_num,
                                              worker_num=test_loader_worker,
                                              use_multiprocessing=use_multiprocessing, name='test',
                                              shuffle=False)

        net = model_4v1(num_embedding=len(atom_dict) + 1,
                        block_num=5,
                        input_dim=75,
                        hidden_dim=256,
                        output_dim=2,
                        aggregation_methods=['rnn', 'sum'],
                        multiple_aggregation_merge='cat',
                        readout_method='sum',
                        eps=0.1,
                        add_dense_connection=True)
        net = net.to(device)

        optimizer = Adam(net.parameters())
        criterion = nn.CrossEntropyLoss()

        best_auc = 0

        logger.info('TESTING PROTEIN: %s' % proteins[j])
        logger.info('----------------------------------------------------------------')

        aucs = []
        ef2s = []
        ef20s = []
        efmaxs = []
        accuracys = []
        for epoch in range(max_epoch_num):
            train_cul_time_epoch = 0

            train_sample_num = len(trainset[0])
            trained_sample_num = 0
            train_data_workers = start_data_loader(batch_data_loader, train_data_queue,
                                                   trainset,
                                                   batch_size=batch_size, batch_size_min=batch_size_min,
                                                   max_epoch_num=1, worker_num=train_loader_worker,
                                                   use_multiprocessing=use_multiprocessing, name='train',
                                                   shuffle=True)
            net.train()

            while trained_sample_num < train_sample_num:
                time0 = time.time()
                batch_train_data = train_data_queue.get()
                time1 = time.time()
                optimizer.zero_grad()

                X, Y, membership_i, membership_v, adj_matrix_i, adj_matrix_v, padded_neighbors = batch_train_data
                X_tensor = torch.from_numpy(X).to(device)
                Y_tensor = torch.from_numpy(Y).to(device)
                membership = torch.sparse.FloatTensor(torch.from_numpy(membership_i),
                                                      torch.from_numpy(membership_v),
                                                      torch.Size([len(Y), len(X)])).to(device)
                adj_matrix = torch.sparse.FloatTensor(torch.from_numpy(adj_matrix_i),
                                                      torch.from_numpy(adj_matrix_v),
                                                      torch.Size([len(X), len(X)])).to(device)
                padded_neighbors = torch.from_numpy(padded_neighbors).to(device)
                dropout = get_annealing_dropout(epoch, anneal_dropouts, max_epoch_num)
                scorematrix = net(X_tensor, adj_matrix=adj_matrix,
                                  membership=membership, padded_neighbors=padded_neighbors,
                                  dropout=dropout)
                loss_classification = criterion(scorematrix, Y_tensor)

                total_loss = loss_classification
                total_loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 10.0)
                optimizer.step()

                batch_sample_num = len(Y)
                trained_sample_num += batch_sample_num

                time2 = time.time()
                data_process_time = time1 - time0
                train_time = time2 - time1
                train_cul_time_epoch += train_time

                logger.info('epoch: %d, loss: %5.2f, trained/total: %d/%d, time = %0.2fs(%0.2f|%0.2f)%.0fs'
                      % (epoch, total_loss.item(), trained_sample_num, train_sample_num,
                         (time2 - time0), train_time, data_process_time, train_cul_time_epoch))

            # evaluate on test case for each epoch
            train_cul_time_protein += train_cul_time_epoch
            logger.info("------------------------ Start Testing -------------------------")

            net.eval()
            tested_sample_num = 0
            correct = 0
            scorematrix_list = []
            groundtruths = []

            while tested_sample_num < test_sample_num:
                batch_test_data = test_data_queue.get()
                X, Y, membership_i, membership_v, adj_matrix_i, adj_matrix_v, padded_neighbors = batch_test_data
                X_tensor = torch.from_numpy(X).to(device)
                membership = torch.sparse.FloatTensor(torch.from_numpy(membership_i),
                                                      torch.from_numpy(membership_v),
                                                      torch.Size([len(Y), len(X)])).to(device)
                adj_matrix = torch.sparse.FloatTensor(torch.from_numpy(adj_matrix_i),
                                                      torch.from_numpy(adj_matrix_v),
                                                      torch.Size([len(X), len(X)])).to(device)
                padded_neighbors = torch.from_numpy(padded_neighbors).to(device)
                scorematrix = net(X_tensor, adj_matrix=adj_matrix,
                                  membership=membership, padded_neighbors=padded_neighbors,
                                  dropout=0)
                predict = scorematrix.max(axis=1)[1].cpu().numpy()
                scorematrix_list.append(F.log_softmax(scorematrix, dim=1).cpu().detach().numpy())
                correct += (predict == Y).sum()
                groundtruths.append(Y)

                batch_sample_num = len(Y)
                tested_sample_num += batch_sample_num

                logger.info('epoch: %d, accuracy: %5.2f, tested/total: %d/%d'
                      % (epoch, correct / tested_sample_num, tested_sample_num, test_sample_num))

            scores = np.concatenate(scorematrix_list)
            scores = np.exp(scores)[:, 1]
            gt = np.concatenate(groundtruths)

            accuracy = 100 * correct / test_sample_num
            scoresToAuc = [(gt[ii], scores[ii]) for ii in range(len(gt))]
            scoresToAuc = sorted(scoresToAuc, key=lambda x: x[1], reverse=True)
            auc = AUCScorer(scoresToAuc).auc

            ef_denominator = gt.sum() / len(gt)
            ef = enrichment_factor(scoresToAuc, ef_denominator)
            efmax = ef.max()
            ef2 = ef[int(len(scores) * 2 / 100)]
            ef20 = ef[int(len(scores) * 20 / 100)]

            aucs.append(auc)
            ef2s.append(ef2)
            ef20s.append(ef20)
            efmaxs.append(efmax)
            accuracys.append(accuracy)

            logger.info('epoch: %d | auc: %.3f | ef2: %.3f | ef20: %.3f | efmax: %.3f | accuracy: %.3f'
                      % (epoch, auc, ef2, ef20, efmax, accuracy))
            # save model if better auc
            if auc > best_auc:
                save_path = './%s/%d.pth' % (arg.save_root_folder, j)
                torch.save(net.state_dict(), save_path)
        logger.info('-------------------- overall testing result --------------------')
        logger.info("Total Training Time: %.1f mins" % (train_cul_time_protein/60))
        for i in range(len(aucs)):
            logger.info('epoch: %d | auc: %.3f | ef2: %.3f | ef20: %.3f | efmax: %.3f | accuracy: %.3f'
                      % (i, aucs[i], ef2s[i], ef20s[i], efmaxs[i], accuracys[i]))
        logger.info('================================================================')

    logger.info("========================= End Training =========================")


if __name__ == '__main__':
    arg = default_arg()
    logger = setup_logger(arg.logger)
    main(arg, logger)
