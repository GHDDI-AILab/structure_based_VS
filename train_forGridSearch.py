import multiprocessing
import queue
import time
import yaml
import os

import argparse
import numpy as np

from utils.logger import init_logger
from utils.tools import split_trainval, get_annealing_dropout, setup_cfg
from utils import data_prep
from utils import metrics

from models import data_loader
from models.model_agent import Model_Agent

import torch
from torch import optim
from torch import nn

def get_parser():
    """
    Parameters Parser
    """
    argparser = argparse.ArgumentParser(description="SBVS")

    # task selection
    argparser.add_argument('-config_file', default='ndgg_1', type=str)

    # model
    argparser.add_argument('-device', default=-1, type=int, help='device, -1=CPU, >=0=GPU')

    # train/val
    argparser.add_argument('-save_folder', default='ndgg_1')
    argparser.add_argument('-save_epoch', default=-1, type=int)
    
    # multi-worker
    argparser.add_argument('-train_loader_worker', default=1, type=int)
    argparser.add_argument('-test_loader_worker', default=1, type=int)
    argparser.add_argument('-use_multiprocessing', default=False, type=str)
    argparser.add_argument('-queue_size', default=3, type=int)

    # network preparation
    argparser.add_argument('-aa_dict_file', default='./data/aadict.csv', type=str)
    argparser.add_argument('-atom_dict_file', default='./data/atom_dict_wd.csv', type=str)
        # DockingGraph
    # argparser.add_argument('-train_protein_dict', default='./data/protein_dude98pos_truev2all.mp', type=str)
    argparser.add_argument('-train_ext_ligands', default='./data/ligands_dude98pos_truev2all.csv', type=str)
        # NonDockingGG and NonDockingSG
    argparser.add_argument('-train_protein_dict', default='./data/protein_dude98pos_truev2all.mp', type=str)
    # argparser.add_argument('-train_ligands', default='./data/ligands_dude98pos_truev2all.csv', type=str)
    argparser.add_argument('-train_ligands', default='./data/miniset_seed7.csv', type=str)
    #     # NonDockingSG
    argparser.add_argument('-train_protein_seq_dict', default='./data/chain_list_pad.mp', type=str)
    # argparser.add_argument('-train_ligands', default='./data/ligands_dude98pos_truev2all.csv', type=str)
    
    # misc
    argparser.add_argument('-loo', default=False, type=bool)
    argparser.add_argument('-kfold', default=5, type=int)
    argparser.add_argument('-logger', default=False, type=bool)
    argparser.add_argument('-logger_filename', default='trial1', type=str)
    argparser.add_argument('-logger_folder', default='NDGG_GridSearch', type=str)
    argparser.add_argument('-remark', default='', type=str)
    argparser.add_argument('-grid_search', default=100, type=int)

    return argparser

def init_random(CFG):
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    torch.cuda.manual_seed(CFG.SEED)
    return

def objective(trial, args):
    # =================================== configs ===================================
    logger = init_logger(args)

    config_file = './configs/%s.yaml' %args.config_file
    CFG = setup_cfg(config_file, logger)
    
    # ----------------------------------- Grid Search -----------------------------------
    CFG.MODEL_CONFIG.atom_embedding_dim = trial.suggest_categorical('atom_embedding_dim', [25, 50, 75, 100])
    CFG.MODEL_CONFIG.config_ligand.output_dim = trial.suggest_int('cfg_lig_output_dim', 50, 150)
    CFG.MODEL_CONFIG.config_ligand.block_num = trial.suggest_int('cfg_lig_block_num', 3, 7)
    CFG.MODEL_CONFIG.config_ligand.hidden_dim = trial.suggest_categorical('cfg_lig_hidden_num', [128, 256, 512])
    
    CFG.MODEL_CONFIG.config_protein.output_dim = trial.suggest_int('cfg_prt_output_dim', 50, 150)
    CFG.MODEL_CONFIG.config_protein.block_num = trial.suggest_int('cfg_prt_block_num', 1, 5)
    CFG.MODEL_CONFIG.config_protein.hidden_dim = trial.suggest_categorical('cfg_prt_hidden_num', [64, 128, 256])

    CFG.OPTIMIZER = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
    CFG.OPTIMIZER_CONFIG.lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    if CFG.OPTIMIZER == 'SGD':
        CFG.OPTIMIZER_CONFIG.momentum = 0.1
    CFG.OPTIMIZER_CONFIG.weight_decay = trial.suggest_uniform('weight_decay', 0, 1e-4)
    
    # -----------------------------------------------------------------------------------

    init_random(CFG)

    # Params
    anneal_dropouts = CFG.MODEL_VER.anneal_dropouts

    load_data = getattr(data_prep, CFG.MODEL_VER.load_data_prep)
    atom_dict, data, *aux_data = load_data(args, CFG)
    
    device = args.device
    if device < 0:
        logger.info('Using CPU for training')
        device = torch.device('cpu')
    else:
        logger.info('Using CUDA%d for training' % device)
        device = torch.device('cuda:%d' % device)

    trainval_generator = split_trainval(data, args, shuffle=True)

    # data queue for loading data
    if args.use_multiprocessing:
        data_container = multiprocessing.Queue
    else:
        data_container = queue.Queue
    train_data_queue = data_container(args.queue_size * args.train_loader_worker)
    test_data_queue = data_container(args.queue_size * args.test_loader_worker)
    
    # Define Metric for evals
    if CFG.METRIC is not None:
        metric_func = getattr(metrics, CFG.METRIC)

    # Training
    # Iter through trainsets and valsets according to loo or kfold
    k = 0
    while True:
        train_idx, val_idx = next(trainval_generator, (None, None))
        if train_idx is None:
            # training done!
            logger.info('-------------- Done! --------------')
            break
        
        # split trainset and valset
        # [1:] for the first subdata is unique id for loo / other select methods
        trainset = [subdata[train_idx] for subdata in data]
        testset = [subdata[val_idx] for subdata in data]
        testset_sample_num = len(val_idx)
        # build model for each diff train/val
        model_agent = Model_Agent(CFG, device=device, atom_dict=atom_dict)

        test_data_workers = data_loader.start_data_loader(model_agent.batch_data_loader, test_data_queue,
                                                    testset,
                                                    batch_size=CFG.TRAINING.batch_size_test,
                                                    max_epoch_num=CFG.TRAINING.max_epoch_num,
                                                    worker_num=args.test_loader_worker,
                                                    use_multiprocessing=args.use_multiprocessing,
                                                    name='val',
                                                    shuffle=False)
        optimizer = getattr(optim, CFG.OPTIMIZER)
        if CFG.OPTIMIZER_CONFIG is None:
            optimizer = optimizer(model_agent.model.parameters())
        else:
            optimizer = optimizer(model_agent.model.parameters(), **CFG.OPTIMIZER_CONFIG.__dict__)

        criterion = getattr(nn, CFG.LOSS_FUNCTION)
        criterion = criterion()

        logger.info('-------------- Cross-Validation %d --------------' %k)
        best_crit_val = 0
        # training by epoch
        for epoch in range(CFG.TRAINING.max_epoch_num):
            train_cul_time_epoch = 0
            trained_sample_num   = 0
            trainset_sample_num  = len(train_idx)
            
            train_data_workers = data_loader.start_data_loader(model_agent.batch_data_loader, train_data_queue,
                                                    trainset,
                                                    batch_size=CFG.TRAINING.batch_size,
                                                    batch_size_min=CFG.TRAINING.batch_size_min,
                                                    max_epoch_num=1,
                                                    worker_num=args.train_loader_worker,
                                                    use_multiprocessing=args.use_multiprocessing,
                                                    name='train',
                                                    shuffle=True)
            model_agent.model.train()
            
            # start training
            acc_epoch_loss = 0
            while trained_sample_num < trainset_sample_num:
                time0 = time.time()
                batch_train_data = train_data_queue.get()
                time1 = time.time()
                optimizer.zero_grad()

                batch_X, batch_Y = batch_train_data

                # only annealing dropout is support for now
                # could extend later on
                dropout = get_annealing_dropout(epoch, anneal_dropouts, CFG.TRAINING.max_epoch_num)
                scorematrix = model_agent.forward(batch_X, dropout=dropout)

                batch_Y_tensor = torch.from_numpy(batch_Y).to(device)
                loss = criterion(scorematrix, batch_Y_tensor)
                loss.backward()
                nn.utils.clip_grad_value_(model_agent.model.parameters(), 10.0)
                optimizer.step()
                # training procedure
                
                # epoch loss
                acc_epoch_loss += loss.item() * len(batch_Y)
                trained_sample_num += len(batch_Y)
                time2 = time.time()
                
                # time calc
                data_proc_time        = time1 - time0
                train_time            = time2 - time1
                total_batch_time      = time2 - time0
                train_cul_time_epoch += total_batch_time

                print('epoch: %d, loss: %5.2f, trained/total: %d/%d, time = %0.2fs(%0.2f|%0.2f)%.0fs'
                    % (epoch, loss.item(), trained_sample_num, trainset_sample_num,
                        total_batch_time, train_time, data_proc_time, train_cul_time_epoch))

            acc_epoch_loss /= trainset_sample_num

            # start evaluate on val set
            if epoch % CFG.TRAINING.val_per_epoch_num == 0:
                model_agent.model.eval()
                tested_sample_num = 0
                scorematrix_list = []
                groundtruths = []
                with torch.no_grad():
                    while tested_sample_num < testset_sample_num:
                        batch_test_data = test_data_queue.get()
                        batch_X, batch_Y = batch_test_data
                        scorematrix = model_agent.forward(batch_X, dropout=0)
                        scorematrix_list.append(scorematrix.cpu().detach().numpy())
                        groundtruths.append(batch_Y)
                        tested_sample_num += len(batch_Y)
                        print('epoch: %d, tested/total: %d/%d'
                        % (epoch, tested_sample_num, testset_sample_num))
                    scorematrix_list = np.concatenate(scorematrix_list, axis=0)
                    groundtruths = np.concatenate(groundtruths, axis=0)
                    metric_values = metric_func(scorematrix_list, groundtruths)
            metric_print = ' | '.join(['%s: %.3f' %(x, metric_values[x]) for x in metric_values])
            logger.info('epoch: %2d | loss: %.3f | %s'
                        % (epoch, acc_epoch_loss, metric_print))
            auc_aupr_hmean = 2 * metric_values['AUROC'] * metric_values['AUPR'] / (metric_values['AUROC'] + metric_values['AUPR'])
            best_crit_val = max(best_crit_val, auc_aupr_hmean)
            # # save by auc for now
            # if 'AUC' in metric_values and args.save_epoch > 0:
            #     auc_epoch = metric_values['AUC']
            #     if auc_epoch > best_crit_val:
            #         save_path = './results/%s/fold_%d_epoch_%d.pth' %(args.save_folder, k, epoch)
            #         if not os.path.exists('./results/%s' %(args.save_folder)):
            #             os.makedirs('./results/%s' %(args.save_folder))
            #         torch.save(model_agent.model.state_dict(), save_path)

            #         best_crit_val = auc_epoch
        k += 1
        break
    return best_crit_val

if __name__ == '__main__':
    argparser = get_parser()
    args = argparser.parse_args()

    if args.grid_search:
        import optuna
        from functools import partial
        import json
        from dandelion.util import gpickle

        rdb = 'results/grid_search/grid_search_ndgg1.db'
        trial_objective_func = partial(objective, args=args)
        study = optuna.create_study(direction='maximize', storage='sqlite:///' + rdb, load_if_exists=True, study_name='grid_search')
        print('Start grid searching for hyperparameter tuning ...')
        time0 = time.time()
        study.optimize(trial_objective_func, n_trials=args.grid_search)
        time1 = time.time()
        print('Number of finished trials = %d, time cost = %0.2fmins' % (len(study.trials), (time1-time0)/60))
        print('Best trial got:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
        with open('results/grid_search/best_trial_ndgg1.json', encoding='utf8', mode='wt') as f:
            json.dump(trial.params, f, ensure_ascii=False, indent=2)
        gpickle.dump(study, 'results/grid_search/study_ndgg1.gpkl')

