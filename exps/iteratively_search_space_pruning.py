

import argparse
import os
from tqdm import tqdm
import random
import json
from scipy import stats
import torch
import numpy as np
import lightgbm as lgb
from keras.utils import to_categorical

from utils import get_archs, generate_data, get_adj_matrix, load_adj, config_dataloader
from generate_seeds import Random_NAS

from NAS_Net.genotypes import PRIMITIVES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for training op combination predictor')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--comb_lr', type=float, default=0.0001)  # 0.001+adam or 0.0001+adam

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--leaves', type=int, default=31)
parser.add_argument('--num_boost_round', type=int, default=100)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--mode', type=str, default='one-shot')
parser.add_argument('--threshold', type=float, default=25.280)

args = parser.parse_args()


def seed_loader(noisy_comb_to_EDF):
    combs, accs = zip(*list(noisy_comb_to_EDF.items()))

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f'mean acc: {mean_acc}')
    normed_accs = [(i - mean_acc) / std_acc for i in accs]
    x = np.array([[int(v) for v in s] for s in combs])
    y = np.array(normed_accs)

    return x, y


def load_final_noisy_train_set(epoch):
    dir = '../seeds/final_08_seeds/final_train_seeds'
    file_names = os.listdir(dir)
    noisy_set = []
    for file_name in file_names:
        file_path = os.path.join(dir, file_name)
        with open(file_path, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch + 1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            noisy_set.append((arch, mae))

    return noisy_set


def load_seed(dir, epoch):
    noisy_set = []
    file_names = os.listdir(dir)
    for file_name in file_names:
        if file_name.endswith('.json'):
            file_path = os.path.join(dir, file_name)
            with open(file_path, "r") as f:
                arch_pairs = json.load(f)
            for arch_pair in arch_pairs:
                arch = arch_pair['arch']
                info = arch_pair['info'][:epoch]
                mae = sorted(info, key=lambda x: x[0])[0][0]
                noisy_set.append((arch, mae))
    return noisy_set


def load_final_noisy_valid_set(epoch):
    dir = '../seeds/final_08_seeds/final_valid_seeds'
    file_names = os.listdir(dir)
    noisy_set = []
    for file_name in file_names:
        file_path = os.path.join(dir, file_name)
        with open(file_path, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch + 1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            noisy_set.append((arch, mae))
    return noisy_set





def generate_EDF(arch_seeds, threshold):  # 这个阈值应该再迭代过程中动态调整？？？考虑到EDF的计算方式，可能会有很多不同comb对应相同label，如何处理这个问题？
    """from (arch, mae) to (comb, EDF)"""

    # collect (comb, acc_list) pair
    comb_to_accs = dict()
    accs = []
    for arch_acc in arch_seeds:
        arch, acc = arch_acc  # 要不要对mae取倒数？
        _, op = zip(*arch)
        comb = to_categorical(op, num_classes=len(PRIMITIVES), dtype=int)
        comb = np.sum(comb, axis=0).tolist()
        str_comb = ''.join(str(v) for v in comb)  # 转成str方便用字典存储
        comb_to_accs.setdefault(str_comb, []).append(acc)
        accs.append(acc)


    threshold_e = threshold
    print(threshold_e)
    comb_to_EDF = dict()

    for k, v in comb_to_accs.items():
        if len(v) < 30:
            continue
        print(k, sorted(v))

        comb_to_EDF[k] = len([i for i in v if i < threshold_e]) / len(v)

        print(comb_to_EDF[k])
    return comb_to_EDF


def main():
    gbm_params = {
        'boosting_type': 'gbdt',  # 是不是应该换成LGBoost？
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': args.leaves,
        'learning_rate': args.lr,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    if args.mode == 'iteratively':
        origin_set = load_seed(os.path.join('../seeds', args.dataset, f'origin'), 4)
        arch, acc = zip(*origin_set)

        sorted_accs = sorted(acc)

        threshold_max = sorted_accs[int(len(sorted_accs) / 2)]
        threshold_min = sorted_accs[int(len(sorted_accs) / 12)]

        iterations = args.iterations
        threshold_gap = (threshold_max - threshold_min) / iterations

        train_set = origin_set
        args.threshold = threshold_max
        search_space = np.load('../seeds/all_combs.npy')
        basenum = 60
        for iter in range(iterations):
            train_x, train_y = seed_loader(train_set, args.threshold)
            # Train GBDT-NAS
            print(f'Train GBDT-NAS')
            lgb_train = lgb.Dataset(train_x, train_y)
            gbm = lgb.train(gbm_params, train_set=lgb_train, num_boost_round=args.num_boost_round)
            print(f'Train GBDT-NAS Done')

            def predict_combs(comb):
                preds = gbm.predict([comb], num_iteration=gbm.best_iteration)
                return preds[0]

            sorted_combs = sorted(search_space, key=lambda x: predict_combs(x), reverse=True)
            length = int(0.5 * len(sorted_combs))
            search_space = sorted_combs[0:length]
            np.save(f'../seeds/comb_{length}.npy', search_space)

            samples = search_space[0:basenum]

            dataset_config = config_dataloader(args)
            searcher = Random_NAS(*dataset_config, samples)
            searcher.run()

            train_set = load_seed(os.path.join('../seeds', args.dataset, f'iteratively_{args.threshold}'), 4)
            args.threshold -= threshold_gap
            search_space = np.load(f'../seeds/comb_{length}.npy', search_space)


    elif args.mode == 'one-shot':
        noisy_train_set = load_seed('../seeds/PEMS07_1/PEMS07_1', 15)
        noisy_comb_to_EDF = generate_EDF(noisy_train_set, args.threshold)

        # filter
        def comb_filter(comb_edf):
            new_comb_edf = {}
            combs = np.concatenate([np.load('../seeds/PEMS07_1/PEMS07_1/selected_combs_3750.npy'), np.load(
                '../seeds/PEMS07_1/PEMS07_1/remained_combs_in_3750_from_selected_combs_in_7500.npy')], axis=0)
            for comb in combs:
                str_comb = ''.join(str(v) for v in comb)
                new_comb_edf[str_comb] = comb_edf[str_comb]

            return new_comb_edf

        new_comb_edf = comb_filter(noisy_comb_to_EDF)
        train_x, train_y = seed_loader(new_comb_edf)

        cnt = 0
        for comb_edf in new_comb_edf.values():
            if comb_edf == 0.0:
                cnt+=1
        # Train GBDT-NAS
        print(f'Train GBDT-NAS')
        lgb_train = lgb.Dataset(train_x, train_y)
        gbm = lgb.train(gbm_params, train_set=lgb_train, num_boost_round=args.num_boost_round)

        comb_space = np.load('../seeds/PEMS07_1/PEMS07_1/combs_3750.npy')

        def predict_combs(comb):
            preds = gbm.predict([comb], num_iteration=gbm.best_iteration)
            return preds[0]

        sorted_combs = sorted(comb_space, key=lambda x: predict_combs(x), reverse=True)

        combs_1875 = sorted_combs[:1875]

        np.save('../seeds/PEMS07_1/PEMS07_1/combs_1875.npy', combs_1875)

if __name__ == '__main__':
    main()
