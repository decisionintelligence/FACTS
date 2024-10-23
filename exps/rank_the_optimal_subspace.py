import argparse
import os
import time
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from functools import cmp_to_key
from pathlib import Path
import concurrent.futures
import utils
from NAS_Net.ArchPredictor.gin_based_arch_predictor import ArchPredictor, train_baseline_epoch, evaluate

from NAS_Net.genotypes import PRIMITIVES
from utils import AP_DataLoader, get_archs
from exist_file_map import *
from scipy.stats import spearmanr
from tqdm import tqdm
import heapq
from Task_Config import Task_Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for zero-cost NAS')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ap_lr', type=float, default=0.0001)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--d_model', type=int, default=128)

parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--dataset', type=str, default='pems/PEMS03', help='location of dataset')
parser.add_argument('--sample_scale', type=int, default=200000, help='the number of samples')
parser.add_argument('--mode', type=str, default='pretrain', help='the mode of the comparator')
parser.add_argument('--loader_mode', type=str, default='quadratic', help='[quadratic linear]')

parser.add_argument('--seq_len', type=int, default=12, help='the sequence length of the sample')
parser.add_argument('--exp_id', type=int, default=-1, help='the exp_id used to identify the experiment')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_threads', type=int, default=5)
parser.add_argument('--top_k', type=int, default=5)

args = parser.parse_args()

torch.set_num_threads(3)


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
                comb = arch_pair['comb']
                task = arch_pair['task']
                mae = sorted(info, key=lambda x: x[0])[0][0]
                noisy_set.append([comb, task, arch, mae])
    return noisy_set


def task_scaler(seed_set):
    # [comb, task, arch, mae]
    task_dict = dict()
    for seed in seed_set:
        task_dict.setdefault(seed[1], []).append([seed[0], seed[2], seed[3]])

    for key, value in task_dict.items():
        task_dict[key] = sorted(value, key=lambda x: x[0])

    for key, value_list in task_dict.items():
        min_value = min(map(lambda x: x[2], value_list))
        max_value = max(map(lambda x: x[2], value_list))

        for id, seed in enumerate(seed_set):
            if seed[1] == key:
                seed_set[id][3] = (seed[3] - min_value) / (max_value - min_value)

    return seed_set


def sample_arch():
    num_ops = len(PRIMITIVES)
    n_nodes = args.steps

    arch = []
    for i in range(n_nodes):
        if i == 0:
            ops = np.random.choice(range(num_ops), 1)
            nodes = np.random.choice(range(i + 1), 1)
            arch.extend([(nodes[0], ops[0])])
        else:
            ops = np.random.choice(range(num_ops), 2)
            nodes = np.random.choice(range(i), 1)
            # nodes = np.random.choice(range(i + 1), 2, replace=False)
            arch.extend([(nodes[0], ops[0]), (i, ops[1])])

    return arch


def calculate_spearmanr(ap, task_dict, set):
    features, _, _ = zip(*set)
    task_feature = task_dict[features[0]]

    ap.eval()

    def compare(x0, x1):
        with torch.no_grad():
            outputs = ap([x0[1]], [x1[1]], [task_feature])
            pred = torch.round(outputs)
        if pred == 0:
            return 1
        else:
            return -1

    # 先按照ap的比较规则对于所有排序，得到一个从小到大的顺序rank
    ap_sorted_set = sorted(set, key=cmp_to_key(compare))
    ap_sorted_index = list(range(len(ap_sorted_set)))

    # 再根据真实的metric指标进行排序，然后获得每一个位置被排序后的大小rank
    _, _, accs = zip(*ap_sorted_set)
    acc_sorted_index = np.argsort(accs)
    rank_index = [0] * len(ap_sorted_set)
    for i, id in enumerate(acc_sorted_index):
        rank_index[id] = i

    rho = spearmanr(ap_sorted_index, rank_index)

    return rho


class normalize_statistical_feature:

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def fit(self, statistical_dict):
        array_list = []
        for key, value in statistical_dict.items():
            array_list.append(value)

        array = np.array(array_list)
        self.mean = np.mean(array, axis=0)
        self.std = np.std(array, axis=0)

    def transform(self, statistical_dict):
        array_list = []

        for key, value in statistical_dict.items():
            array_list.append(value)

        array = np.array(array_list)

        normalized_array = (array - self.mean) / (self.std + self.epsilon)
        for id, (key, value) in enumerate(statistical_dict.items()):
            statistical_dict[key] = normalized_array[id]
        return statistical_dict

    def inverse_transform(self, statistical_dict):
        array_list = []

        for key, value in statistical_dict.items():
            array_list.append(value)

        array = np.array(array_list)

        normalized_array = (array * (self.std + self.epsilon)) + self.mean
        for id, (key, value) in enumerate(statistical_dict.items()):
            statistical_dict[key] = normalized_array[id]


def main():
    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    utils.set_seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    DataLoader = AP_DataLoader

    ap = ArchPredictor(n_nodes=None, n_ops=len(PRIMITIVES), n_layers=args.layers, embedding_dim=args.d_model).to(DEVICE)
    # criterion = list_mle
    criterion = nn.MSELoss()
    ap_optimizer = optim.Adam(ap.parameters(), lr=args.ap_lr)
    model_dir = '../NAS_Net/ArchPredictor/ArchPredictor_param/GIN'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.mode == 'pretrain':
        with open(model_dir + f'/train_log_{args.exp_id}.txt', 'w') as f:
            print(
                f'''pretrain the Task-Oriented ap:
                the args:{args}''',
                file=f)

        ap.train()

        noisy_train_set = load_seed('../seeds/pretrain', 15)
        scaler_train_set = task_scaler(noisy_train_set)

        task_dict = {}
        statistics_dict = {}

        for item in Task_Config:
            dataset_name = item['dataset_name']
            task_feature_dir = '../task_feature/{}'.format(dataset_name)
            task_file_path = os.listdir(task_feature_dir)[0]
            task_feature = np.load(os.path.join(task_feature_dir, task_file_path))

            task_dict[dataset_name] = task_feature

            statistics_feature_dir = '../statistical_feature/{}'.format(dataset_name)
            statistics_file_path = os.listdir(statistics_feature_dir)[0]
            statistics_feature = np.load(os.path.join(statistics_feature_dir, statistics_file_path))

            statistics_dict[dataset_name] = statistics_feature

        noisy_set = []
        for seed in scaler_train_set:
            for config in Task_Config:
                if config['id'] == seed[1]:
                    noisy_set.append([config['dataset_name'], seed[-2], seed[-1]])

        scaler = normalize_statistical_feature()
        scaler.fit(statistics_dict)
        statistics_dict = scaler.transform(statistics_dict)

        np.save(os.path.join(model_dir, f'mean_{args.exp_id}.npy'), scaler.mean)
        np.save(os.path.join(model_dir, f'std_{args.exp_id}.npy'), scaler.std)
        np.save(os.path.join(model_dir, f'epsilon_{args.exp_id}.npy'), scaler.epsilon)

        train_loader = DataLoader(noisy_set, task_dict, statistics_dict, args.batch_size)
        his_rho = -1
        tolerance = 0
        train_noisy_loop = tqdm(range(args.epochs), ncols=250, desc='pretrain ap with noisy_set')

        for epoch in train_noisy_loop:
            train_loss, valid_loss, rho, acc = train_baseline_epoch(train_loader,
                                                                    train_loader,
                                                                    ap,
                                                                    criterion,
                                                                    ap_optimizer)

            train_noisy_loop.set_description(f'Epoch {epoch}:')
            train_noisy_loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
            with open(model_dir + f'/train_log_{args.exp_id}.txt', 'a') as f:
                print(
                    f'''Noisy Epoch {epoch}: train loss:{train_loss}, valid loss:{valid_loss}, spearmanr_rho:{rho}, acc:{acc} rho {"increases and model is saved" if rho[0] > his_rho else "doesn't increase"}''',
                    file=f)
            if rho[0] > his_rho:
                train_noisy_loop.set_description(f'valid rho increases [{his_rho}->{rho[0]}]')

                tolerance = 0
                his_rho = rho[0]
                torch.save(ap.state_dict(), model_dir + f"/AP_{args.exp_id}.pth")
            else:
                tolerance += 1
            if tolerance >= 10:
                break

        ap.load_state_dict(torch.load(model_dir + f"/AP_{args.exp_id}.pth"))

        test_loss, rho, acc = evaluate(test_loader=train_loader, ap=ap, criterion=criterion)
        with open(model_dir + f'/train_log_{args.exp_id}.txt', 'a') as f:
            print(
                f'''\nTEST RESULT: test loss:{test_loss},spearmanr_rho:{rho}''',
                file=f)

    elif args.mode == 'search':
        torch.set_num_threads(args.num_threads)
        ap.load_state_dict(torch.load(os.path.join(model_dir, f'ArchPredictor.pth')))

        task_feature = np.load(os.path.join('../task_feature', args.dataset, f'{args.seq_len}_ts2vec_task_feature.npy'))
        statistical_feature = np.load(
            os.path.join('../statistical_feature', args.dataset, 'statistical_feature.npy'))

        statistical_feature = np.squeeze(statistical_feature)

        search_space_path = os.path.join(model_dir, f'archs.npy')
        if not os.path.exists(search_space_path):
            search_space = np.load(os.path.join(model_dir, 'combs_3750.npy'), allow_pickle=True)
            archs = []
            for comb in search_space:
                current_archs = get_archs(comb)
                archs.extend(current_archs)
            archs = archs[:7000000]
            np.save(search_space_path, archs, allow_pickle=True)
        else:
            np.load(search_space_path, allow_pickle=True)

        slice_len = args.inference_batch_size
        slice_num = len(archs) // slice_len
        archs_slices = [
            archs[i * slice_len:(i + 1) * slice_len] if i < slice_num - 1 else archs[i * slice_len:]
            for i in range(slice_num)]

        outputs = []
        ap.eval()
        with torch.no_grad():
            for arch_batch in archs_slices:
                output = ap(arch_batch, [task_feature] * len(arch_batch), [statistical_feature] * len(arch_batch))
                outputs.extend(output.detach().cpu().numpy())

        sorted_archs = list(sorted(zip(outputs, archs), key=lambda x: x[0]))
        _, archs = zip(*sorted_archs)
        print(archs[:args.top_k][1])
        param_dir = os.path.join('../results/searched_archs', args.dataset)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        np.save(param_dir + f'/{args.seq_len}_param_{args.exp_id}.npy', np.array(sorted_archs[:args.top_k]))

        with open(param_dir + f'/{args.seq_len}_param_{args.exp_id}.txt', 'w') as f:
            for arch in sorted_archs[:args.top_k]:
                print(arch, file=f)


if __name__ == '__main__':
    main()
