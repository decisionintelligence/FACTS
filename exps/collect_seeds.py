import os
import argparse
import numpy as np

import torch
import time
import random
import json
import matplotlib.pyplot as plt
import lightgbm as lgb

import utils
from utils import masked_mae, masked_mape, masked_rmse, metric, \
    config_dataloader
from NAS_Net.genotypes import PRIMITIVES
from NAS_Net.st_net import Network
from Task_Config import Task_Config
from tqdm import tqdm

from keras.utils import to_categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for generating clean set')
parser.add_argument('--dataset', type=str, default='PEMS07_1/PEMS07_1',
                    help='the location of  dataset')
parser.add_argument('--datatype', type=str, default='subset',
                    help='type of dataset')
parser.add_argument('--mode', type=str, default='manual',
                    help='the training mode')
parser.add_argument('--ap', type=str, default='GIN',
                    help='the ap choice:[GIN,PINAT]')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--sample_num', type=int, default=20, help='number of archs')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=42)
parser.add_argument('--hid_dim', type=int, default=32,
                    help='for residual_channels and dilation_channels')
parser.add_argument('--randomadj', type=bool, default=True,
                    help='whether random initialize adaptive adj')
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--layers', type=int, default=4, help='number of cells')
parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
# parser.add_argument('--lr_min', type=float, default=0.0, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
# parser.add_argument('--grad_clip', type=float, default=5,
#                     help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--exp_id', type=int, default=1, help='the exp_id used to identify the experiment')
parser.add_argument('--gpu_id', type=int, default=1,
                    help='the gpu id：1-8, used to tell the difference between different logs')
parser.add_argument('--range', nargs='+', type=int, default=[8820, 8821])
args = parser.parse_args()
torch.set_num_threads(3)
num_ops = len(PRIMITIVES)
print(f'number of op: {num_ops}')
num_edges = 1 + 2 * (args.steps - 1)


def generate_EDF(arch_seeds, threshold):
    """from (arch, mae) to (comb, EDF)"""

    # collect (comb, acc_list) pair
    comb_to_accs = dict()
    accs = []
    for arch_acc in arch_seeds:
        comb, task, arch, acc = arch_acc
        _, op = zip(*arch)
        comb = to_categorical(op, num_classes=len(PRIMITIVES), dtype=int)
        comb = np.sum(comb, axis=0).tolist()
        str_comb = ''.join(str(v) for v in comb)
        comb_to_accs.setdefault(str_comb, []).append(acc)
        accs.append(acc)


    sorted_accs = sorted(accs)
    # threshold_e = sorted_accs[int(len(sorted_accs) * threshold_ratio)]
    # threshold_50 = sorted_accs[int(len(sorted_accs) * 0.5)]
    # threshold_25 = sorted_accs[int(len(sorted_accs) * 0.25)]
    # threshold_12 = sorted_accs[int(len(sorted_accs) * 0.125)]
    # threshold_6 = sorted_accs[int(len(sorted_accs) * 0.0625)]
    # threshold_3 = sorted_accs[int(len(sorted_accs) * 0.03125)]
    # threshold_1 = sorted_accs[int(len(sorted_accs) * 0.015625)]

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


def seed_loader(noisy_comb_to_EDF):
    combs, accs = zip(*list(noisy_comb_to_EDF.items()))

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    normed_accs = [(i - mean_acc) / std_acc for i in accs]
    x = np.array([[int(v) for v in s] for s in combs])
    y = np.array(normed_accs)

    return x, y


def get_splits(num):

    all_splits = []

    def partition(n, search_start=1, res=[]):
        # 整数拆分
        for i in range(search_start, n + 1):
            if i == n:
                print([*res, n])
                all_splits.append([*res, n])
                return
            if i > n:
                return
            partition(n - i, search_start=i, res=res + [i])

    partition(num)
    return all_splits


def permuteUnique(nums):
    ans = [[]]
    for n in nums:
        ans = [l[:i] + [n] + l[i:]
               for l in ans
               for i in range((l + [n]).index(n) + 1)]
    return ans


def get_combs(num_ops, splits):

    all_combs = []
    for split in splits:
        if num_ops < len(split):
            continue
        fill_length = num_ops - len(split)
        filled_split = [0] * fill_length + split
        perm_split = permuteUnique(filled_split)
        all_combs.extend(perm_split)
    return all_combs


def get_all_topos(num_nodes):
    def get_topos(arr1, arrlist):
        if arrlist:
            string = []
            for x in arr1:
                for y in arrlist[0]:
                    string.append(x + [y])
            result = get_topos(string, arrlist[1:])
            return result
        else:
            return arr1

    buckets = [list(range(i)) for i in range(1, num_nodes)]
    topos = get_topos([buckets[0]], buckets[1:])
    return topos


def get_base_genos(topos):
    genos = []
    for topo in topos:
        geno = [(0, 0)]
        for i in range(len(topo)):
            geno.append((topo[i], 0))
            geno.append((i + 1, 0))
        genos.append(geno)
    return genos


def get_archs(comb):

    op_list = []
    for i in range(num_ops):
        op_list += [i] * comb[i]
    all_archs = permuteUnique(op_list)

    topos = get_all_topos(args.steps)
    base_genos = get_base_genos(topos)

    all_genos = []
    for arch in all_archs:
        for base_geno in base_genos:
            geno = []
            for i in range(args.steps):
                if i == 0:
                    op = arch[i]
                    geno.extend([(base_geno[i][0], op)])
                else:
                    op1 = arch[2 * i - 1]
                    op2 = arch[2 * i]
                    prenode1 = base_geno[2 * i - 1][0]
                    prenode2 = i
                    geno.extend([(prenode1, op1), (prenode2, op2)])
            all_genos.append(geno)

    random.shuffle(all_genos)
    return all_genos


class Random_NAS:
    def __init__(self):

        utils.set_seed(args.seed)
        if args.cuda:
            torch.backends.cudnn.deterministic = True

        self.save_dir = os.path.join('../seeds', 'pretrain')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @staticmethod
    def configure_task(task_id):
        for dic in Task_Config:
            if dic['id'] == task_id:
                args.datatype = "subsets"
                args.dataset = dic["dataset_name"]
                args.seq_len = dic["seq_len"]
                args.batch_size = dic["batch_size"]

        return config_dataloader(args)

    def run(self):

        args.epochs = 15

        sampled_seeds = np.load(f'../seeds/pretrain/sampled_archs_from_selected_combs_3750.npy',
                                allow_pickle=True)
        Lborder, Rborder = args.range
        meta_archs = sampled_seeds[Lborder:Rborder]

        labeled_archs = []
        for i in range(len(meta_archs)):
            print(f'arch number: {i}')
            meta_arch = meta_archs[i]
            task_id = meta_arch[1]
            arch = meta_arch[2]
            comb_id = meta_arch[0]

            self.dataloader, self.adj_mx, self.scaler, _, self.geo_mask, self.sem_mask = self.configure_task(
                task_id)

            info1, info2, info3 = train_arch_from_scratch(self.dataloader, self.adj_mx, self.scaler, arch,
                                                          self.geo_mask, self.sem_mask)

            labeled_archs.append(
                {"comb": comb_id, "task": task_id, "arch": np.array(arch).tolist(), "info": np.array(info2).tolist()})
            with open(self.save_dir + f'/cellout_clean_3750_{args.gpu_id}.json', "w") as fw:
                json.dump(labeled_archs, fw)


def main():
    all_combs = np.load('../seeds/pretrain/all_combs.npy')
    np.random.shuffle(all_combs)
    selected_combs_12w = []
    for comb in all_combs:
        if len(selected_combs_12w) == 100:
            break
        if len(get_archs(comb)) < 200:
            continue
        selected_combs_12w.append(comb)

    new_selected_combs_12w = [[i, list(selected_combs_12w[i])] for i in range(len(selected_combs_12w))]
    np.save('../seeds/pretrain/selected_combs_12w.npy', np.array(new_selected_combs_12w, dtype=object),
            allow_pickle=True)
    new_selected_combs_12w = np.load('../seeds/pretrain/selected_combs_12w.npy', allow_pickle=True)

    all_archs = []
    for comb_id, comb in new_selected_combs_12w:
        archs = get_archs(comb)
        all_archs.append([comb_id, archs])
    np.save(f'../seeds/pretrain/all_archs_from_selected_combs_12w.npy',
            np.array(all_archs, dtype=object), allow_pickle=True)

    selected_combs_archs = np.load(f'../seeds/pretrain/all_archs_from_selected_combs_12w.npy',
                                   allow_pickle=True)
    clean_seeds = []
    for id, (comb_id, archs) in enumerate(selected_combs_archs):
        selected_archs = archs[:100]
        comb_archs = [[comb_id, i, selected_archs[i]] for i in range(len(selected_archs))]
        clean_seeds.extend(comb_archs)

    np.save(f'../seeds/pretrain/sampled_archs_from_selected_combs_12w.npy', np.array(clean_seeds, dtype=object),
            allow_pickle=True)
    sampled_archs_from_selected_combs_12w = np.load(f'../seeds/pretrain/sampled_archs_from_selected_combs_12w.npy',
                                                    allow_pickle=True)

    noisy_train_set = load_seed('../seeds/pretrain', 15)

    def comb_filter(train_set):
        combs_new = np.load('../seeds/pretrain/selected_combs_7500.npy', allow_pickle=True)
        combs_old = np.load('../seeds/pretrain/remained_combs_in_7500_from_selected_combs_in_15000.npy',
                            allow_pickle=True)
        comb_id_collection = [value[0] for value in combs_new] + [value[0] for value in combs_old]
        new_train_set = []
        for seed in train_set:
            if seed[0] in comb_id_collection:
                new_train_set.append(seed)

        return new_train_set

    noisy_train_set = comb_filter(noisy_train_set)

    scaler_train_set = task_scaler(noisy_train_set)

    length = int(0.03125 * len(scaler_train_set))

    sorted_scaler_train_set = sorted(scaler_train_set, key=lambda x: x[-1])

    index = length
    for id, seed in enumerate(sorted_scaler_train_set):
        if seed[-1] != 0:
            index = length + id
            break

    EDF_threshold = sorted_scaler_train_set[index][-1]
    noisy_comb_to_EDF = generate_EDF(scaler_train_set, EDF_threshold)

    train_x, train_y = seed_loader(noisy_comb_to_EDF)

    cnt = 0
    for comb_edf in noisy_comb_to_EDF.values():
        if comb_edf == 0.0:
            cnt += 1
    # Train GBDT-NAS
    print(f'Train GBDT-NAS')
    lgb_train = lgb.Dataset(train_x, train_y)

    gbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    gbm = lgb.train(gbm_params, train_set=lgb_train, num_boost_round=100)

    comb_space = np.load('../seeds/pretrain/combs_7500.npy')

    def predict_combs(comb):
        preds = gbm.predict([comb], num_iteration=gbm.best_iteration)
        return preds[0]

    sorted_combs = sorted(comb_space, key=lambda x: predict_combs(x), reverse=True)

    combs_slice = sorted_combs[:7500]

    np.save('../seeds/pretrain/combs_3750.npy', combs_slice)

    selected_combs = []
    # 先挑comb
    for comb in combs_slice:
        if len(selected_combs) == 100:
            break
        if len(get_archs(comb)) < 200:
            continue
        selected_combs.append(comb)

    new_selected_combs = [[i + 500, list(selected_combs[i])] for i in range(len(selected_combs))]
    np.save('../seeds/pretrain/selected_combs_3750.npy', np.array(new_selected_combs, dtype=object),
            allow_pickle=True)
    new_selected_combs = np.load('../seeds/pretrain/selected_combs_3750.npy', allow_pickle=True)

    all_archs = []
    for comb_id, comb in new_selected_combs:
        archs = get_archs(comb)
        all_archs.append([comb_id, archs])
    np.save(f'../seeds/pretrain/all_archs_from_selected_combs_3750.npy',
            np.array(all_archs, dtype=object), allow_pickle=True)

    selected_combs_archs = np.load(f'../seeds/pretrain/all_archs_from_selected_combs_3750.npy',
                                   allow_pickle=True)
    clean_seeds = []
    # 对于每个comb，挑archs
    for id, (comb_id, archs) in enumerate(selected_combs_archs):
        selected_archs = archs[:100]
        comb_archs = [[comb_id, i, selected_archs[i]] for i in range(len(selected_archs))]
        clean_seeds.extend(comb_archs)

    np.save(f'../seeds/pretrain/sampled_archs_from_selected_combs_3750.npy', np.array(clean_seeds, dtype=object),
            allow_pickle=True)
    sampled_archs_from_selected_combs = np.load(f'../seeds/pretrain/sampled_archs_from_selected_combs_3750.npy',
                                                allow_pickle=True)

    old_select_combs_1 = np.load(f'../seeds/pretrain/selected_combs_7500.npy', allow_pickle=True)
    old_select_combs_2 = np.load(f'../seeds/pretrain/remained_combs_in_7500_from_selected_combs_in_15000.npy',
                                 allow_pickle=True)
    old_select_combs = np.concatenate([old_select_combs_1, old_select_combs_2], axis=0)
    remained_combs = []
    for comb_id, comb in old_select_combs:
        for id, comb1 in enumerate(combs_slice):
            if (comb == comb1).all():
                remained_combs.append([comb_id, comb])

    np.save('../seeds/pretrain/remained_combs_in_3750_from_selected_combs_in_7500.npy',
            np.array(remained_combs, dtype=object), allow_pickle=True)

    searcher = Random_NAS()
    searcher.run()


def train_arch_from_scratch(dataloader, adj_mx, scaler, arch, geo_mask, sem_mask):
    model = Network(adj_mx, args, arch, geo_mask, sem_mask)
    print(model.state_dict())

    if args.cuda:
        model = model.cuda()

    # train
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    train_metrics_list = []
    valid_metrics_list = []
    test_metrics_list = []
    for epoch_num in range(args.epochs):
        print(f'epoch num: {epoch_num}')
        model = model.train()

        dataloader['train_loader'].shuffle()
        t2 = time.time()
        train_loss = []
        train_rmse = []
        train_mape = []
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            x = torch.Tensor(x).to(DEVICE)

            x = x.transpose(1, 3)

            y = torch.Tensor(y).to(DEVICE)  # [64, 12, 207, 2]

            y = y.transpose(1, 3)[:, 0, :, :]

            optimizer.zero_grad()
            output = model(x)  # [64, 12, 207, 1]
            output = output.transpose(1, 3)
            y = torch.unsqueeze(y, dim=1)
            predict = scaler.inverse_transform(output)  # unnormed x

            loss = masked_mae(predict, y, 0.0)  # y也是unnormed
            train_loss.append(loss.item())
            rmse = masked_rmse(predict, y, 0.0)
            train_rmse.append(rmse.item())
            mape = masked_mape(predict, y, 0.0)
            train_mape.append(mape.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        train_metrics_list.append((np.mean(train_loss), np.mean(train_rmse), np.mean(train_mape)))
        print(f'train epoch time: {time.time() - t2}')

        # eval
        with torch.no_grad():
            model = model.eval()

            valid_loss = []
            valid_rmse = []
            valid_mape = []
            for i, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                x = torch.Tensor(x).to(DEVICE)
                x = x.transpose(1, 3)
                y = torch.Tensor(y).to(DEVICE)
                y = y.transpose(1, 3)[:, 0, :, :]  # [64, 207, 12]

                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y = torch.unsqueeze(y, dim=1)
                predict = scaler.inverse_transform(output)

                loss = masked_mae(predict, y, 0.0)
                rmse = masked_rmse(predict, y, 0.0)
                mape = masked_mape(predict, y, 0.0)
                valid_loss.append(loss.item())
                valid_rmse.append(rmse.item())
                valid_mape.append(mape.item())
            valid_metrics_list.append((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))
            print((np.mean(valid_loss), np.mean(valid_rmse), np.mean(valid_mape)))

        # test
        # if np.mean(valid_rmse) < 50.3:
        with torch.no_grad():
            model = model.eval()

            y_p = []
            y_t = torch.Tensor(dataloader['y_test']).to(DEVICE)
            y_t = y_t.transpose(1, 3)[:, 0, :, :]
            for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
                x = torch.Tensor(x).to(DEVICE)
                x = x.transpose(1, 3)

                # x = nn.functional.pad(x, (1, 0, 0, 0))
                output = model(x)
                output = output.transpose(1, 3)  # [64, 1, 207, 12]
                y_p.append(output.squeeze(1))

            y_p = torch.cat(y_p, dim=0)
            y_p = y_p[:y_t.size(0), ...]

            amae = []
            amape = []
            armse = []
            for i in range(args.seq_len):
                pred = scaler.inverse_transform(y_p[:, :, i])
                real = y_t[:, :, i]
                metrics = metric(pred, real)
                print(f'{i + 1}, MAE:{metrics[0]}, MAPE:{metrics[1]}, RMSE:{metrics[2]}')
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

            test_metrics_list.append((np.mean(amae), np.mean(armse), np.mean(amape)))
            print(f'On average over 12 horizons, '
                  f'Test MAE: {np.mean(amae)}, Test MAPE: {np.mean(amape)}, Test RMSE: {np.mean(armse)}')

    return train_metrics_list, valid_metrics_list, test_metrics_list


if __name__ == '__main__':
    main()
