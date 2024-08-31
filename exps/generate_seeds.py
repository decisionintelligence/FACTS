import os
import argparse
import numpy as np

import torch
import time
import random
import json
import matplotlib.pyplot as plt

import NAS_Net.st_net_1
import utils
from utils import masked_mae, masked_mape, masked_rmse, metric, \
    config_dataloader
from NAS_Net.genotypes import PRIMITIVES
from NAS_Net.st_net import Network
from tqdm import tqdm
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for generating clean set')
parser.add_argument('--dataset', type=str, default='PEMS07_1/PEMS07_1',
                    help='the location of  dataset')
parser.add_argument('--datatype', type=str, default='npy',
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
parser.add_argument('--range', nargs='+', type=int, default=[0, 1000])
args = parser.parse_args()
torch.set_num_threads(3)
num_ops = len(PRIMITIVES)
print(f'number of op: {num_ops}')
num_edges = 1 + 2 * (args.steps - 1)


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


def load_seed(dir, epoch):
    noisy_set = []
    for i in range(1, 9):
        file_path = os.path.join(dir, f'cellout_clean_12w_{i}.json')
        with open(file_path, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            noisy_set.append((arch, mae))
    return noisy_set


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
    def __init__(self, dataloader, adj_mx, scaler, save_dir, geo_mask, sem_mask, archs=None):
        self.save_dir = save_dir
        self.dataloader = dataloader
        self.adj_mx = adj_mx
        self.scaler = scaler
        self.geo_mask = geo_mask
        self.sem_mask = sem_mask
        self.archs = None

        utils.set_seed(args.seed)
        if args.cuda:
            torch.backends.cudnn.deterministic = True

    @staticmethod
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

    def run(self):

        archs = []
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if args.mode == 'clean_seeds':
            args.epochs = 100
            for i in range(args.sample_num):
                archs.append(Random_NAS.sample_arch())
        elif args.mode == 'noisy_seeds':
            args.epochs = 5
            for i in range(args.sample_num):
                archs.append(Random_NAS.sample_arch())
        elif args.mode == 'train':
            archs = np.load(f'../results/searched_archs/{args.dataset}/{args.ap}/param_{args.exp_id}.npy')

            self.save_dir = '../seeds/params_pool'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        elif args.mode == 'inherit':
            archs = np.load(f'../results/searched_archs/pems/PEMS08/{args.ap}/param_{args.exp_id}.npy')

            for i in range(len(archs)):
                print(f'arch number: {i}')
                arch = archs[i]

                _, _, _, inherit_model = train_inherit_model(self.dataloader, self.adj_mx, self.scaler, arch,
                                                             self.geo_mask, self.sem_mask)
                t1 = time.time()

                model = Network(self.adj_mx, args, arch, self.geo_mask, self.sem_mask)

                state_dict_A = inherit_model.state_dict()
                state_dict_B = model.state_dict()

                state_dict_B.update({k: v for k, v in state_dict_A.items() if k in state_dict_B})

                model.load_state_dict(state_dict_B)

                info1, info2, info3, model = train_arch_from_scratch(self.dataloader, self.adj_mx, self.scaler, arch,
                                                                     self.geo_mask, self.sem_mask, model=model)
                t2 = time.time()
                time_cost = t2 - t1

                self.result_process(arch, i, info1, info2, info3, time_cost)

                param_dir = os.path.join(self.save_dir, 'params')
                if not os.path.exists(param_dir):
                    os.makedirs(param_dir)

                torch.save(model.state_dict(), os.path.join(param_dir, str(arch) + f'_{args.exp_id}.pth'))

                return

        elif args.mode == 'mean':
            archs = np.load(f'../results/searched_archs/{args.dataset}/{args.ap}/param_{args.exp_id}.npy')

            self.save_dir = '../seeds/params_pool'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

    def result_process(self, arch, id, info1, info2, info3, time_cost):
        clean_set = []
        test_set = []
        train_set = []

        train_set.append({"arch": np.array(arch).tolist(), "info": np.array(info1).tolist()})
        clean_set.append({"arch": np.array(arch).tolist(), "info": np.array(info2).tolist()})
        test_set.append({"arch": np.array(arch).tolist(), "info": np.array(info3).tolist()})
        with open(self.save_dir + f'/train{id}_{args.exp_id}.json', "w") as fw:
            json.dump(train_set, fw)
        with open(self.save_dir + f'/valid{id}_{args.exp_id}.json', "w") as vw:
            json.dump(clean_set, vw)
        with open(self.save_dir + f'/test{id}_{args.exp_id}.json', 'w') as tw:
            json.dump(test_set, tw)

        train_mae = [info1[i][0] for i in range(args.epochs)]
        train_rmse = [info1[i][1] for i in range(args.epochs)]
        train_mape = [info1[i][2] for i in range(args.epochs)]

        valid_mae = [info2[i][0] for i in range(args.epochs)]
        valid_rmse = [info2[i][1] for i in range(args.epochs)]
        valid_mape = [info2[i][2] for i in range(args.epochs)]

        test_mae = [info3[i][0] for i in range(args.epochs)]
        test_rmse = [info3[i][1] for i in range(args.epochs)]
        test_mape = [info3[i][2] for i in range(args.epochs)]
        bestid = np.argmin(valid_mae)

        with open(self.save_dir + f'/log_{args.exp_id}.txt', 'a') as f:
            print(f'arch:{arch}', file=f)
            print(f'total train time:{time_cost}', file=f)
            print(f'best_epoch:{bestid}', file=f)
            print(f'valid mae:{valid_mae[bestid]},rmse:{valid_rmse[bestid]},mape:{valid_mape[bestid]}', file=f)
            print(f'test mae:{test_mae[bestid]},rmse:{test_rmse[bestid]},mape:{test_mape[bestid]}', file=f)

        figure_save_path = self.save_dir
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)

        x = np.array([i for i in range(args.epochs)])

        plt.figure(3 * id)
        plt.plot(x, np.array(train_mae))
        plt.plot(x, np.array(valid_mae))
        plt.plot(x, np.array(test_mae))
        plt.title('mae')
        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.legend(['train', 'valid', 'test'], loc='upper center')
        plt.savefig(os.path.join(figure_save_path, f'mae{id}_{args.exp_id}.png'))
        plt.clf()
        plt.close()

        plt.figure(3 * id + 1)
        plt.plot(x, np.array(train_rmse))
        plt.plot(x, np.array(valid_rmse))
        plt.plot(x, np.array(test_rmse))
        plt.title('rmse')
        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.legend(['train', 'valid', 'test'], loc='upper center')
        plt.savefig(os.path.join(figure_save_path, f'rmse{id}_{args.exp_id}.png'))
        plt.clf()
        plt.close()

        plt.figure(3 * id + 2)
        plt.plot(x, np.array(train_mape))
        plt.plot(x, np.array(valid_mape))
        plt.plot(x, np.array(test_mape))
        plt.title('mape')
        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.legend(['train', 'valid', 'test'], loc='upper center')
        plt.savefig(os.path.join(figure_save_path, f'mape{id}_{args.exp_id}.png'))
        plt.clf()
        plt.close()


def main():
    # Fill in with root output path

    searcher = Random_NAS(*config_dataloader(args))
    searcher.run()


def train_arch_from_scratch(dataloader, adj_mx, scaler, arch, geo_mask, sem_mask, model=None):
    if model is None:
        model = Network(adj_mx, args, arch, geo_mask, sem_mask)

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

        # for name, param in model.named_parameters():
        #     print(name)

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

            loss = masked_mae(predict, y, 0.0)
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

    return train_metrics_list, valid_metrics_list, test_metrics_list, model


def train_inherit_model(dataloader, adj_mx, scaler, arch, geo_mask, sem_mask):
    model = NAS_Net.st_net_1.Network(adj_mx, args, arch, geo_mask, sem_mask)

    if args.cuda:
        model = model.cuda()

    # train
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    train_metrics_list = []
    valid_metrics_list = []
    test_metrics_list = []
    for epoch_num in range(10):
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

            loss = masked_mae(predict, y, 0.0)
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

    return train_metrics_list, valid_metrics_list, test_metrics_list, model


if __name__ == '__main__':
    main()
