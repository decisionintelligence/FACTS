import numpy as np
import torch
import torch.nn as nn
import json

import utils
from NAS_Net.ArchPredictor.gin_based_arch_predictor import ArchPredictor, train_baseline_epoch, evaluate
from NAS_Net.genotypes import PRIMITIVES
from utils import NAC_DataLoader, get_archs, standard_normalization, SPLLoss, DynamicWeightedLoss
import os
import argparse
from tqdm import tqdm, trange
import random
from scipy.stats import spearmanr

parser = argparse.ArgumentParser(description='Args for training ap predictor')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--ap_lr', type=float, default=0.001)  # 0.001+adam or 0.0001+adam
parser.add_argument('--layers', type=int, default=4)  # 0.001+adam or 0.0001+adam
parser.add_argument('--d_model', type=int, default=256)  # 0.001+adam or 0.0001+adam

parser.add_argument('--dataset', type=str, default='pems/PEMS08')  # 0.001+adam or 0.0001+adam
parser.add_argument('--mode', type=str, default='noisy', help='the mode of the arch predictor')

parser.add_argument('--infer_batch_size', type=int, default=2560)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--exp_id', type=int, default=2, help='the exp_id used to identify the experiment')
parser.add_argument('--valid', type=str, default='gap', help='the mode of valid set [gap uniform]')
parser.add_argument('--addition', type=int, default=0, help='the addition of the train set')
parser.add_argument('--sup', type=float, default=50, help='the max mae')
parser.add_argument('--gap', type=float, default=20.5, help='the gap when the weight==1')
parser.add_argument('--order', type=float, default=8, help='the order of the function')

parser.add_argument('--loss', type=str, default='L1', help='[L1 MSE]')
parser.add_argument('--curriculum', type=str, default='None', help='Use curriculum learning [CL,SPL,None]')

parser.add_argument('--valid_metric', type=str, default='spearmanr', help='[spearmanr kendall]')

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_another_top4000_noisy_set(epoch):
    noisy_set = []
    for i in range(4):
        # dir = f'cellout_drop_noisy_{i}.json'
        dir = f'../seeds/final4000archs/08_final4000archs_{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch + 1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < args.sup:
                noisy_set.append((arch, mae))

    # np.random.shuffle(noisy_set)
    return noisy_set


def load_top4000_noisy_set(epoch):
    # 5 epochs
    noisy_set = []
    for i in range(24):
        # dir = f'cellout_drop_noisy_{i}.json'
        dir = f'../seeds/pems08_top4000/cellout_drop_noisy_{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch + 1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < args.sup:
                noisy_set.append((arch, mae))

    return noisy_set


def load_noisy_set(epoch):
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
            if mae < args.sup:
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
            if mae < args.sup:
                noisy_set.append((arch, mae))
    return noisy_set


def load_set(data_dir, epoch):
    set = []
    arch_pairs = []
    files = os.listdir(data_dir)
    for file in files:
        if 'drop' in file:
            with open(os.path.join(data_dir, file), "r") as f:
                archs = json.load(f)
            arch_pairs += archs

    for arch_pair in arch_pairs:
        arch = arch_pair['arch']
        info = arch_pair['info'][:epoch]
        mae = sorted(info, key=lambda x: x[0])[0][0]
        if mae < args.sup:
            set.append((arch, mae))

    train_set = []
    valid_set = []
    for i, (arch, mae) in enumerate(set):
        small_gap = 0
        for j, (arch2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((arch, mae))
        else:
            train_set.append((arch, mae))

    return train_set, valid_set


def filter_train_set(train_set):
    cnt_18_19 = 0
    cnt_19_20 = 0
    cnt_20_21 = 0
    cnt_geq_21 = 0
    new_train_set = []
    sup = 3000
    for pair in train_set:
        if 18.5 <= pair[1] < 19:
            if cnt_18_19 < sup:
                new_train_set.append(pair)
                cnt_18_19 += 1
        elif 19 <= pair[1] < 20:
            if cnt_19_20 < sup:
                new_train_set.append(pair)
                cnt_19_20 += 1
        elif 20 <= pair[1] < 21:
            if cnt_20_21 < sup:
                new_train_set.append(pair)
                cnt_20_21 += 1
        else:
            if cnt_geq_21 < sup:
                new_train_set.append(pair)
                cnt_geq_21 += 1

    return new_train_set


def filter_with_sup(set):
    new_set = []
    for arch, acc in set:
        if acc < args.sup:
            new_set.append((arch, acc))

    return new_set


def generate_arch_space():
    search_comb = np.load('../seeds/pems08_top_1000_comb.npy')
    search_archs = []
    for comb in search_comb:
        archs = get_archs(comb)
        search_archs += archs

    print(len(search_archs))
    padding_num = args.infer_batch_size - len(search_archs) % args.infer_batch_size
    padding = [search_archs[-1]] * padding_num
    search_archs += padding
    search_archs = np.array(search_archs)
    np.save('../seeds/pems08_top_1000_comb_archs.npy', search_archs)
    # slices = [search_archs[i:i + args.infer_batch_size] for i in
    #           range(0, len(search_archs) - args.infer_batch_size, args.infer_batch_size)]


def main():
    utils.set_seed(args.seed)

    ap = ArchPredictor(n_nodes=None, n_ops=len(PRIMITIVES), n_layers=args.layers, embedding_dim=args.d_model).to(DEVICE)

    mode = 'none' if args.curriculum != 'None' else 'mean'

    if args.loss == 'L1':
        criterion = nn.L1Loss(reduction=mode)
    else:
        criterion = nn.MSELoss(reduction=mode)

    optimizer = torch.optim.Adam(ap.parameters(), lr=args.ap_lr, betas=(0.5, 0.999), weight_decay=5e-4)

    model_dir = '../NAS_Net/ArchPredictor/ArchPredictor_param/GIN'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.mode == 'noisy':
        torch.set_num_threads(3)
        with open(model_dir + f'/train_log_{args.exp_id}.txt', 'w') as f:
            print(
                f'''The data setting: valid mode:{args.valid}, train set addition:{args.addition}, max_mae threshold:{args.sup} ''',
                file=f)
            print(
                f'''The model setting: gcn layers:{args.layers}, embedding dimension:{args.d_model}''',
                file=f)
            print(
                f'''The training setting: loss:{args.loss}, learning rate:{args.ap_lr}, batch size:{args.batch_size}, valid metric:{args.valid_metric} \n''',
                file=f)
            print(
                f'''The difficulites function setting: gap:{args.gap}, order:{args.order} \n''',
                file=f)
        with open(model_dir + f'/train_info_{args.exp_id}.txt', 'w') as f:
            print('record the train valid test result on checkpoint:', file=f)

        noisy_scaler = standard_normalization()
        noisy_set = load_noisy_set(4)
        xinle_noisy_set = load_another_top4000_noisy_set(4)


        if args.addition != 0:
            if args.addition > 0:
                if args.valid == 'gap':
                    xingjian_train_set = np.load('../seeds/valid_test/train1.npy', allow_pickle=True)
                elif args.valid == 'uniform':
                    xingjian_train_set = np.load('../seeds/valid_test/train.npy', allow_pickle=True)
                max_length = len(xingjian_train_set)
                xingjian_train_set = xingjian_train_set[:min(args.addition, max_length)].tolist()
            else:
                length = len(xinle_noisy_set) + args.addition
                xinle_noisy_set = xinle_noisy_set[:max(0, length)]
                xingjian_train_set = []

        else:
            xingjian_train_set = []

        if args.valid == 'gap':
            xingjian_valid_set = np.load('../seeds/valid_test/valid1.npy', allow_pickle=True)
        elif args.valid == 'uniform':
            xingjian_valid_set = np.load('../seeds/valid_test/valid.npy', allow_pickle=True)

        xingjian_test_set = np.load('../seeds/valid_test/test.npy', allow_pickle=True)

        origin_noisy_train_set = filter_with_sup(noisy_set + xinle_noisy_set + xingjian_train_set)
        origin_noisy_valid_set = filter_with_sup(xingjian_valid_set.tolist())
        origin_noisy_test_set = filter_with_sup(xingjian_test_set.tolist())


        noisy_scaler.fit(origin_noisy_train_set)

        noisy_train_set = noisy_scaler.transform(origin_noisy_train_set)
        noisy_valid_set = noisy_scaler.transform(origin_noisy_valid_set)
        noisy_test_set = noisy_scaler.transform(origin_noisy_test_set)

        train_loader = NAC_DataLoader(noisy_train_set, batch_size=args.batch_size)
        valid_loader = NAC_DataLoader(noisy_valid_set, batch_size=1)
        test_loader = NAC_DataLoader(noisy_test_set, batch_size=1)

        if args.curriculum == 'SPL':
            train_criterion = SPLLoss(criterion=criterion)
        elif args.curriculum == 'CL':
            train_criterion = DynamicWeightedLoss(scaler=noisy_scaler, criterion=criterion, reduction='mean',
                                                  gap=args.gap, order=args.order)
            test_criterion = DynamicWeightedLoss(scaler=noisy_scaler, criterion=criterion, reduction='none',
                                                 gap=args.gap, order=args.order)
        else:
            train_criterion = criterion

        his_metric = -1
        tolerance = 0

        train_noisy_loop = tqdm(range(args.epochs), total=args.epochs, ncols=250,
                                desc='training the GIN-based ArchPredictor with noisy set')

        for epoch in train_noisy_loop:
            train_loss, valid_loss, rho, ken, acc_ratio, add_info = train_baseline_epoch(train_loader, valid_loader, ap,
                                                                                         train_criterion, criterion,
                                                                                         args.curriculum, optimizer)
            train_noisy_loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            train_noisy_loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss, spearmanr_rho=rho, kendall=ken,
                                         acc_rate=acc_ratio)
            if args.valid_metric == 'spearmanr':
                valid_metric = rho[0]
            else:
                valid_metric = ken[0]

            with open(model_dir + f'/train_log_{args.exp_id}.txt', 'a') as f:
                print(
                    f'''Noisy Epoch {epoch}: train loss:{train_loss}, valid loss:{valid_loss}, spearmanr rho:{rho}, kendall:{ken}, acc rate:{acc_ratio}, valid metric {"increases and model is saved" if valid_metric > his_metric else "doesn't increase"}''',
                    file=f)

            if valid_metric > his_metric:

                tolerance = 0

                torch.save(ap.state_dict(),
                           model_dir + f'/arch_predictor_{args.exp_id}.pth')
                train_noisy_loop.set_description(f'valid_metric increases [{his_metric}->{valid_metric}]')
                his_metric = valid_metric

                with open(model_dir + f'/train_info_{args.exp_id}.txt', 'a') as f:
                    print(
                        f'''\nThe last checkpoint is epoch {epoch}\n''',
                        file=f)
                    print(f'The actual and the pred on train set:(ranked by weighted loss)\n', file=f)
                    train_actual, train_result, valid_actual, valid_result = add_info['train_actual'], add_info[
                        'train_result'], add_info['valid_actual'], add_info['valid_result']
                    actual = noisy_scaler.reverse_transform(train_actual)
                    result = noisy_scaler.reverse_transform(train_result)
                    _, train_actual_acc = zip(*train_actual)
                    _, train_result_acc = zip(*train_result)
                    origin_metric = criterion(torch.tensor(train_result_acc), torch.tensor(train_actual_acc))
                    weighted_metric = test_criterion(torch.tensor(train_result_acc), torch.tensor(train_actual_acc))
                    ranked_indices = np.argsort(-weighted_metric)

                    origin_metric = origin_metric[ranked_indices]
                    weighted_metric = weighted_metric[ranked_indices]
                    actual = [actual[i] for i in ranked_indices]
                    result = [result[i] for i in ranked_indices]

                    loss_18_19 = []
                    loss_19_20 = []
                    loss_20_21 = []
                    loss_g_21 = []
                    for i in range(len(actual)):

                        if actual[i][1] < 19:
                            loss_18_19.append(weighted_metric[i])
                        elif 19 <= actual[i][1] < 20:
                            loss_19_20.append(weighted_metric[i])
                        elif 20 <= actual[i][1] < 21:
                            loss_20_21.append(weighted_metric[i])
                        else:
                            loss_g_21.append(weighted_metric[i])

                    print(
                        f'18-19 loss:{np.sum(loss_18_19)},19-20 loss:{np.sum(loss_19_20)},20-21 loss:{np.sum(loss_20_21)}, >21 loss:{np.sum(loss_g_21)}\n',
                        file=f)

                    for i in range(len(train_actual)):
                        print(
                            f'arch:{actual[i][0]},actual:{actual[i][1]},pred:{result[i][1]}, origin train loss:{origin_metric[i]}, weighted train loss:{weighted_metric[i]}',
                            file=f)

                    print(f'\nThe actual and the pred on valid set:\n', file=f)

                    valid_actual = noisy_scaler.reverse_transform(valid_actual)
                    valid_result = noisy_scaler.reverse_transform(valid_result)
                    for i in range(len(valid_actual)):
                        print(
                            f'arch:{valid_actual[i][0]},actual:{valid_actual[i][1]},pred:{valid_result[i][1]}',
                            file=f)

                    test_loss, rho, ken, acc_rate, add_info = evaluate(test_loader, ap, criterion)
                    test_actual, test_result = add_info['test_actual'], add_info['test_result']
                    test_actual = noisy_scaler.reverse_transform(test_actual)
                    test_result = noisy_scaler.reverse_transform(test_result)
                    _, accs = zip(*test_actual)
                    _, preds = zip(*test_result)
                    accs_185 = []
                    preds_185 = []
                    accs_middle = []
                    preds_middle = []
                    accs_190 = []
                    preds_190 = []
                    for i in range(len(accs)):
                        if accs[i] < 18.6:
                            accs_185.append(accs[i])
                            preds_185.append(preds[i])
                        if 18.6 <= accs[i] < 19:
                            accs_middle.append(accs[i])
                            preds_middle.append(preds[i])
                        if accs[i] < 19:
                            accs_190.append(accs[i])
                            preds_190.append(preds[i])

                    rho_185 = spearmanr(accs_185, preds_185)
                    rho_middle = spearmanr(accs_middle, preds_middle)
                    rho_190 = spearmanr(accs_190, preds_190)

                    with open(model_dir + f'/train_log_{args.exp_id}.txt', 'a') as f:
                        print(
                            f'''\nTEST results: test loss:{test_loss}, spearmanr:{rho}, kendall:{ken}, acc rate:{acc_rate}''',
                            file=f)
                        print(
                            f'''TEST results on mae<18.6: spearmanr:{rho_185}''',
                            file=f)
                        print(
                            f'''TEST results on 18.6<mae<19.0: spearmanr:{rho_middle}''',
                            file=f)
                        print(
                            f'''TEST results on mae<19.0: spearmanr:{rho_190}''',
                            file=f)

                    with open(model_dir + f'/train_info_{args.exp_id}.txt', 'a') as f:

                        print(f'\nThe actual and the pred on test set:\n', file=f)

                        for i in range(len(test_actual)):
                            print(
                                f'arch:{test_actual[i][0]},actual:{test_actual[i][1]},pred:{test_result[i][1]}',
                                file=f)

            else:
                tolerance += 1

            if tolerance >= 100:
                break

    if args.mode == 'train':
        # process the noisy data
        torch.set_num_threads(3)
        noisy_scaler = standard_normalization()
        root_path = '../seeds/PEMS07_1/PEMS07_1'
        train_set = np.load(os.path.join(root_path, 'train_1875_2400.npy'), allow_pickle=True)
        valid_set = np.load(os.path.join(root_path, 'valid_1875_2400.npy'), allow_pickle=True)
        train_set = list(train_set)
        valid_set = list(valid_set)
        noisy_scaler.fit(train_set)
        train_set = noisy_scaler.transform(train_set)
        valid_set = noisy_scaler.transform(valid_set)

        train_loader = NAC_DataLoader(train_set, args.batch_size, pad_with_last_sample=True)
        valid_loader = NAC_DataLoader(valid_set, args.batch_size, pad_with_last_sample=True)

        his_rho = 0
        tolerance = 0
        train_noisy_loop = tqdm(range(args.epochs), total=args.epochs, ncols=250,
                                desc='training the GIN-based ArchPredictor with noisy set')
        for epoch in train_noisy_loop:
            train_loss, valid_loss, rho, ken, acc_ratio, additional_info = train_baseline_epoch(train_loader,
                                                                                                valid_loader, ap,
                                                                                                criterion, criterion,
                                                                                                None, optimizer)
            train_noisy_loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            train_noisy_loop.set_postfix(train_loss=train_loss, valid_loss=valid_loss, spearmanr_rho=rho,
                                         acc_rate=acc_ratio)

            with open(model_dir + f'/train_log_{args.exp_id}.txt', 'w' if epoch == 0 else 'a') as f:
                print(
                    f'''Noisy Epoch {epoch}: train loss:{train_loss}, valid loss:{valid_loss}, spearmanr rho:{rho}, acc rate:{acc_ratio}, spearmanr {"increases and model is saved" if rho[0] > his_rho else "doesn't increase"}''',
                    file=f)

            if rho[0] > his_rho:
                tolerance = 0
                torch.save(ap.state_dict(),
                           model_dir + f'/arch_predictor_{args.exp_id}.pth')
                train_noisy_loop.set_description(f'valid_spearmanr increases [{his_rho}->{rho[0]}]')
                his_rho = rho[0]
            else:
                tolerance += 1

            if tolerance >= 30:
                break


    elif args.mode == 'search':

        # ap.load_state_dict(
        #     torch.load(model_dir + f'/arch_predictor_{args.exp_id}_fine_tune.pth'))
        ap.load_state_dict(
            torch.load(model_dir + f'/arch_predictor_{args.exp_id}.pth'))

        ap.eval()
        with torch.no_grad():
            search_archs = np.load('../seeds/PEMS07_1/PEMS07_1/all_archs_from_combs_1875.npy')

            pred_archs = []

            fake_acc = [0] * len(search_archs)
            arch_acc = zip(search_archs, fake_acc)
            infer_loader = NAC_DataLoader(arch_acc, batch_size=args.infer_batch_size)
            for i, (arch, acc) in tqdm(enumerate(infer_loader.get_iterator()), total=infer_loader.num_batch):
                pred_archs += ap(arch).cpu()

            sorted_indices = np.argsort(pred_archs[0:len(search_archs)])
            sorted_archs = search_archs[sorted_indices]
            print(f'the best 10 archs:{sorted_archs[:10]}')

            save_dir = f'../results/searched_archs/{args.dataset}/GIN'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(os.path.join(save_dir, f'param_{args.exp_id}.npy'), sorted_archs[:10])

            with open(os.path.join(save_dir, f'top100_{args.exp_id}.txt'), 'w') as f:
                for i in range(100):
                    print(sorted_archs[i], file=f)

            with open(os.path.join(save_dir, f'bottom100_{args.exp_id}.txt'), 'w') as fw:
                for i in range(100, 0, -1):
                    print(sorted_archs[-i], file=fw)


if __name__ == '__main__':
    main()
    # arch = [(0, 3), (0, 10), (1, 7), (0, 11), (2, 9), (1, 11), (3, 14)]
    # adj, ops = geno_to_adj(arch)
    # print(adj)
    # print(ops)
