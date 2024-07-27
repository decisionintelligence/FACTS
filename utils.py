import time
from scipy.sparse import csc_matrix
import pickle
import csv
from pathlib import Path
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import copy
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from distutils.util import strtobool
from scipy.stats import skew, kurtosis, entropy
from scipy.special import softmax
from NAS_Net.genotypes import PRIMITIVES
import random
import os
import dgl
import h5py
from torch import Tensor
import sklearn.preprocessing


def config_dataloader(args):
    # Fill in with root output path
    adj_mx = np.zeros((args.num_nodes, args.num_nodes))
    if args.datatype == 'csv':
        data_dir = os.path.join('../data', args.dataset + '.csv')
        if args.dataset == 'sz_taxi/sz_speed':
            adj_mx = pd.read_csv('../data/sz_taxi/sz_adj.csv', header=None).values.astype(np.float32)
        if args.dataset == 'los_loop/los_speed':
            adj_mx = pd.read_csv('../data/los_loop/los_adj.csv', header=None).values.astype(np.float32)
    elif args.datatype == 'txt':
        data_dir = os.path.join('../data', args.dataset + '.txt')

    elif args.datatype == 'npz':
        data_dir = os.path.join('../data', args.dataset + '.npz')
        adj_dir = os.path.join('../data', args.dataset + '.csv')
        if args.dataset == 'NYC_TAXI/NYC_TAXI':
            adj_mx = pd.read_csv(adj_dir, header=None).values.astype(np.float32)
        elif args.dataset == 'NYC_BIKE/NYC_BIKE':
            adj_mx = pd.read_csv(adj_dir, header=None).values.astype(np.float32)
        elif args.dataset == 'pems/PEMS03':
            adj_mx = get_adj_matrix(adj_dir, args.num_nodes, id_filename='../data/pems/PEMS03.txt')
        elif args.dataset == 'PEMSD7M/PEMSD7M':
            adj_dir = os.path.join('../data/PEMSD7M/adj.npz')
            adj_mx = load_PEMSD7_adj(adj_dir)
        else:
            adj_mx = get_adj_matrix(adj_dir, args.num_nodes)
    elif args.datatype == 'npy':
        data_dir = os.path.join('../data', args.dataset + '.npy')
        adj_dir = os.path.join('../data', args.dataset + '_adj.npy')
        if os.path.exists(adj_dir):
            adj_mx = np.load(os.path.join('../data', args.dataset + '_adj.npy'))
    elif args.datatype == 'tsf':
        data_dir = os.path.join('../data', args.dataset + '.tsf')
    elif args.datatype == 'h5':
        data_dir = os.path.join('../data', args.dataset + '.h5')
        if 'metr-la' in args.dataset:
            adj_dir = '../data/METR-LA/adj_mx.pkl'
            _, _, adj_mx = load_adj(adj_dir)
        elif 'pems-bay' in args.dataset:
            adj_dir = '../data/PEMS-BAY/adj_mx_bay.pkl'
            _, _, adj_mx = load_adj(adj_dir)

    elif args.datatype == 'subsets':
        data_dir = os.path.join('../subsets', args.dataset + '.npy')
        adj_dir = os.path.join('../subsets', args.dataset + '_adj.npy')
        if os.path.exists(adj_dir):
            adj_mx = np.load(adj_dir)
            args.num_nodes = adj_mx.shape[0]
        else:
            data = np.load(data_dir)
            args.num_nodes = data.shape[1]
            adj_mx = np.zeros((args.num_nodes, args.num_nodes))

    test_batch_size = args.batch_size
    if args.mode in ['train', 'inherit']:
        save_dir = os.path.join('../results/test', args.dataset, args.ap)
    if args.mode == 'clean_seeds':
        save_dir = os.path.join('../seeds', args.dataset, 'clean')
    if args.mode == 'noisy_seeds':
        save_dir = os.path.join('../seeds', args.dataset, 'noisy')
    if args.mode == 'iteratively':
        save_dir = os.path.join('../seeds', args.dataset, f'iteratively_{args.threshold}')
    if args.mode == 'manual':
        save_dir = os.path.join('../seeds', args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False

    dataloader, geo_mask, sem_mask = generate_data(data_dir, args.seq_len, args.seq_len, args.in_dim, args.datatype,
                                                   args.batch_size,
                                                   test_batch_size,
                                                   adj_mx)
    scaler = dataloader['scaler']

    return dataloader, adj_mx, scaler, save_dir, geo_mask, sem_mask

def load_PEMSD7_adj(adj_path):
    adj = sp.load_npz(os.path.join(adj_path))
    adj = adj.todense()

    return adj

######################################################################
# dataset processing
######################################################################
class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        :param shuffle:
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data_len[-int(data_len * (val_ratio + test_ratio)): -int(data_len * test_ratio)]
    train_data = data[: -int(data_len * (val_ratio + test_ratio))]

    return train_data, val_data, test_data


def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    """
    data format for seq2seq task or seq to single value task.
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :param single:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window + horizon - 1: index + window + horizon])
            index += 1
    else:
        while index < end_index:
            X.append(data[index: index + window])
            Y.append(data[index + window: index + window + horizon])
            index += 1
    X = np.array(X).astype('float32')
    Y = np.array(Y).astype('float32')

    return X, Y


def load_dataset(data_dir, batch_size, test_batch_size=None,mode='mix'):
    """
    generate dataset
    :param data_dir:
    :param batch_size:
    :param test_batch_size:
    :param kwargs:
    :return:
    """
    data = {}
    if 'pollution' not in data_dir and 'weather' not in data_dir:  # 数据集已分割
        for category in ['train', 'val', 'test']:
            cat_data = np.load(Path().joinpath(data_dir, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']

        scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:  # norm?
            data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

        train_len = len(data['x_train'])
        permutation = np.random.permutation(train_len)
        data['x_train_1'] = data['x_train'][permutation][:int(train_len / 2)]
        data['y_train_1'] = data['y_train'][permutation][:int(train_len / 2)]
        data['x_train_2'] = data['x_train'][permutation][int(train_len / 2):]
        data['y_train_2'] = data['y_train'][permutation][int(train_len / 2):]
        data['x_train_3'] = copy.deepcopy(data['x_train_2'])
        data['y_train_3'] = copy.deepcopy(data['y_train_2'])
        data['train_loader_1'] = DataLoader(data['x_train_1'], data['y_train_1'], batch_size)
        data['train_loader_2'] = DataLoader(data['x_train_2'], data['y_train_2'], batch_size)
        data['train_loader_3'] = DataLoader(data['x_train_3'], data['y_train_3'], batch_size)

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scalar

        return data
    else:
        dataset = np.load(data_dir, allow_pickle=True)
        data_train, data_val, data_test = split_data_by_ratio(dataset, 0.1, 0.2)
        if mode == 'channel':
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(data_train)
            data_train = scaler.transform(data_train)
            data_val = scaler.transform(data_val)
            data_test = scaler.transform(data_test)

        x_tr, y_tr = Add_Window_Horizon(data_train, 12, 12, False)
        x_tr_orig = x_tr.copy()
        x_val, y_val = Add_Window_Horizon(data_val, 12, 12, False)
        x_test, y_test = Add_Window_Horizon(data_test, 12, 12, False)
        data['x_train'] = x_tr
        data['y_train'] = y_tr
        data['x_val'] = x_val
        data['y_val'] = y_val
        data['x_test'] = x_test
        data['y_test'] = y_test

        real_scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        # Data format
        for category in ['train', 'val', 'test']:
            for i in range(x_tr.shape[-1]):
                scaler = StandardScaler(mean=x_tr_orig[..., i].mean(), std=x_tr_orig[..., i].std())
                data['x_' + category][..., i] = scaler.transform(data['x_' + category][..., i])
            print('x_' + category, data['x_' + category].shape)

        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = real_scaler

        return data


def load_adj(pkl_filename):
    """
    为什么gw的邻接矩阵要做对称归一化，而dcrnn的不做？其实做了，在不同的地方，是为了执行双向随机游走算法。
    所以K-order GCN需要什么样的邻接矩阵？
    这个应该参考ASTGCN，原始邻接矩阵呢？参考dcrnn
    为什么ASTGCN不采用对称归一化的拉普拉斯矩阵？
    :param pkl_filename: adj_mx.pkl
    :return:
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

    return sensor_ids, sensor_id_to_ind, adj_mx
    # return sensor_ids, sensor_id_to_ind, adj_mx.astype('float32')


def load_pickle(pkl_filename):
    try:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with Path(pkl_filename).open('rb') as f:
            pkl_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pkl_filename, ':', e)
        raise

    return pkl_data


######################################################################
# generating chebyshev polynomials
######################################################################
def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


######################################################################
# generating diffusion convolution adj
######################################################################
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


######################################################################
# metrics
######################################################################
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


######################################################################
# Exponential annealing for softmax temperature
######################################################################
class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.05, last_epoch=-1):
        self.total_epochs = total_epochs
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        self.curr_temp = max(self.base_temp * 0.90 ** self.last_epoch, self.temp_min)

        return self.curr_temp


def generate_data(graph_signal_matrix_name, train_len, pred_len, in_dim, type, batch_size, test_batch_size=None,
                  adj_mx=None, transformer=None):
    """shape=[num_of_samples, 12, num_of_vertices, 1]"""
    data, geo_mask, sem_mask = data_preprocess(graph_signal_matrix_name, train_len, pred_len, in_dim, adj_mx, type)

    scalar = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scalar.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scalar

    return data, geo_mask, sem_mask


def generate_from_train_val_test(origin_data, train_len, pred_len, in_dim, transformer=None):
    data = {}
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(origin_data[key], train_len, pred_len, in_dim)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')


    return data


def generate_from_data(origin_data, length, train_len, pred_len, in_dim, transformer=None):
    """origin_data shape: [17856, 170, 3]"""
    data = generate_sample(origin_data, train_len, pred_len, in_dim)
    train_line, val_line = int(length * 0.6), int(length * 0.8)

    for key, line1, line2 in (('train', 0, train_line),
                              ('val', train_line, val_line),
                              ('test', val_line, length)):
        x, y = generate_seq(origin_data[line1: line2], train_len, pred_len, in_dim)
        print(x.shape)
        data['x_' + key] = x.astype('float32')
        data['y_' + key] = y.astype('float32')
        # if transformer:  # 啥意思？
        #     x = transformer(x)
        #     y = transformer(y)

    return data


def generate_sample(origin_data, train_len, pred_len, in_dim):
    data = {}
    data['origin'] = origin_data
    x, y = generate_seq(origin_data, train_len, pred_len, in_dim)
    data['x'] = x.astype('float32')
    data['y'] = y.astype('float32')
    return data


def generate_seq(data, train_length, pred_length, in_dim):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: in_dim]

    if train_length == pred_length:
        return np.split(seq, 2, axis=1)
    else:
        return np.split(seq, [train_length], axis=1)


def sample_split(data, train_length, in_dim, overlap=0):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length], 0)
        for i in range(0, data.shape[0] - train_length + 1, train_length - overlap)],
        axis=0)[:, :, :, 0: in_dim]

    return seq


def dim_uniform(origin_data):
    if origin_data.ndim == 1:
        data = origin_data.reshape((origin_data.shape[0], 1, 1))
    elif origin_data.ndim == 2:
        data = origin_data.reshape((origin_data.shape[0], origin_data.shape[1], 1))
    else:
        data = origin_data

    return data


def data_preprocess(data_path, train_len, pred_len, in_dim, adj_mx, type='csv', transformer=None):
    if type == 'csv':
        origin_data = pd.read_csv(data_path)

        origin_data = origin_data.iloc[1:, 1:]

        origin_data = np.array(origin_data)
        # origin_data = np.expand_dims(origin_data, -1)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)

    elif type == 'txt':
        origin_data = np.loadtxt(data_path, delimiter=',')

        origin_data = np.array(origin_data)
        origin_data = dim_uniform(origin_data)
        # origin_data = np.expand_dims(origin_data, -1)
        length = len(origin_data)
    elif type == 'tsf':
        origin_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            data_path)
        origin_data = origin_data.iloc[:, 1:]
        # origin_data = origin_data.T

        origin_data = np.array(origin_data.values)

        data = [[origin_data[i][0][j] for i in range(origin_data.size)] for j in range(origin_data[0][0].size)]
        origin_data = np.array(data)

        # origin_data = np.expand_dims(origin_data, -1)
        origin_data = dim_uniform(origin_data)
        length = len(origin_data)
    elif type == 'npz':
        origin_data = np.load(data_path)  # shape=[17856, 170, 3]
        keys = origin_data.keys()
        if 'train' in keys and 'val' in keys and 'test' in keys:
            data = generate_from_train_val_test(dim_uniform(origin_data['data']), train_len, pred_len, in_dim,
                                                transformer)
            return data

        elif 'data' in keys:
            length = origin_data['data'].shape[0]
            origin_data = origin_data['data']

        else:
            raise KeyError("neither data nor train, val, test is in the data")
    elif type == 'h5':
        origin_data = pd.read_hdf(data_path)
        origin_data = np.array(origin_data)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)
    elif type in ['npy', 'subsets']:
        origin_data = np.load(data_path)
        origin_data = dim_uniform(origin_data)

        length = len(origin_data)

    num_samples, num_nodes, feature_dim = origin_data.shape
    data_list = [origin_data]

    dynafile = pd.read_csv('data/pems/PEMS04/PeMS04.dyna')
    timesolts = list(dynafile['time'][:int(dynafile.shape[0] / num_nodes)])
    if not dynafile['time'].isna().any():
        timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), timesolts))
        timesolts = np.array(timesolts, dtype='datetime64[ns]')

    # time in day
    time_ind = (timesolts - timesolts.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    data_list.append(time_in_day)

    # day in week
    dayofweek = []
    for day in timesolts.astype("datetime64[D]"):

        dayofweek.append(datetime.datetime.strptime(str(day), '%Y-%m-%d').weekday())
    day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))  # (17856, 170, 7)
    day_in_week[np.arange(num_samples), :, dayofweek] = 1
    data_list.append(day_in_week)

    # generate DTW matrix
    time_intervals = 300  # 5 mins
    points_per_hour = 3600 // time_intervals
    # print('%' * 100)
    # print(origin_data.shape)
    # print(origin_data)
    data_mean = np.mean(
        [origin_data[24 * points_per_hour * i: 24 * points_per_hour * (i + 1)]
         for i in range(origin_data.shape[0] // (24 * points_per_hour))], axis=0)
    # print(data_mean.shape)
    # print('=' * 100)
    # print(data_mean)
    dtw_distance = np.zeros((num_nodes, num_nodes))
    for i in tqdm(range(num_nodes)):
        for j in range(i, num_nodes):
            dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
    for i in range(num_nodes):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]
    cache_path = "../data/pems/dtw_PeMS08.npy"
    np.save(cache_path, dtw_distance)
    dtw_matrix = np.load(cache_path)

    # generate geo and sem masks
    sh_mx = adj_mx.copy()
    sh_mx[sh_mx > 0] = 1
    sh_mx[sh_mx == 0] = 511
    for i in range(num_nodes):
        sh_mx[i, i] = 0
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)

    sh_mx = sh_mx.T  # (170, 170) 为什么转置啊？
    far_mask_delta = 7  # mask稀疏程度
    dtw_delta = 5  # mask稀疏程度
    geo_mask = torch.zeros(num_nodes, num_nodes)
    indices = torch.tensor(sh_mx >= far_mask_delta)
    geo_mask[indices] = 1
    geo_mask = geo_mask.bool()
    sem_mask = torch.ones(num_nodes, num_nodes)
    sem_mask_sort = dtw_matrix.argsort(axis=1)[:, :dtw_delta]
    for i in range(sem_mask.shape[0]):
        sem_mask[i][sem_mask_sort[i]] = 0
    sem_mask = sem_mask.bool()
    sem_mask = None
    data = generate_from_data(origin_data, length, train_len, pred_len, in_dim)

    return data, geo_mask, sem_mask


def convert_tsf_to_dataframe(
        full_file_path_and_name,
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                    len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                    len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                                numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)
        loaded_data = loaded_data.iloc[:, 1:3]
        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(r"D:\大学学习\下载的东东\Edge浏览器下载\nn5_weekly_dataset.tsf")
# print(loaded_data)
def get_adj_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':  # 啥意思啊，表里有的就置1？
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be connectivity or distance!")

    return A


def calculate_statistical(data, seq_len):
    N = data.shape[1]  # numnodes
    T = data.shape[0]  # length
    D = data.shape[2]

    Mean = np.mean(data, axis=0)
    Skewness = skew(data, axis=0)
    Kurtosis = kurtosis(data, axis=0)
    Variance = np.var(data, axis=0)
    Slope = np.zeros((N, D))
    StandardDeviation = np.std(data, axis=0)

    x = np.arange(T)
    for i in range(N):
        for j in range(D):
            Slope[i, j] = np.polyfit(x, data[:, i, j], 1)[0]

    entropy_data = softmax(data, axis=0)
    Entropy = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            Entropy[i, j] = entropy(entropy_data[:, i, j], base=2)

    Ema = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            Ema[i, j] = pd.Series(data[:, i, j]).ewm(span=5, adjust=False).mean().values[-1]

    Statistics = np.stack([Mean, Skewness, Kurtosis, Variance, Slope, StandardDeviation, Entropy, Ema])

    Statistics = Statistics.mean(axis=1)
    Statistics = Statistics.mean(axis=1)
    addition_feature = np.array([seq_len, T, N, D])
    Statistics = np.append(Statistics, addition_feature, axis=0)
    return Statistics


######################################################################
# MLP for spatial attention
######################################################################
class MLP(nn.Module):
    def __init__(self, hiddens, input_size, activation_function, out_act, dropout_ratio=0.):

        super(MLP, self).__init__()
        # dropout_ratio = 0.2
        # layers = [nn.Dropout(dropout_ratio)]
        layers = []

        previous_h = input_size
        for i, h in enumerate(hiddens):

            activation = None if i == len(hiddens) - 1 and not out_act else activation_function
            layers.append(nn.Linear(previous_h, h))

            if activation is not None:
                layers.append(activation)

            # layers.append(nn.Dropout(dropout_ratio))
            previous_h = h
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


######################################################################
# Darts utils
######################################################################
if __name__ == '__main__':
    adj = get_adj_matrix('data/pems/PEMS08/PEMS08.csv', 170)
    print(adj)
    # dataloader = load_dataset('data/METR-LA', 64, 64)
    # train_iterator = dataloader['train_loader_1'].get_iterator()
    # val_iterator = dataloader['train_loader_2'].get_iterator()
    # train_val = dataloader['train_loader'].get_iterator()
    # print(len(list(train_iterator)))


######################################################################
# AHC dataset processing
######################################################################


class AHC_DataLoader(object):
    def __init__(self, arch_pairs, task_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        task_name, x0, x1, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x0) % batch_size)) % batch_size
            task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)
            x0_padding = np.repeat(x0[-1:], num_padding, axis=0)
            x1_padding = np.repeat(x1[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            task_name = np.concatenate([task_name, task_name_padding], axis=0)
            x0 = np.concatenate([x0, x0_padding], axis=0)
            x1 = np.concatenate([x1, x1_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
        self.size = len(x0)
        self.num_batch = int(self.size // self.batch_size)
        self.task_name = task_name
        self.x0 = x0
        self.x1 = x1
        self.y = y
        self.task_dict = task_dict

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        task_name, x0, x1, y = self.task_name[permutation], self.x0[permutation], self.x1[permutation], self.y[
            permutation]
        self.task_name = task_name
        self.x0 = x0
        self.x1 = x1
        self.y = y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                task_i = [self.task_dict[i] for i in self.task_name[start_ind: end_ind, ...]]
                x0_i = self.x0[start_ind: end_ind, ...]  # ...代替多个:
                x1_i = self.x1[start_ind: end_ind, ...]
                y_i = self.y[start_ind: end_ind, ...]
                yield task_i, x0_i, x1_i, y_i
                self.current_ind += 1

        return _wrapper()


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
    # 计算有重复数组的无重复全排列
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
    for i in range(len(PRIMITIVES)):
        op_list += [i] * comb[i]
    all_archs = permuteUnique(op_list)

    topos = get_all_topos(4)
    base_genos = get_base_genos(topos)

    all_genos = []
    for arch in all_archs:
        for base_geno in base_genos:
            geno = []
            for i in range(4):
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


######################################################################
# nac dataset processing
######################################################################

class AP_DataLoader(object):
    def __init__(self, arch_pairs, task_dict, statistics_dict, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        self.task_dict = task_dict
        self.statistics_dict = statistics_dict
        task_name, x, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x) % batch_size)) % batch_size
            x_padding = np.repeat(x[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            task_name_padding = np.repeat(task_name[-1:], num_padding, axis=0)

            x = np.concatenate([x, x_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
            task_name = np.concatenate([task_name, task_name_padding], axis=0)

        self.size = len(x)
        self.num_batch = int(self.size // self.batch_size)
        self.x = x
        self.y = y
        self.task_name = task_name

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x, y, task_name = self.x[permutation], self.y[permutation], self.task_name[permutation]
        self.x = x
        self.y = y
        self.task_name = task_name

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.y[start_ind: end_ind, ...]
                task_i = [self.task_dict[i] for i in self.task_name[start_ind: end_ind, ...]]
                statistics_i = [np.squeeze(self.statistics_dict[i])[0:256] for i in
                                self.task_name[start_ind: end_ind, ...]]
                yield x_i, task_i, statistics_i, y_i
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return self.num_batch
class NAC_DataLoader(object):
    def __init__(self, arch_pairs, batch_size, pad_with_last_sample=True):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        x, y = zip(*arch_pairs)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x) % batch_size)) % batch_size
            x_padding = np.repeat(x[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)
            x = np.concatenate([x, x_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)
        self.size = len(x)
        self.num_batch = int(self.size // self.batch_size)
        self.x = x
        self.y = y

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x, y = self.x[permutation], self.y[permutation]
        self.x = x
        self.y = y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind: end_ind, ...]  # ...代替多个:
                y_i = self.y[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return self.num_batch


class NAC_Dataloader_for_PINAT(object):
    def __init__(self, arch_pairs, batch_size, pad_with_last_sample=True, live_generation=True, save_file=False):
        """
        generate data batches
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0  # index
        x, y = zip(*arch_pairs)
        if live_generation:
            adjacency = []
            features = []
            edge_indices = []
            edge_num = []
            lapla = []
            nor_lapla = []
            num_vertices = []
            operations = []
            for xi in x:
                adj, ops = geno_to_adj(xi)
                adjacency.append(adj)
                features.append(ops)
                operations.append(generate_operations(ops))
                num_vertex = len(ops)
                num_vertices.append(num_vertex)
                edge_index = generate_edge_index(adj)
                edge_num.append(edge_index.shape[1])
                if edge_index.shape[1] < 13:
                    edge_index = np.pad(edge_index, ((0, 0), (0, 13 - edge_index.shape[1])), 'constant')
                edge_indices.append(edge_index)
                la = generate_lapla_matrix(adj)
                lapla.append(la)

        else:
            with h5py.File(f'/home/AutoCTS++/seeds/arch_data_of_optimal_subspace.h5', 'r') as hr:
                dataset_adjacency = hr['adjacency']
                dataset_features = hr['features']
                dataset_lapla = hr['lapla']
                # dataset_nor_lapla = hr['nor_lapla']
                dataset_operations = hr['operations']
                dataset_edge_num = hr['edge_num']
                dataset_edge_indices = hr['edge_indices']

                adjacency = dataset_adjacency[:]
                features = dataset_features[:]
                lapla = dataset_lapla[:]
                # nor_lapla = dataset_nor_lapla[:]
                operations = dataset_operations[:]
                num_vertices = [9] * len(operations)

                edge_indices = list(dataset_edge_indices[:])
                edge_num = list(dataset_edge_num[:].squeeze())


        if save_file:
            with h5py.File('/home/AutoCTS++/seeds/arch_data_x.h5', 'a') as hf:
                add = True
                if 'adjacency' not in hf:
                    dataset_adjacency = hf.create_dataset('adjacency', data=adjacency, shape=(len(adjacency), 9, 9),
                                                          maxshape=(None, 9, 9),
                                                          dtype='f')
                    add = False
                else:
                    dataset_adjacency = hf['adjacency']
                if 'features' not in hf:
                    dataset_features = hf.create_dataset('features', data=features, shape=(len(features), 9),
                                                         maxshape=(None, 9),
                                                         dtype='f')
                    add = False
                else:
                    dataset_features = hf['features']

                if 'lapla' not in hf:
                    dataset_lapla = hf.create_dataset('lapla', data=lapla, shape=(len(lapla), 9, 9),
                                                      maxshape=(None, 9, 9),
                                                      dtype='f')
                    add = False
                else:
                    dataset_lapla = hf['lapla']

                if 'edge_indices' not in hf:
                    dataset_edge_indices = hf.create_dataset('edge_indices', data=edge_indices,
                                                             shape=(len(edge_indices), 2, 13),
                                                             maxshape=(None, 2, 13),
                                                             dtype='i')
                    add = False
                else:
                    dataset_edge_indices = hf['edge_indices']

                if 'edge_num' not in hf:
                    dataset_edge_num = hf.create_dataset('edge_num', data=edge_num, shape=(len(edge_num), 1),
                                                         maxshape=(None, 1), dtype='i')
                    add = False
                else:
                    dataset_edge_num = hf['edge_num']

                if 'operations' not in hf:
                    dataset_operations = hf.create_dataset('operations', data=operations,
                                                           shape=(len(operations), 9, 17),
                                                           maxshape=(None, 9, 17),
                                                           dtype='f')
                    add = False
                else:
                    dataset_operations = hf['operations']

                if add:

                    current_length = len(dataset_lapla)
                    new_length = current_length + len(lapla)

                    for dataset, new_data in [(dataset_edge_num, edge_num), (dataset_edge_indices, edge_indices),
                                              (dataset_lapla, lapla),
                                              (dataset_operations, operations),
                                              (dataset_adjacency, adjacency),
                                              (dataset_features, features)]:
                        dataset.resize((new_length, *dataset.shape[1:]))

                        dataset[current_length:new_length] = new_data

        if pad_with_last_sample:
            num_padding = (batch_size - (len(x) % batch_size)) % batch_size
            adjacency_padding = np.repeat(adjacency[-1:], num_padding, axis=0)
            features_padding = np.repeat(features[-1:], num_padding, axis=0)
            edge_indices_padding = [edge_indices[-1]] * num_padding
            edge_num_padding = np.repeat(edge_num[-1:], num_padding, axis=0)
            lapla_padding = np.repeat(lapla[-1:], num_padding, axis=0)
            # lapla_nor_padding = np.repeat(nor_lapla[-1:], num_padding, axis=0)
            num_vertices_padding = np.repeat(num_vertices[-1:], num_padding, axis=0)
            operations_padding = np.repeat(operations[-1:], num_padding, axis=0)
            y_padding = np.repeat(y[-1:], num_padding, axis=0)

            adjacency = np.concatenate([adjacency, adjacency_padding], axis=0)
            features = np.concatenate([features, features_padding], axis=0)
            edge_indices = edge_indices + edge_indices_padding
            edge_num = np.concatenate([edge_num, edge_num_padding], axis=0)
            lapla = np.concatenate([lapla, lapla_padding], axis=0)
            # nor_lapla = np.concatenate([nor_lapla, lapla_nor_padding], axis=0, dtype=np.float32)
            num_vertices = np.concatenate([num_vertices, num_vertices_padding], axis=0)
            operations = np.concatenate([operations, operations_padding], axis=0, dtype=np.float32)
            y = np.concatenate([y, y_padding], axis=0)

        self.size = len(features)
        self.num_batch = int(self.size // self.batch_size)
        self.adjacency = adjacency
        self.features = features
        self.edge_indices = edge_indices
        self.edge_num = edge_num
        self.lapla = lapla
        # self.nor_lapla = nor_lapla
        self.num_vertices = num_vertices
        self.operations = operations
        self.y = y

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        adjacency = self.adjacency[permutation]
        features = self.features[permutation]
        edge_indices = [self.edge_indices[i] for i in permutation]
        edge_num = self.edge_num[permutation]
        lapla = self.lapla[permutation]
        # nor_lapla = self.nor_lapla[permutation]
        num_vertices = self.num_vertices[permutation]
        operations = self.operations[permutation]
        y = self.y[permutation]

        self.adjacency = adjacency
        self.features = features
        self.edge_indices = edge_indices
        self.edge_num = edge_num
        self.lapla = lapla
        # self.nor_lapla = nor_lapla
        self.num_vertices = num_vertices
        self.operations = operations
        self.y = y

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                inputs = {}
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                inputs['adjacency'] = self.adjacency[start_ind: end_ind, ...]
                inputs['features'] = torch.tensor(self.features[start_ind: end_ind, ...], dtype=torch.long)
                inputs['edge_index_list'] = self.edge_indices[start_ind:end_ind]
                inputs['edge_num'] = self.edge_num[start_ind: end_ind, ...]
                inputs['lapla'] = self.lapla[start_ind: end_ind, ...]
                # inputs['nor_lapla'] = self.nor_lapla[start_ind: end_ind, ...]
                inputs['num_vertices'] = self.num_vertices[start_ind: end_ind, ...]
                inputs['operations'] = self.operations[start_ind: end_ind, ...]
                inputs['val_acc'] = self.y[start_ind: end_ind, ...]
                yield inputs
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return self.num_batch


def geno_to_adj(arch):

    node_num = len(arch) + 2
    adj = np.zeros((node_num, node_num))
    ops = [len(PRIMITIVES)]
    for i in range(len(arch)):
        connect, op = arch[i]
        ops.append(arch[i][1])
        if connect == 0 or connect == 1:
            adj[connect][i + 1] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 1] = 1
            adj[(connect - 2) * 2 + 3][i + 1] = 1
    adj[-3][-1] = 1
    adj[-2][-1] = 1  # output
    ops.append(len(PRIMITIVES) + 1)

    return adj, ops


def generate_operations(feature):
    node_num = len(feature)
    op_num = len(PRIMITIVES) + 2
    operations = np.zeros((node_num, op_num))
    for id in range(node_num):
        idx = feature[id]
        operations[id][idx] = 1

    return operations


def generate_edge_index(adjacency):
    edge_list = []
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i][j] == 1:
                edge_list.append([i, j])
    node1, node2 = zip(*edge_list)
    node1 = list(node1)
    node2 = list(node2)
    edge_index = np.array([node1, node2])

    return edge_index


def generate_lapla_matrix(adjacency):
    # un-normalized lapla_matrix
    adj_matrix = adjacency
    degree = np.diag(np.sum(adj_matrix, axis=1))
    unnormalized_lapla = degree - adj_matrix

    return np.array(unnormalized_lapla)


def generate_normalized_lapla(adjacency):
    # normalized lapla_matrix
    adj_matrix = adjacency
    degree = torch.tensor(np.sum(adj_matrix, axis=1))
    degree = sp.diags(dgl.backend.asnumpy(degree).clip(1) ** -0.5, dtype=float)
    normalized_lapla = np.array(sp.eye(adj_matrix.shape[0]) - degree * adj_matrix * degree)
    return np.array(normalized_lapla)


# def laplacian_positional_encoding(adj, pos_enc_dim, number_nodes=7):
#     """
#         Graph positional encoding v/ Laplacian eigenvectors
#     """
#
#     # Laplacian
#     lap_pos_enc = []
#     num = 0
#     for i in range(len(adj)):
#         adj_matrix = adj[i] + adj[i].T
#         degree = torch.tensor(np.sum(adj_matrix, axis=1))
#         degree = sp.diags(dgl.backend.asnumpy(degree).clip(1) ** -0.5, dtype=float)
#         L = sp.eye(number_nodes) - degree * adj_matrix * degree
#         try:
#             EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
#         except:
#             num += 1
#             print('')
#             print(adj_matrix)
#             print(i, num)
#         EigVec = EigVec[:, EigVal.argsort()]  # increasing order
#         # lap_pos_enc.append(torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1].real).float())
#         lap_pos_enc.append(EigVec[:, 1:pos_enc_dim + 1].real)
#
#     #     if (i+1) % 10000 == 0:
#     #         print(i)
#     # np.save('pos_enc.npy', np.array(lap_pos_enc))
#     # import sys
#     # sys.exit()
#
#     # Laplacian
#     # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
#     # N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
#     # L = sp.eye(g.number_of_nodes()) - N * A * N
#     # # Eigenvectors with scipy
#     # #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
#     # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
#     # EigVec = EigVec[:, EigVal.argsort()] # increasing order
#     # g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
#     # return g
#     return lap_pos_enc


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(obj, device):
    if torch.is_tensor(obj):
        # return obj.cuda()
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_cuda(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t, device) for t in obj]
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    if isinstance(obj, dict):
        return {k: to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


class standard_normalization:
    def __init__(self):
        pass

    def fit(self, data):
        # 计算每个任务的均值和标准差
        archs, accs = zip(*data)
        mean = np.mean(accs, axis=0)
        std = np.std(accs, axis=0)
        self.mean = mean
        self.std = std

    def transform(self, data):
        # 对每个任务的输出值进行归一化
        archs, accs = zip(*data)
        normalized_accs = (accs - self.mean) / self.std
        return list(zip(archs, normalized_accs))

    def reverse_transform(self, data):
        # 对每个任务的输出值进行反归一化
        archs, accs = zip(*data)
        archs = np.array(archs)
        accs = np.array(accs)
        reverse_normalized_accs = (accs * self.std) + self.mean
        return list(zip(archs, reverse_normalized_accs))


class SPLLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss(reduction='none'), threshold=0.1, growing_factor=1.01):
        super(SPLLoss, self).__init__()
        self.threshold = threshold
        self.growing_factor = growing_factor
        self.criterion = criterion

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        super_loss = self.criterion(input, target)
        v = self.spl_loss(super_loss)

        return (super_loss * v).mean()

    def increase_threshold(self):
        self.threshold *= self.growing_factor

    def spl_loss(self, super_loss):
        v = super_loss < self.threshold
        return v.int()


class DynamicWeightedLoss(nn.Module):
    def __init__(self, scaler, criterion=nn.L1Loss(reduction='none'), reduction='mean', gap=20.5, order=8):
        super(DynamicWeightedLoss, self).__init__()
        self.criterion = criterion
        self.scaler = scaler
        self.reduction = reduction
        self.gap = gap
        self.order = order

    def forward(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        difficulties = self.calculate_difficulites(targets)

        weight = self.calculate_weight(difficulties)

        if self.reduction == 'mean':
            return (loss * weight).mean()
        else:
            return loss * weight

    def calculate_difficulites(self, targets):
        return targets * self.scaler.std + self.scaler.mean

    def calculate_weight(self, difficulities):
        '''
        :formula (x-21) ^2 + 1   (x-30)^10 / 9^10
        :param difficulities:
        :return: calculated weights
        '''
        # gap is the time weight=1
        # v1 = difficulities <= gap
        # v2 = difficulities > gap
        #
        # f1 = 0.4 * (difficulities - gap) ** 2 + 1
        # f2 = (self.sup - difficulities) ** 9 / (self.sup - gap) ** 9
        #
        # f1 = 10000/difficulities**3
        # f2 = (self.sup - difficulities) ** 9 / (self.sup - gap) ** 9

        # f1 = 0.2 * (difficulities - gap) ** 2 + 1
        # f2 = (self.sup - difficulities) ** 9 / (self.sup - gap) ** 9
        # f = (gap ** 8) / difficulities ** 8

        # f = gap ** 15 / difficulities ** 15

        # f = gap ** 12 / difficulities ** 12

        f = self.gap ** self.order / difficulities ** self.order

        return f
