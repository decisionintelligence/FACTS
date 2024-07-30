import numpy as np
import torch
import torch.nn as nn
from .gcn_net import GCN
from ..genotypes import PRIMITIVES
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from .set_encoder.setenc_models import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def train_baseline_epoch(train_loader, valid_loader, ap, train_criterion, valid_criterion, curriculum, optimizer):
#     train_loss = []
#     ap.train()
#     if curriculum != 'SPL':
#         train_loader.shuffle()
#
#     train_result = []
#     train_actual = []
#     for i, (arch, acc) in enumerate(train_loader.get_iterator()):  # 对每个batch
#         acc = torch.Tensor(acc).to(DEVICE)
#         outputs = ap(arch)
#         loss = train_criterion(outputs, acc)
#
#         train_loss.append(loss.item())
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_result += list(zip(arch, outputs.detach().cpu().tolist()))
#         train_actual += list(zip(arch, acc.detach().cpu().tolist()))
#
#     if curriculum == 'SPL':
#         train_criterion.increase_threshold()
#
#     valid_result = []
#     valid_actual = []
#     # eval
#     with torch.no_grad():
#         ap = ap.eval()
#         valid_loss = []
#         accs = []
#         preds = []
#         for i, (arch, acc) in enumerate(valid_loader.get_iterator()):
#             accs += list(acc)
#             acc = torch.Tensor(acc).to(DEVICE)
#             outputs = ap(arch)
#             loss = valid_criterion(outputs, acc)
#             valid_loss.append(loss.item())
#             preds += list(outputs.detach().cpu().numpy())
#
#             valid_result += list(zip(arch, outputs.detach().cpu().tolist()))
#             valid_actual += list(zip(arch, acc.detach().cpu().tolist()))
#
#         # calculate spearmanr
#         rho = spearmanr(accs, preds)
#         ken = kendalltau(accs, np.round(np.array(preds), decimals=2))
#         # calculate acc ratio
#         num_correct = 0
#         num = 0
#         for i in range(len(accs)):
#             for j in range(i + 1, len(preds)):
#                 num += 1
#                 if (accs[i] - accs[j]) * (preds[i] - preds[j]) > 0:
#                     num_correct += 1
#
#         acc_ratio = num_correct / num
#
#     addition_info = dict(train_result=train_result, train_actual=train_actual, valid_result=valid_result,
#                          valid_actual=valid_actual)
#     return np.mean(train_loss), np.mean(valid_loss), rho, ken, acc_ratio, addition_info
#
#
# def evaluate(test_loader, ap, criterion):
#     with torch.no_grad():
#         ap = ap.eval()
#         test_loss = []
#         accs = []
#         preds = []
#         test_result = []
#         test_actual = []
#         for i, (arch, acc) in enumerate(test_loader.get_iterator()):
#             accs += list(acc)
#             acc = torch.Tensor(acc).to(DEVICE)
#             outputs = ap(arch)
#             loss = criterion(outputs, acc)
#             test_loss.append(loss.item())
#             preds += list(outputs.detach().cpu().numpy())
#
#             test_result += list(zip(arch, outputs.detach().cpu().tolist()))
#             test_actual += list(zip(arch, acc.detach().cpu().tolist()))
#
#         addition_info = dict(test_result=test_result, test_actual=test_actual)
#         # calculate spearmanr
#         rho = spearmanr(accs, preds)
#
#         # 后续考虑计算过程中对于preds做四舍五入，保留一位或者两位小数点
#         ken = kendalltau(accs, np.round(np.array(preds), decimals=2))
#         # calculate acc ratio
#         num_correct = 0
#         num = 0
#         for i in range(len(accs)):
#             for j in range(i + 1, len(preds)):
#                 num += 1
#                 if (accs[i] - accs[j]) * (preds[i] - preds[j]) > 0:
#                     num_correct += 1
#
#         acc_ratio = num_correct / num
#
#     return np.mean(test_loss), rho, ken, acc_ratio, addition_info
#
#
# def geno_to_adj(arch):
#     # arch.shape = [7, 2]
#     # 输出邻接矩阵，和节点特征
#     # 这里的邻接矩阵对应op为顶点的DAG，和Darts相反
#     # GCN处理无向图，这里DAG是有向图，所以需要改改？？？参考Wei Wen的文章
#     node_num = len(arch) + 2  # 加上一个input和一个output节点
#     adj = np.zeros((node_num, node_num))
#     ops = [len(PRIMITIVES)]
#     for i in range(len(arch)):
#         connect, op = arch[i]
#         ops.append(arch[i][1])
#         if connect == 0 or connect == 1:
#             adj[connect][i + 1] = 1
#         else:
#             adj[(connect - 2) * 2 + 2][i + 1] = 1
#             adj[(connect - 2) * 2 + 3][i + 1] = 1
#     adj[-3][-1] = 1
#     adj[-2][-1] = 1  # output
#     ops.append(len(PRIMITIVES) + 1)
#
#     return adj, ops
#
#
# class ArchPredictor(nn.Module):
#
#     def __init__(self, n_nodes, n_ops, n_layers=2, ratio=1, embedding_dim=128):
#         # 后面要参考下Wei Wen文章的GCN实现
#         super(ArchPredictor, self).__init__()
#         self.n_nodes = n_nodes
#         self.n_ops = n_ops
#
#         # +2用于表示input和output node
#         self.embedding = nn.Embedding(self.n_ops + 2, embedding_dim=embedding_dim)
#         self.gcn = GCN(n_layers=n_layers, in_features=embedding_dim,
#                        hidden=embedding_dim, num_classes=embedding_dim)
#
#         self.fc = nn.Linear(embedding_dim * ratio, 1, bias=True)  # f_out=1  ratio是啥意思？
#
#     def forward(self, arch):
#         # 先将数组编码改成邻接矩阵编码
#         # arch0.shape = [batch_size, 7, 2]
#
#         b_adj0, b_ops0, = [], []
#         # print(f'batch size:{len(arch0)}')
#         for i in range(len(arch)):
#             adj0, ops0 = geno_to_adj(arch[i])
#             b_adj0.append(adj0)
#             b_ops0.append(ops0)
#
#         b_adj0 = torch.Tensor(b_adj0).to(DEVICE)
#         b_ops0 = torch.LongTensor(b_ops0).to(DEVICE)
#
#         feature = self.extract_features((b_adj0, b_ops0))
#
#         score = self.fc(feature).view(-1)
#
#         return score
#
#     def extract_features(self, arch):
#         # 分别输入邻接矩阵和operation？
#         if len(arch) == 2:
#             matrix, op = arch
#             return self._extract(matrix, op)
#         else:
#             print('error')
#
#     def _extract(self, matrix, ops):
#
#         ops = self.embedding(ops)
#         feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)  # shape=[b, nodes, dim] pooling
#         return feature

def train_baseline_epoch(data_loader, valid_loader, ap, criterion, optimizer):
    train_loss = []
    ap.train()
    train_dataloader = data_loader

    train_dataloader.shuffle()

    for (arch, label) in train_dataloader.get_iterator():  # 对每个batch
        label = torch.Tensor(label).to(DEVICE)
        outputs = ap(arch)
        loss = criterion(outputs, label)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # eval
    with torch.no_grad():
        ap = ap.eval()
        valid_loss = []
        actual = []
        pred = []
        for i, (arch, label) in enumerate(valid_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)

            outputs = ap(arch)
            actual.extend(label.cpu().numpy())
            pred.extend(outputs.detach().cpu().numpy())
            loss = criterion(outputs, label)
            valid_loss.append(loss.item())

        num_correct = 0

        for i in range(0, len(actual) - 1):
            for j in range(i + 1, len(actual)):
                if (actual[i] - actual[j]) * (pred[i] - pred[j]) > 0:
                    num_correct += 1

        acc = num_correct / (len(actual) * (len(actual) - 1) / 2)
        rho = spearmanr(actual, pred)
    return np.mean(train_loss), np.mean(valid_loss), rho, acc


def evaluate(test_loader, ap, criterion):
    ap = ap.eval()
    with torch.no_grad():
        actual = []
        pred = []
        test_loss = []
        for i, (arch,label) in enumerate(test_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)

            outputs = ap(arch)
            actual.extend(label.cpu().numpy())
            pred.extend(outputs.detach().cpu().numpy())
            loss = criterion(outputs, label)
            test_loss.append(loss.item())

        num_correct = 0
        for i in range(0, len(actual) - 1):
            for j in range(i + 1, len(actual)):
                if (actual[i] - actual[j]) * (pred[i] - pred[j]) > 0:
                    num_correct += 1

        acc = num_correct / (len(actual) * (len(actual) - 1) / 2)
        rho = spearmanr(actual, pred)

    return np.mean(test_loss), rho, acc


def geno_to_adj(arch):
    # arch.shape = [7, 2]
    # 输出邻接矩阵，和节点特征
    # 这里的邻接矩阵对应op为顶点的DAG，和Darts相反
    # GCN处理无向图，这里DAG是有向图，所以需要改改？？？参考Wei Wen的文章
    node_num = len(arch) + 2  # 加上一个input和一个output节点
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


class ArchPredictor(nn.Module):
    def __init__(self, n_nodes, n_ops, n_layers=2, ratio=2, embedding_dim=128):
        # 后面要参考下Wei Wen文章的GCN实现
        super(ArchPredictor, self).__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops

        # +2用于表示input和output node
        self.embedding = nn.Embedding(self.n_ops + 2, embedding_dim=embedding_dim)
        self.gcn = GCN(n_layers=n_layers, in_features=embedding_dim,
                       hidden=embedding_dim, num_classes=embedding_dim)
        self.nz = 256
        self.fz = 128


        self.graph_fc = nn.Sequential(
            nn.Linear(embedding_dim, self.fz), nn.ReLU())


        self.pred_fc = nn.Sequential(nn.Linear(self.fz * ratio, self.fz, bias=True), nn.ReLU(),
                                     nn.Linear(self.fz, 1, bias=True))

    def forward(self, arch):

        # 先将数组编码改成邻接矩阵编码
        # arch.shape = [batch_size, 7, 2]
        b_adj, b_ops, features = [], [], []
        for i in range(len(arch)):
            adj, ops = geno_to_adj(arch[i])
            b_adj.append(adj)

            b_ops.append(ops)



        # extract the arch feature
        b_adj = np.array(b_adj)

        b_ops = np.array(b_ops)

        b_adj = torch.Tensor(b_adj).to(DEVICE)

        b_ops = torch.LongTensor(b_ops).to(DEVICE)

        feature = self.extract_features((b_adj, b_ops))

        # extract the task feature

        logits = self.pred_fc(feature).squeeze(1)

        return logits

    def set_encode(self, X):
        proto_batch = []
        for x in X:
            cls_protos = self.intra_setpool(x).squeeze(1)
            proto_batch.append(
                cls_protos)

        v = torch.stack(proto_batch)
        v = self.inter_setpool(v).squeeze(1)
        return v

    def extract_features(self, arch):
        # 分别输入邻接矩阵和operation？
        if len(arch) == 2:
            matrix, op = arch
            return self._extract(matrix, op)
        else:
            print('error')

    def _extract(self, matrix, ops):

        ops = self.embedding(ops)
        feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)  # shape=[b, nodes, dim] pooling

        return feature

