import torch

from .operations import *
from .genotypes import PRIMITIVES


class DataEmbedding(nn.Module):
    def __init__(self, feature_dim, embed_dim, lape_dim, add_time_in_day=True,
                 add_day_in_week=True, device=torch.device('cpu')):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = linear(feature_dim, embed_dim)

        self.pe = PositionalEmbedding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.spe = LaplacianPE(lape_dim, embed_dim)

    def forward(self, x, lap_mx):
        origin_x = x  # [64, 9, 170, 12]
        x = self.value_embedding(origin_x[:, :self.feature_dim, :, :])  # [64, 32, 170, 12]

        # add positional embedding
        b, D, N, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, T, D)  # [64, 170, 12, 32]->[64*170, 12, 32]
        x = x * math.sqrt(self.embed_dim)  # 这是为啥来着？为啥放大几倍？
        x = x + self.pe(x)
        x = x.reshape(b, -1, T, D).permute(0, 3, 1, 2)  # [64, 32, 170, 12]

        # add periodic embedding
        x = x.permute(0, 3, 2, 1)  # [64, 12, 170, 32]
        origin_x = origin_x.permute(0, 3, 2, 1)  # [64, 12, 170, 9]
        # if self.add_time_in_day:
        #     x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        # if self.add_day_in_week:
        #     x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))

        # add spatial embedding [64, 12, 170, 32]
        spe = self.spe(lap_mx)  # [1, 1, 170, 32]
        x = x + spe
        x = x.permute(0, 3, 2, 1)  # [64, 32, 170, 12]

        return x


class Cell(nn.Module):

    def __init__(self, genotype, C, steps, supports, nodevec1, nodevec2, ge_mask, se_mask, seq_len):
        super(Cell, self).__init__()

        self.preprocess = Identity()
        self._steps = steps
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.supports = supports
        self.geo_mask = ge_mask
        self.sem_mask = se_mask
        self.seq_len = seq_len

        indices, op_names = zip(*genotype)
        self._compile(C, indices, op_names)

    def _compile(self, C, indices, op_names):
        assert len(op_names) == len(indices)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            if 'diff_gcn' in PRIMITIVES[name]:
                op = OPS[PRIMITIVES[name]](C, self.supports, self.nodevec1, self.nodevec2, True)
            elif 'mix' in PRIMITIVES[name]:
                op = OPS[PRIMITIVES[name]](C, self.nodevec1, self.nodevec2)
            elif 'NLinear' in PRIMITIVES[name]:
                op = OPS[PRIMITIVES[name]](C, self.seq_len)
            elif 'mask' in PRIMITIVES[name]:
                op = OPS[PRIMITIVES[name]](C, self.geo_mask, self.sem_mask)
            else:
                op = OPS[PRIMITIVES[name]](C)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0):
        s0 = self.preprocess(s0)

        states = [s0]
        for i in range(self._steps):
            if i == 0:
                # assert 0 == self._indices[0]
                h1 = states[0]
                op1 = self._ops[0]
                h1 = op1(h1)
                h2 = torch.zeros_like(h1)
            else:
                h1 = states[self._indices[2 * i - 1]]
                h2 = states[self._indices[2 * i]]
                op1 = self._ops[2 * i - 1]
                op2 = self._ops[2 * i]
                h1 = op1(h1)
                h2 = op2(h2)  # 因为有bias导致不为0
            s = h1 + h2
            states += [s]

        output = []
        for j in range(self._steps // 2):
            output.append(states[2 * j + 1])
        output.append(states[-1])

        return output


class Network(nn.Module):

    def __init__(self, adj_mx, args, arch, ge_mask, se_mask, lape_dim=8):
        super(Network, self).__init__()
        self._args = args
        self._layers = args.layers
        self._steps = args.steps

        # for diff_gcn
        new_adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        supports = [torch.tensor(i).to(DEVICE) for i in new_adj]

        # for
        self.lap_mx = cal_lape(adj_mx, lape_dim=lape_dim).to(DEVICE)
        if self.lap_mx.shape[1] <= lape_dim:
            self.lap_mx = torch.FloatTensor(adj_mx).to(DEVICE)

        self.lape_dim = self.lap_mx.shape[1]

        self.nodevec1 = nn.Parameter(torch.randn(args.num_nodes, 10).to(DEVICE), requires_grad=True).to(DEVICE)
        self.nodevec2 = nn.Parameter(torch.randn(10, args.num_nodes).to(DEVICE), requires_grad=True).to(DEVICE)

        C = args.hid_dim
        self.cells = nn.ModuleList()
        self.skip_connect = nn.ModuleList()
        for i in range(self._layers):
            #geo_mask和seo_mask在启用时要to(DEVICE)
            cell = Cell(arch, C, self._steps, supports, self.nodevec1, self.nodevec2, ge_mask.to(DEVICE), se_mask, args.seq_len)
            self.cells += [cell]

        for i in range(self._layers * self._steps // 2):
            # skip_connect number is 2 or 3 times the number of layers (steps=4 or 6)
            self.skip_connect.append(nn.Conv2d(C, args.hid_dim * 8, (1, 1)))

        # input layer
        # self.start_linear = linear(args.in_dim, args.hid_dim)
        self.embedding_layer = DataEmbedding(args.in_dim, args.hid_dim, lape_dim=self.lape_dim, device=DEVICE)

        # output layer
        self.end_linear_1 = linear(c_in=args.hid_dim * 8, c_out=args.hid_dim * 16)  # 改成4和256试试？
        self.end_linear_2 = linear(c_in=args.hid_dim * 16, c_out=args.seq_len)

        # # position encoding
        # self.pe = PositionalEmbedding(args.hid_dim)

    def forward(self, input):
        # x = self.start_linear(input)
        x = self.embedding_layer(input, self.lap_mx)  # [64, 32, 170, 12]
        skip = 0

        for i, cell in enumerate(self.cells):
            output = cell(x)
            for j in range(len(output) - 1):
                skip = self.skip_connect[i * self._steps // 2 + j](output[j]) + skip
            x = output[-1]

        state = torch.max(F.relu(skip), dim=-1, keepdim=True)[0]
        out = F.relu(self.end_linear_1(state))
        logits = self.end_linear_2(out)

        return logits
