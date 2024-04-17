from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')  # 前面是类名，后面是字段名

PRIMITIVES = [
    # 'none',
    'skip_connect',  # 0
    'NLinear',  # 1

    'cnn',  # 2
    'dcc_1',  # 3
    'inception',  # 4

    'gru',  # 5
    'lstm',  # 6

    'diff_gcn',  # 7
    'mix_hop',  # 8

    'trans',  # 9
    'informer',  # 10
    'convformer',  # 11

    's_trans',  # 12
    's_informer',  # 13
    'masked_trans',  # 14
]
