# FACTS: Fully Automated Correlated Time Series Forecasting in Minutes

FACTS is an efficient and fully automated CTS forecasting framework that can find an ST-block that offers optimal accuracy on arbitrary unseen tasks and can make forecasts in minutes.

We provide the source code of FACTS in this repository. 

We are further optimizing our code and welcome any suggestions for modifications.

## Dataset link

You can obtained the well pre-processed datasets from [DropBox](https://www.dropbox.com/scl/fi/49385685dgo0grdfb476w/datasets.zip?rlkey=n303gyh7w7zunecxw2hzastt7&dl=0).Then place the downloaded data under the folder `./dataset`.Click on the link below to access the experimental datasets in the paper.

## Structure of FACTS

We introduce some main files in our project as follows:

- `NAS_Net/ `   The main code of FACTS.

  - `ArchPredictor`  The main structure of TAP.

  - `operations.py`  The operators used in FACTS' search space.

  - `st_net.py `   The main backbone of FACTS.

  - `operations_inherit.py`  `st_net_1.py`   The main methodology of weights inheritance.

- `exps/`    Experimental scripts.

  - `collect_seeds.py`   The process of iteratively pruning the search space, sampling the ST-blocks and training them.

  - `generate_seeds.py`   The script for training the optimal ST-blocks.

  - `dataset_slice.py`    The script for creating CTS forecasting tasks.

  - `Generate_task_feature.py`  Generating the semantic features for tasks.

  - `Generate_statistical_feature.py`  Generating the statistical features for tasks.

- `scripts/` automated testing scripts

- `utils.py`    Some utility functions such like dataloader, normalization.

  

## Quick start

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```



## Experiments reproduction

### Main results

We provide all the scripts to repoduce the main experimental results in our paper:

As illustrated in FACTS,  we first iteratively prune the large search space through a GBM-based pruner, while the pruning process is coupled with the collection of seeds. We provide the correspongding scripts as belows:

First, we create **100**  tasks from 11 datasets equipped with two forecasting settings for pretraining, this process is implemented by a heuristic slicing method.  With the random seed provided in the code, we can reproduce the process and generate subsets in a specified dir ./subsets , we also show the task settings in `exps/Task_config.py` to ensure the reproducibility of pretraining tasks:

```shell
# Create tasks for pretraining
python ./exps/dataset_slice.py
```

Then we generate statistical features for all subsets with the [tsfresh](https://tsfresh.readthedocs.io/en/latest/) package:

```shell
# pip install tsfresh
# Generate statistical features for all subsets
bash ./scripts/generate_statistical_feature_for_subsets.sh
```

The semantic features are generated by representation learning model [TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/view/20881), we inregrate it into our code:

```shell
# Generate semantic features for all subsets
bash ./scripts/generate_task_feature_for_subsets.sh
```

After running the scripts above, we create all the static information for tasks, we then **iteratively prune the search space** and collect the seeds in each phase. Specifically, we can prune the search space with the script below, and there are some hyperparameters needed to be changed in the script to keep a **half-cut**:

```shell
# iteratively prune the search space
python ./exps/iteratively_search_space_pruning.py --mode iteratively
```

Then we collect the seeds in the pruned search space by parallel running on 32 A800 gpus. We provide the scripts used in our machine. One can edit their own gpu number in the scripts:

```shell
# Collect seeds
bash ./scripts/pretrain_0.sh &
bash ./scripts/pretrain_1.sh &
bash ./scripts/pretrain_2.sh &
bash ./scripts/pretrain_3.sh &
bash ./scripts/pretrain_4.sh &
bash ./scripts/pretrain_5.sh &
bash ./scripts/pretrain_6.sh &
```

By looping above two processes 7 times, one can prune the search space to a minimal one as illustrated in our paper. Meanwhile, a large number of seeds are collected in the whole process and stored in the dir `./seeds`. With the huge amounts of seeds and the static information (statistical features and semantic features) of each task, we can pretrain the **TAP** with the scripts below:

```shell
# Pretrain the TAP
python ./exps/rank_the_optimal_subspace.py --mode train
```

Then we can use the pretrained TAP to rank the optimal subspace pruned above and find an optimal architecture for each task:

```shell
# Search the optimal ST-block (template)
python ./exps/rank_the_optimal_subspace.py --mode search --dataset <dataset_name> --seq_len <the windwow size> --pred_len <the horizon size>

# Specific
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMS-BAY --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMS-BAY --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMS-BAY --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMS-BAY --seq_len 168 --pred_len 1

python ./exps/rank_the_optimal_subspace.py --mode search --dataset Electricity --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset Electricity --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset Electricity --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset Electricity --seq_len 168 --pred_len 1

python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMSD7M --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMSD7M --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMSD7M --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset PEMSD7M --seq_len 168 --pred_len 1

python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-TAXI --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-TAXI --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-TAXI --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-TAXI --seq_len 168 --pred_len 1

python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-BIKE --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-BIKE --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-BIKE --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset NYC-BIKE --seq_len 168 --pred_len 1

python ./exps/rank_the_optimal_subspace.py --mode search --dataset Los-Loop --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset Los-Loop --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset Los-Loop --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset Los-Loop --seq_len 168 --pred_len 1

python ./exps/rank_the_optimal_subspace.py --mode search --dataset SZ-TAXI --seq_len 12 --pred_len 12
python ./exps/rank_the_optimal_subspace.py --mode search --dataset SZ-TAXI --seq_len 24 --pred_len 24
python ./exps/rank_the_optimal_subspace.py --mode search --dataset SZ-TAXI --seq_len 48 --pred_len 48
python ./exps/rank_the_optimal_subspace.py --mode search --dataset SZ-TAXI --seq_len 168 --pred_len 1
```

After finding the optimal architectures for all the tasks, we can reproduce the SOTA results in **Table 6** and **Table 7** of our paper by training them with our **fast parameter adaptation** strategy:

```shell
# Train the optimal ST-blocks with fast parameter adaptation (template)
python ./exps/generate_seeds.py --mode train --dataset <dataset_name> --seq_len <the windwow size> --pred_len <the horizon size>

# Specific
python ./exps/generate_seeds.py --mode train --dataset PEMS-BAY --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset PEMS-BAY --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset PEMS-BAY --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset PEMS-BAY --seq_len 168 --pred_len 1

python ./exps/generate_seeds.py --mode train --dataset Electricity --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset Electricity --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset Electricity --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset Electricity --seq_len 168 --pred_len 1

python ./exps/generate_seeds.py --mode train --dataset PEMSD7M --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset PEMSD7M --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset PEMSD7M --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset PEMSD7M --seq_len 168 --pred_len 1

python ./exps/generate_seeds.py --mode train --dataset NYC-TAXI --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset NYC-TAXI --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset NYC-TAXI --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset NYC-TAXI --seq_len 168 --pred_len 1

python ./exps/generate_seeds.py --mode train --dataset NYC-BIKE --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset NYC-BIKE --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset NYC-BIKE --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset NYC-BIKE --seq_len 168 --pred_len 1

python ./exps/generate_seeds.py --mode train --dataset Los-Loop --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset Los-Loop --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset Los-Loop --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset Los-Loop --seq_len 168 --pred_len 1

python ./exps/generate_seeds.py --mode train --dataset SZ-TAXI --seq_len 12 --pred_len 12
python ./exps/generate_seeds.py --mode train --dataset SZ-TAXI --seq_len 24 --pred_len 24
python ./exps/generate_seeds.py --mode train --dataset SZ-TAXI --seq_len 48 --pred_len 48
python ./exps/generate_seeds.py --mode train --dataset SZ-TAXI --seq_len 168 --pred_len 1
```



### Other results

We compare our iteratively pruning strategy with the one-time pruning baseline, we also provide the scripts to reproduce the results of the one-time pruning strategy:

```shell
# prune the search space in one time
python ./exps/iteratively_search_space_pruning.py --mode one-shot
```

With the one-time strategy, we obtain a pruned search space and we also collect seeds from them to pretrain the TAP:

```shell
# Collect seeds
bash ./scripts/pretrain_0.sh &
bash ./scripts/pretrain_1.sh &
bash ./scripts/pretrain_2.sh &
bash ./scripts/pretrain_3.sh &
bash ./scripts/pretrain_4.sh &
bash ./scripts/pretrain_5.sh &
bash ./scripts/pretrain_6.sh &
```

```shell
# Pretrain the TAP
python ./exps/rank_the_optimal_subspace.py --mode train
```

And we can also reproduce the results by running the scripts as below:

```shell
# Search the optimal ST-block (template)
python ./exps/rank_the_optimal_subspace.py --mode search --dataset <dataset_name> --seq_len <the windwow size> --pred_len <the horizon size>

# Train the optimal ST-blocks (template)
python ./exps/generate_seeds.py --mode train --dataset <dataset_name> --seq_len <the windwow size> --pred_len <the horizon size>
```

