# FACTS: Fully Automated Correlated Time Series Forecasting in Minutes

FACTS is an efficient and fully automated CTS forecasting framework that can find an ST-block that offers optimal accuracy on arbitrary unseen tasks and can make forecasts in minutes.

We provide the source code of FACTS in this repository. 

We are further optimizing our code and welcome any suggestions for modifications.

## Dataset link

You can obtained the well pre-processed datasets from [DropBox](https://www.dropbox.com/scl/fi/49385685dgo0grdfb476w/datasets.zip?rlkey=n303gyh7w7zunecxw2hzastt7&dl=0).Then place the downloaded data under the folder `./dataset`.Click on the link below to access the experimental datasets in the paper.

## Structure of FACTS

We introduce some main files in our project as follows:

- NAS_Net/    The main code of FACTS.

  - ArchPredictor  The main structure of TAP.

  - operations.py  The operators used in FACTS' search space.

  - st_net.py    The main backbone of FACTS.

  - operations_inherit.py   The main methodology of weights inheritance.

- exps/    Experimental scripts.

  - collect_seeds.py   The process of iteratively pruning the search space, sampling the ST-blocks and training them.

  - generate_seeds.py   The script for training the optimal ST-blocks.

  - dataset_slice.py    The script for creating CTS forecasting tasks.

  - Generate_task_feature.py  Generating the semantic features for tasks.

  - Generate_statistical_feature.py  Generating the statistical features for tasks.

- scripts/ automated testing scripts

- utils.py    Some utility functions such like dataloader, normalization.

  

## Quick start

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```shell
pip install -r requirements.txt
```

### Example usage

```shell
# Create tasks for pretraining
python ./exps/dataset_slice.py

# Generate statistical and semantic features for all subsets
bash ./scripts/generate_statistical_feature_for_subsets.sh &
bash ./scripts/generate_task_feature_for_subsets.sh

# Collect samples and iteratively prune the search space
bash ./scripts/pretrain_0.sh
bash ./scripts/pretrain_1.sh
bash ./scripts/pretrain_2.sh
bash ./scripts/pretrain_3.sh
bash ./scripts/pretrain_4.sh
bash ./scripts/pretrain_5.sh
bash ./scripts/pretrain_6.sh

# Pretrain the TAP
python ./exps/rank_the_optimal_subspace.py --mode train

# Search the optimal ST-block
python ./exps/rank_the_optimal_subspace.py --mode search --dataset <dataset_name> --seq_len <the windwow size> 

# Train the optimal ST-blocks with fast parameter adaptation
python ./exps/generate_seeds.py --mode train --dataset <dataset_name> --seq_len <the windwow size>
```



