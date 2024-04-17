## README

### Dataset link

Click on the link below to access the experimental datasets.

https://www.dropbox.com/scl/fi/49385685dgo0grdfb476w/datasets.zip?rlkey=n303gyh7w7zunecxw2hzastt7&dl=0



### Structure of FACTS

- NAS_Net/ the main code of FACTS

- exps/ experimental scripts 

- scripts/ automated testing scripts

- utils.py    some utility functions

  

### Quick start

```shell
# create tasks for pretraining
python ./exps/dataset_slice.py

# generate statistical and semantic features for all subsets
bash ./scripts/generate_statistical_feature_for_subsets.sh &
bash ./scripts/generate_task_feature_for_subsets.sh

# collect samples and iteratively prune the search space
bash ./scripts/pretrain_0.sh
bash ./scripts/pretrain_1.sh
bash ./scripts/pretrain_2.sh
bash ./scripts/pretrain_3.sh
bash ./scripts/pretrain_4.sh
bash ./scripts/pretrain_5.sh
bash ./scripts/pretrain_6.sh

# pretrain the TAP
python ./exps/rank_the_optimal_subspace.py --mode train

# search the optimal ST-block
python ./exps/rank_the_optimal_subspace.py --mode search
```



