
import argparse
import concurrent.futures

import numpy as np
import os
from utils import set_seed
import pandas as pd
import NAS_Net.ts2vec_model.datautils as datautils
from tsfresh import extract_features


def convert_to_time_series_container_and_extract_features(data: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df.insert(0, "id", 1)
    df.insert(1, "time", range(df.shape[0]))
    features = extract_features(df, column_id="id", column_sort="time", n_jobs=0)

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='The dataset name')
    parser.add_argument('--loader', type=str, required=True,
                        help='The data loader used to load the experimental data.')
    parser.add_argument('--repr_dims', type=int, default=128, help='The representation dimension (defaults to 320)')
    parser.add_argument('--seed', type=int, default=301, help='The random seed')
    parser.add_argument('--ratio', nargs='+', type=float, default=[0.6, 0.2, 0.2])

    args = parser.parse_args()

    set_seed(args.seed)
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    print('Loading data... ', end='')
    if args.loader == 'PEMS':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_forecast_PEMS(args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'ETT':
        data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = datautils.load_forecast_csv(
            args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'h5':
        data, train_slice, valid_slice, test_slice, scaler, n_covariate_cols = datautils.load_forecast_h5(
            args.dataset, args.ratio)
        train_data = data[:, train_slice]
    elif args.loader == 'txt':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_forecast_txt(
            args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'subset':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_subset_npy(args.dataset)
        train_data = data[:, train_slice]
    elif args.loader == 'csv':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_csv(args.dataset, args.ratio)
        train_data = data[:, train_slice]
    elif args.loader == 'npy':
        data, train_slice, valid_slice, test_slice, scaler = datautils.load_npy(args.dataset, args.ratio)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Unknown loader {args.loader}.")

    print('done')


    train_data = train_data[:, :, 0]
    train_data = np.transpose(train_data, (1, 0))
    train_data = scaler.inverse_transform(train_data)
    train_data = np.mean(train_data, axis=1, keepdims=False)
    statistical_feature = convert_to_time_series_container_and_extract_features(train_data).values

    dir = f'../statistical_feature/{args.dataset}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(os.path.join(dir, f'statistical_feature.npy'), statistical_feature)

    print("Finished.")
