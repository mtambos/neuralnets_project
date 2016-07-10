#! /usr/bin/env python
from __future__ import division, print_function

import json

import click
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClusterMixin
import sklearn.cluster as cluster

from mgng import mgng


def get_windowed_encodings(unit_encodings, window_size, spk_aggr_func):
    encodings_len = len(unit_encodings) - window_size + 1
    encodings = np.zeros((encodings_len, unit_encodings.shape[1]))
    for i in xrange(encodings_len):
        enc = unit_encodings[i: i+window_size, :]
        if spk_aggr_func == 'mean':
            encodings[i] = enc.mean(axis=0)
        else:
            encodings[i] = enc.sum(axis=0).astype(bool).astype(int)
    return encodings


# noinspection PyPep8Naming
def transform_data(X, window_size):
    X = X.as_matrix()
    if window_size > 1:
        data_len = X.shape[0] - window_size + 1

        transformed_data = np.array([X[i: i+window_size].T.flatten()
                                     for i in xrange(data_len)])

        return transformed_data

    return X


def get_predictions(winner_units, entr_threshold, window_size,
                    sum_threshold, spk_aggr_func, clustering_model):
    unit_encoder = OneHotEncoder(sparse=False)
    unit_encodings = unit_encoder.fit_transform(
        winner_units[:, np.newaxis]
    )

    entropies = np.array([stats.entropy([enc.mean(), 1-enc.mean()])
                          for enc in unit_encodings.T])
    low_entr_idx = entropies <= entr_threshold

    denoised_unit_encodings = unit_encodings.copy()
    denoised_unit_encodings[:, low_entr_idx] = 0

    windowed_encodings = get_windowed_encodings(
        denoised_unit_encodings, window_size=window_size,
        spk_aggr_func=spk_aggr_func
    )

    possible_spikes_idx = windowed_encodings.sum(axis=1) >= sum_threshold

    print(pd.Series(possible_spikes_idx).value_counts())
    conv_mask = np.ones(window_size)
    counts = np.convolve(possible_spikes_idx, conv_mask, mode='same')

    clustering_model = clustering_model
    clustering_model.fit(windowed_encodings)

    predictions_idx = counts >= window_size
    predictions = -1 * np.ones_like(predictions_idx)
    predictions[predictions_idx] = clustering_model.predict(
        windowed_encodings[predictions_idx]
    )

    return predictions


# noinspection PyPep8Naming
class SpikeSorter(BaseEstimator, ClusterMixin):
    def __init__(self, clustering_model, entr_threshold=0.1,
                 spk_aggr_func='sum', nrn_aggr_func='median',
                 dist_metric='hamming', window_size=42, sum_threshold=13,
                 transform_window_size=42, **mgng_params):
        assert spk_aggr_func in ('sum', 'mean')
        assert nrn_aggr_func in ('median', 'mean')

        self.mgng_params = mgng_params
        self.entr_threshold = entr_threshold
        self.spk_aggr_func = spk_aggr_func
        self.nrn_aggr_func = nrn_aggr_func
        self.dist_metric = dist_metric
        self.window_size = window_size
        self.sum_threshold = sum_threshold
        self.clustering_model = clustering_model
        self.transform_window_size = transform_window_size

        self.estimator = None

    # noinspection PyUnresolvedReferences
    def fit_predict(self, X, y=None):
        params = self.mgng_params
        assert params['e_n'] < params['e_w']

        estimator = mgng.MGNG(**params)
        self.estimator = estimator

        X_trans = transform_data(X, self.transform_window_size)
        estimator.fit(X_trans)
        winner_units = estimator.transform(X_trans)
        joblib.dump(winner_units, 'winner_units.pickle', compress=3)

        predictions = get_predictions(winner_units, self.entr_threshold,
                                       self.window_size, self.sum_threshold,
                                       self.spk_aggr_func,
                                       self.clustering_model)
        # noinspection PyTypeChecker
        predictions = np.hstack(
            (predictions, -1 * np.ones(len(X) - len(predictions)))
        )
        return predictions

    def score(self, X, firings):
        winner_units = self.estimator.transform(X)
        score = mgng.scorer(winner_units, self.window_size, firings,
                            self.spk_aggr_func, self.nrn_aggr_func,
                            self.dist_metric)
        return score


@click.command()
@click.option('--datafile', required=True, type=click.Path(),
              help='Path to the file containing the multielectrode recording.')
@click.option('--outfile', required=True, type=click.Path(),
              help='Path where the data plus the spike train will be dumped.')
@click.option('--mgng_params', required=True, type=click.Path(),
              help="Path to the JSON file containing the MGNG model's "
                   "hyperparameters.")
@click.option('--clustering', default='MiniBatchKMeans',
              help='Member of sklearn.cluster to use for grouping the spikes.')
@click.option('--entropy_threshld', default=0.1, type=float,
              help='Code dimensions with less than this entropy across the '
                   'data will be zeroed.')
@click.option('--window_size', default=42, type=int,
              help='Size of the sliding window.')
@click.option('--sum_threshld', default=13, type=int,
              help='Windowed encodings with a sum higher than this will be '
                   'considered spikes.')
@click.option('--transform_window_size', default=1, type=int,
              help='If greater than one, transform the input data by sliding '
                   'a window across it and then stacking everything as a '
                   'single time step.')
@click.option('--data_limit', default=-1, type=int,
              help='Only consider this many time steps from the data.')
def main(datafile, outfile, mgng_params, clustering, entropy_threshld,
         window_size, sum_threshld, transform_window_size, data_limit):
    cluster_cls = cluster.__dict__[clustering]
    cluster_model = cluster_cls()

    click.echo('Reading data')
    data = pd.read_csv(datafile)
    if data_limit > 0:
        data = data.iloc[:data_limit, :]
    click.echo('Sorting spikes')
    with open(mgng_params, 'rb') as fp:
        mgng_params = json.load(fp)
    sorter = SpikeSorter(cluster_model, entr_threshold=entropy_threshld,
                         window_size=window_size, sum_threshold=sum_threshld,
                         transform_window_size=transform_window_size,
                         **mgng_params)
    spike_train = sorter.fit_predict(X=data)

    click.echo('Storing spike train')

    click.echo('Storing spike train')
    data['spike_train'] = spike_train
    data.to_csv(outfile, index=False)

    click.echo('END!')


if __name__ == '__main__':
    main()
