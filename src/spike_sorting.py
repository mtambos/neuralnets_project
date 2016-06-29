from __future__ import division, print_function

import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans

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
class SpikeSorter(BaseEstimator, ClusterMixin):
    def __init__(self, clustering_model, entr_threshold=0.1,
                 spk_aggr_func='sum', nrn_aggr_func='median',
                 dist_metric='hamming', window_size=42, sum_threshold=13,
                 **mgng_params):
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

        self.estimator = None

    def fit_predict(self, X, y=None):
        params = self.mgng_params
        assert params['e_n'] > params['e_w']

        estimator = mgng.MGNG(**params)
        self.estimator = estimator

        estimator.fit(X)
        winner_units = self.estimator.transform(X)

        unit_encoder = OneHotEncoder(sparse=False)
        unit_encodings = unit_encoder.fit_transform(
            winner_units.w1.as_matrix()[:, np.newaxis]
        )

        entropies = np.array([stats.entropy([enc.mean(), 1-enc.mean()])
                              for enc in unit_encodings.T])
        low_entr_idx = entropies <= self.entr_threshold

        denoised_unit_encodings = unit_encodings.copy()
        denoised_unit_encodings[:, low_entr_idx] = 0

        window_size = self.window_size
        windowed_encodings = get_windowed_encodings(
            denoised_unit_encodings, window_size=window_size,
            spk_aggr_func=self.spk_aggr_func
        )

        possible_spikes_idx = (windowed_encodings.sum(axis=1) >=
                               self.sum_threshold)

        mask = np.ones(window_size)
        counts = np.convolve(possible_spikes_idx, mask, mode='same')

        clustering_model = self.clustering_model
        clustering_model.fit(windowed_encodings)
        predictions = clustering_model.predict(
            windowed_encodings[counts >= window_size]
        )

        return predictions

    def score(self, X, firings):
        winner_units = self.estimator.transform(X)
        score = mgng.scorer(winner_units, self.window_size, firings,
                            self.spk_aggr_func, self.nrn_aggr_func,
                            self.dist_metric)
        return score
