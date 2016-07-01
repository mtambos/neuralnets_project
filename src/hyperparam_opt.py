#! /usr/bin/env python

from __future__ import division

import os
import pprint

from joblib import Parallel, delayed, dump
import pandas as pd
# import psutil
import numpy as np
from sklearn.grid_search import ParameterSampler
from sklearn.preprocessing import MinMaxScaler
import rtnorm as rt
from scipy.stats import randint

from utils import generate_waveforms, NeuronConfig
from mgng import mgng


class TruncNorm(object):
    def __init__(self, a, b, loc, scale):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def rvs(self):
        return rt.rtnorm(self.a, self.b, mu=self.loc, sigma=self.scale)[0]


# noinspection PyPep8Naming
def transform_data(X, window_size):
    data_len = X.shape[0] - window_size + 1

    transformed_data = np.array([X[i: i+window_size].T.flatten()
                                 for i in xrange(data_len)])

    return transformed_data


# noinspection PyPep8Naming
def fit_with_params(params, X, firings, window_size, i):
    X = transform_data(X.as_matrix(), window_size)
    pid = os.getpid()
    print "fitting {}th iteration. PID: {}".format(i, pid)
    if params['e_n'] > params['e_w']:
        params['e_w'], params['e_n'] = params['e_n'], params['e_w']

    spk_aggr_func = params['spk_aggr_func']
    nrn_aggr_func = params['nrn_aggr_func']
    dist_metric = params['dist_metric']
    mgng_params = dict(params)
    del mgng_params['spk_aggr_func']
    del mgng_params['nrn_aggr_func']
    del mgng_params['dist_metric']
    try:
        estimator = mgng.MGNG(**mgng_params)
        estimator.fit(X)
        winner_units = estimator.transform(X)
        score = mgng.scorer(winner_units, window_size,
                            firings[firings.fire_idx <
                                    (len(winner_units) - window_size)],
                            spk_aggr_func, nrn_aggr_func, dist_metric)
        ret_val = score + (params, pid)
        pprint.pprint(ret_val)
        dump(winner_units, 'winner_units_{}.pickle'.format(pid), compress=3)
    except Exception as e:
        pprint.pprint(e)
        ret_val = (-np.infty, -np.infty, np.infty, params, pid)

    print "{}th iteration finished. PID: {}".format(i, pid)
    with open('hyperparam_opt_{}.log'.format(pid), 'ab') as fp:
        fp.write('{}\n'.format(pprint.pformat(ret_val)))
    return ret_val


def main():
    np.random.seed(0)
    firing_rates = np.random.random(size=3)/1000
    window_scales = np.random.randint(10, size=3)
    neuron_params = [NeuronConfig(f, w) for f in firing_rates
                     for w in window_scales]
    neuron_nr = len(neuron_params)
    channels_nr = 5
    data_len = int(2e6)
    window_size = 42

    data_fname = '{}n_{}c_{}l.csv'.format(neuron_nr, channels_nr, data_len)
    firings_fname = '{}n_{}c_{}l_firings.csv'.format(neuron_nr, channels_nr,
                                                     data_len)

    print "looking for files {}, {}".format(data_fname, firings_fname)
    if os.path.exists(data_fname):
        print "files {}, {} found".format(data_fname, firings_fname)
        data = pd.read_csv(data_fname)
        firings = pd.read_csv(firings_fname, index_col=0)
        firings.columns = ['neuron', 'fire_idx']
    else:
        print "files {}, {} not found, generating data".format(data_fname,
                                                               firings_fname)
        data, neurons = generate_waveforms(data_len=data_len,
                                           channels_nr=channels_nr,
                                           neuron_params=neuron_params,
                                           window_size=window_size)

        data = pd.DataFrame(data)
        data['index'] = data.index
        data.to_csv(data_fname, index=False)

        firings = [[i, j]
                   for i, n in enumerate(neurons)
                   for j in n['fire_seq']]
        firings = pd.DataFrame(firings, columns=['neuron', 'fire_idx'])
        firings.sort_values('fire_idx', inplace=True)
        firings.set_index('fire_idx', drop=False, inplace=True)
        firings.to_csv(firings_fname, index=False)

    param_space = {
        # 'alpha': [0.39794323643107143],
        # 'beta': [0.47455583402142376],
        # 'gamma': [8443],
        # 'e_w': [0.9185565637630273],
        # 'e_n': [0.41185290298184318],
        # 'eta': [0.34436856439985564],
        # 'theta': [5],
        'alpha': TruncNorm(0, 1, loc=0.32, scale=0.2),
        'beta': TruncNorm(0, 1, loc=0.26, scale=0.2),
        'gamma': randint(5000, 10000),
        'e_w': TruncNorm(0, 1, loc=0.82, scale=0.2),
        'e_n': TruncNorm(0, 1, loc=0.3, scale=0.2),
        'eta': TruncNorm(0, 1, loc=0.25, scale=0.2),
        'theta': randint(20, 80),
        'dimensions': [channels_nr*window_size],
        'spk_aggr_func': ['sum'],
        'nrn_aggr_func': ['median'],
        'dist_metric': ['hamming'],
        'verbose': [False]
    }

    param_sampler = ParameterSampler(param_space, n_iter=30)

    upper_limit = len(data)
    data = data.iloc[:upper_limit, :-1]
    # noinspection PyPep8Naming
    X = pd.DataFrame(MinMaxScaler().fit_transform(data),
                     columns=data.columns, index=data.index)
    partial_firings = firings[(firings.fire_idx < upper_limit)]

    # n_jobs = psutil.cpu_count() - 2
    n_jobs = 4
    nbytes = sum(block.values.nbytes for block in X.blocks.values())
    parallel = Parallel(n_jobs=n_jobs, max_nbytes=nbytes)
    result = parallel(delayed(fit_with_params)(params, X, partial_firings,
                                               window_size, i)
                      for i, params in enumerate(param_sampler))

    result = sorted(result)
    dump(result, './hyperparam_opt_result.pickle', compress=3)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
