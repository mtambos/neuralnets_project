#! /usr/bin/env python

from __future__ import division

import os
import pprint

from joblib import Parallel, delayed, dump
import pandas as pd
import numpy as np
from sklearn.grid_search import ParameterSampler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, truncnorm

from utils import generate_waveforms, NeuronConfig
from mgng import mgng


# noinspection PyPep8Naming
def fit_with_params(params, X, firings, window_size, i):
    pid = os.getpid()
    print "fitting {}th iteration. PID: {}".format(i, pid)
    try:
        params['alpha'] = abs(params['alpha'])
        params['beta'] = abs(params['beta'])
        params['e_n'] = abs(params['e_n'])
        params['e_w'] = abs(params['e_w'])
        params['eta'] = abs(params['eta'])
        spk_aggr_func = params['spk_aggr_func']
        del params['spk_aggr_func']
        estimator = mgng.MGNG(**params)
        estimator.fit(X)
        score = mgng.scorer(window_size, firings, estimator, spk_aggr_func)
        ret_val = score + (params, spk_aggr_func)
        pprint.pprint(ret_val)
    except Exception as e:
        pprint.pprint(e)
        ret_val = (-np.infty, -np.infty, np.infty, params, spk_aggr_func)

    print "{}th iteration finished. PID: {}".format(i, pid)
    with open('hyperparam_ot_{}.log'.format(pid), 'ab') as fp:
        fp.write('{}\n'.format(pprint.pformat(ret_val)))
    return ret_val


def _calc_truncnorm_a_b(clip_a, clip_b, loc, scale=1):
    """
    Helper function to calculate the truncated normal distribution between
    `clip_a` and `clip_b`. Used for hyperparameter optimization.
    :param clip_a: left limit of the truncated normal distribution.
    :param clip_b: right limit of the truncated normal distribution.
    :param loc: mean of the truncated normal distribution.
    :param scale: standard deviation of the truncated normal distribution.
    :return: `a` and `b`parameters of scipy.stats.truncnorm.
    """
    return (clip_a - loc) / scale, (clip_b - loc) / scale


def main():
    np.random.seed(0)
    firing_rates = np.random.random(size=3)/1000
    window_scales = np.random.randint(10, size=3)
    neuron_params = [NeuronConfig(f, w)
                     for f in firing_rates
                     for w in window_scales]
    neuron_nr = len(neuron_params)
    channels_nr = 5
    data_len = int(2e6)
    window_size = 42

    data_fname = '{}n_{}c_{}l.csv'.format(neuron_nr, channels_nr, data_len)
    firings_fname = '{}n_{}c_{}l_firings.csv'.format(neuron_nr, channels_nr,
                                                     data_len)

    if os.path.exists(data_fname):
        data = pd.read_csv(data_fname)
        firings = pd.read_csv(firings_fname, index_col=0)
        firings.columns = ['neuron', 'fire_idx']
    else:
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

    a_alpha, b_alpha = _calc_truncnorm_a_b(0, 1, 0.4)
    a_beta, b_beta = _calc_truncnorm_a_b(0, 1, 0.5)
    a_e_w, b_e_w = _calc_truncnorm_a_b(0, 1, 0.9)
    a_e_n, b_e_n = _calc_truncnorm_a_b(0, 1, 0.4)
    a_eta, b_eta = _calc_truncnorm_a_b(0, 1, 0.3)
    param_space = {
        # 'alpha': [0.39794323643107143],
        # 'beta': [0.47455583402142376],
        # 'gamma': [8443],
        # 'e_w': [0.9185565637630273],
        # 'e_n': [0.41185290298184318],
        # 'eta': [0.34436856439985564],
        # 'theta': [5],
        'alpha': truncnorm(a_alpha, b_alpha),
        'beta': truncnorm(a_beta, b_beta),
        'gamma': randint(50, 10000),
        'e_w': truncnorm(a_e_w, b_e_w),
        'e_n': truncnorm(a_e_n, b_e_n),
        'eta': truncnorm(a_eta, b_eta),
        'dimensions': [channels_nr],
        'theta': randint(5, 150),
        'dimensions': [channels_nr],
        'spk_aggr_func': ['mean', 'sum'],
        'verbose': [False]
    }

    param_sampler = ParameterSampler(param_space, n_iter=30)

    upper_limit = 1000000
    data = data.iloc[:upper_limit, :-1]
    # noinspection PyPep8Naming
    X = pd.DataFrame(MinMaxScaler().fit_transform(data),
                     columns=data.columns, index=data.index)
    partial_firings = firings[(firings.fire_idx < upper_limit)]

    nbytes = sum(block.values.nbytes for block in X.blocks.values())
    parallel = Parallel(n_jobs=6, max_nbytes=nbytes)
    result = parallel(delayed(fit_with_params)(params, X, partial_firings,
                                               window_size, i)
                      for i, params in enumerate(param_sampler))

    result = sorted(result)
    dump(result, './hyperparam_opt_result.pickle', compress=3)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
