#! /usr/bin/env python

from __future__ import division

import os
import pprint

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.grid_search import ParameterSampler

import pysax
from py4j.java_gateway import JavaGateway

from utils import generate_waveforms, NeuronConfig
import rtnorm as rt


class TruncNorm(object):
    def __init__(self, a, b, loc, scale):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def rvs(self):
        return rt.rtnorm(self.a, self.b, mu=self.loc, sigma=self.scale)[0]


# noinspection PyPep8Naming
def transform_data(X, gateway):
    data_len = X.shape[0]
    channels_nr = X.shape[1]
    multiTs = gateway.new_array(gateway.jvm.double, channels_nr, data_len)
    for j, xx in X.T.iterrows():
        j = int(j)
        tS = gateway.new_array(gateway.jvm.double, data_len)
        for i, x in xx.iteritems():
            tS[i] = x
        multiTs[j] = tS
    return multiTs


def build_classes_df(positionsForClasses):
    positions_for_classes = {}
    for classId in positionsForClasses.keySet():
        positionsForClass = positionsForClasses.get(classId)
        positions_for_class = {i: {'start': position[0],
                                   'end': position[1]}
                               for i, position
                               in enumerate(positionsForClass)}
        positions_for_classes[classId] = positions_for_class

    multi_index = pd.MultiIndex(levels=[[], []], labels=[[], []],
                                names=['class_id', 'position'])
    df = pd.DataFrame(index=multi_index, columns=['start', 'end'])
    for class_id, positions_for_class in positions_for_classes.iteritems():
        for p_id, position in positions_for_class.iteritems():
            df.loc[(class_id, p_id), :] = position.values()

    return df


# noinspection PyPep8Naming
def fit_with_params(params, X, i):
    gateway = JavaGateway()
    SaxGateway = gateway.jvm.SaxGateway

    X = transform_data(X, gateway)
    pid = os.getpid()
    print "fitting {}th iteration. PID: {}".format(i, pid)

    mixStrategy = params['mixStrategy']
    algorithm = params['algorithm']
    useSlidingWindow = params['useSlidingWindow']
    numRedStrategy = params['numRedStrategy']
    windowSize = params['windowSize']
    paaSizeFactor = params['paaSizeFactor']
    paaSize = int(windowSize * paaSizeFactor)
    alphabetSize = params['alphabetSize']
    normalizationThreshold = params['normalizationThreshold']
    thresholdLength = params['thresholdLength']
    thresholdCom = params['thresholdCom']
    try:
        saxGateway = SaxGateway()
        saxData = saxGateway.processData(X, mixStrategy, algorithm,
                                         useSlidingWindow, numRedStrategy,
                                         windowSize, paaSize, alphabetSize,
                                         normalizationThreshold,
                                         thresholdLength, thresholdCom)
        positionsForClasses = saxGateway.getPositionsForClasses(saxData)
        positions_for_classes = build_classes_df(positionsForClasses)
        positions_for_classes.to_csv(
            'positions_for_classes_{}.csv'.format(i)
        )
    except Exception as e:
        pprint.pprint(e)

    print "{}th iteration finished. PID: {}".format(i, pid)


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
        'mixStrategy': [0, 1],
        'algorithm': [0, 1],
        'useSlidingWindow': [True, False],
        'numRedStrategy': [0],
        'windowSize': randint(window_size//2, window_size*3),
        'paaSizeFactor': uniform(0, 1),
        'alphabetSize': randint(3, 20),
        'normalizationThreshold': [0.05],
        'thresholdLength': TruncNorm(0, 1, loc=0.1, scale=0.2),
        'thresholdCom': TruncNorm(0, 1, loc=0.5, scale=0.2),
    }

    param_sampler = ParameterSampler(param_space, n_iter=120)

    upper_limit = 100000  # len(data)
    data = data.iloc[:upper_limit, :-1]
    # noinspection PyPep8Naming

    # n_jobs = psutil.cpu_count() - 2
    n_jobs = 4
    nbytes = sum(block.values.nbytes for block in data.blocks.values())
    parallel = Parallel(n_jobs=n_jobs, max_nbytes=nbytes)
    parallel(delayed(fit_with_params)(params, data, i)
             for i, params in enumerate(param_sampler))


if __name__ == '__main__':
    main()
