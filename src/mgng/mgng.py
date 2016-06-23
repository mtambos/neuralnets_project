#!/usr/bin/env python
# ----------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------
"""
@author: Mario Tambos
Based on:
    Andreakis, A.; Hoyningen-Huene, N. v. & Beetz, M.
    Incremental unsupervised time series analysis using merge growing neural
    gas
    Advances in Self-Organizing Maps, Springer, 2009, 10-18
"""

from __future__ import print_function, division

from numpy.random import random_sample

import networkx as nx
import numpy as np
from numba import autojit
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder


# noinspection PyPep8Naming
@autojit(target='cpu')
def _distances(xt, W, C, c_t, alpha):
    r"""
    :param xt: current data point
    :param W: weight matrix
    :param C: temporal context matrix
    :param c_t: current temporal context vector
    :param alpha: balance between xt and c_t
    :returns: d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
    """
    xt_w = (1 - alpha)*(xt - W)**2
    ct_c = alpha*(c_t - C)**2
    # noinspection PyTypeChecker
    tot = np.sum(xt_w + ct_c, axis=1)
    return tot


# noinspection PyPep8Naming
@autojit(target='cpu')
def _find_winner_neurons(xt, W, C, c_t, alpha):
    r"""
    :param xt: current data point
    :param W: weight matrix
    :param C: temporal context matrix
    :param c_t: current temporal context vector
    :param alpha: balance between xt and c_t
    :returns: d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
    find winner r := arg min_{n \in K} d_n(t)
    and second winner q := arg min_{n \in K\{r}} d_n(t)
    where d_n(t) = (1 - \alpha) * ||x_t - w_n||^2 + \alpha||C_t - c_n||^2
    """
    xt_w = (1 - alpha)*(xt - W)**2
    ct_c = alpha*(c_t - C)**2
    # noinspection PyTypeChecker
    dists = np.sum(xt_w + ct_c, axis=1)
    r, q = dists.argpartition(1)[:2]
    dist_r = dists[r]
    dist_q = dists[q]
    if dist_r > dist_q:
        r, dist_r, q, dist_q = q, dist_q, r, dist_r
    return dist_r, r, dist_q, q


def _get_spike_encodings(firings, unit_encodings, window_size, spk_aggr_func):
    fire_ids = np.unique(firings.fire_idx)
    # noinspection PyTypeChecker
    indexes = np.array([np.arange(i, i + window_size) for i in fire_ids])
    if spk_aggr_func == 'mean':
        encodings = unit_encodings[indexes].mean(axis=1)
        encodings = pd.DataFrame(encodings, index=fire_ids)
    else:
        encodings = unit_encodings[indexes].sum(axis=1).astype(bool)
        encodings = pd.DataFrame(encodings, index=fire_ids, dtype=int)
    return firings.join(encodings)


def _get_encoding_mean_corr(encoding):
    corr_matrix = encoding.iloc[:, 2:].T.corr()
    corr_len = len(corr_matrix)
    corr_matrix = corr_matrix.as_matrix()
    indexes = np.tril_indices(corr_len, -1)
    corr_matrix = corr_matrix[indexes]
    return corr_matrix.mean()


def _get_encoding_mean_dist(encoding, dist_metric):
    distances = pdist(encoding.iloc[:, 2:].T, metric=dist_metric)
    return distances.mean()


def scorer(winner_units, window_size, firings, spk_aggr_func, nrn_aggr_func,
           dist_metric):
    unit_encoder = OneHotEncoder(sparse=False)
    unit_encodings = unit_encoder.fit_transform(winner_units[:, np.newaxis])
    spike_encodings = _get_spike_encodings(firings, unit_encodings,
                                           window_size, spk_aggr_func)

    spike_encoding_mean_dist = spike_encodings.groupby('neuron').apply(
        _get_encoding_mean_dist, args=(dist_metric,)
    )
    spike_encoding_error = spike_encoding_mean_dist.mean()

    if nrn_aggr_func == 'median':
        mean_neuron_encodings = spike_encodings.groupby('neuron').apply(
            lambda g: g.iloc[:, 2:].median().astype(bool).astype(int)
        )
    else:
        mean_neuron_encodings = spike_encodings.groupby('neuron').apply(
            lambda g: g.iloc[:, 2:].mean()
        )
    neuron_encoding_error = _get_encoding_mean_dist(mean_neuron_encodings,
                                                    dist_metric)

    ret_val = neuron_encoding_error - spike_encoding_error
    return ret_val, spike_encoding_error, neuron_encoding_error


class MGNG(BaseEstimator):
    def __init__(self, dimensions=1, alpha=0.5, beta=0.75, gamma=88,
                 delta=0.5, theta=100, eta=0.9995, lmbda=600,
                 e_w=0.05, e_n=0.0006, verbose=False):
        """
        :param dimensions: (int) dimensionality of the input vectors
        :param alpha: (float in (0,1)) balances the importance of the input
                      vector (max at 0) against the temporal context (max at 1)
                      when searching for units
        :param beta: (float in (0,1)) balances the importance of the input
                      vector (max at 0) against the temporal context (max at 1)
                      when building the global temporal context
        :param gamma: (int) maximum age the edges are allowed to reach before
                      being deleted
        :param delta: (float in (0,1)) controls the error with which new units
                      are initialized, and the fraction by which old units'
                      error is decreased according to:
                        e_q := error of unit with biggest error
                        e_f = error of e_q's neighbor with biggest error
                        new_unit_error := delta * (e_q + e_f)
                        e_q := (1-delta)*e_q
                        e_f := (1-delta)*e_f
        :param theta: (int) maximum number of units the network is allowed to
                      have
        :param eta: (float in (0,1)) fraction by which to reduce the errors of
                    all units at each time step
        :param lmbda: (int) number of time steps between unit creations.
        :param e_w: (float in (0,1)) winner unit's learning rate. Its weight
                    vector will be moved by this fraction toward the input
                    vector, and its context vector will be moved by this
                    fraction toward the global temporal context.
        :param e_n: (float in (0,1)) second winner unit's learning rate. Its
                    weight vector will be moved by this fraction toward the
                    input vector, and its context vector will be moved by this
                    fraction toward the global temporal context.
        :param verbose:
        """
        self.verbose = verbose
        self.dimensions = dimensions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.theta = theta
        self.eta = eta
        self.lmbda = lmbda
        self.e_w = e_w
        self.e_n = e_n
        # 4. initialize global temporal context C1 := 0
        self.c_t = np.zeros(dimensions)
        # 1. time variable t := 0
        self.t = None
        self.winners = None
        # 3. initialize connections set E \in K * K := \empty;
        self.model = nx.Graph()
        self.empty_row = None
        self.weights = None
        self.contexts = None
        self.errors = None
        self.matrix_indices = None
        self._init_matrices()
        # 2. initialize neuron set K with 2 neurons with counter e := 0
        # and random weight and context vectors
        self.param_names = {'dimensions', 'alpha', 'beta', 'gamma', 'delta',
                            'theta', 'eta', 'lmbda', 'e_w', 'e_n', 'verbose'}
        self.params = {k: getattr(self, k) for k in self.param_names}
        self._add_node()
        self._add_node()

    def _init_matrices(self):
        dimensions = self.dimensions
        theta = self.theta
        self.t = 0
        self.empty_row = np.array([np.nan]*dimensions)
        self.weights = np.array([[np.nan]*dimensions]*theta)
        self.contexts = np.array([[np.nan]*dimensions]*theta)
        self.errors = np.array([np.nan]*theta)
        self.matrix_indices = np.zeros(theta)
        self.winners = np.array([])

    def _update_neighbors(self, r, xt):
        """
        update neuron r and its direct topological neighbors N_r:
            w_r := w_r + \epsilon_w * (x_t - w_r)
            c_r := c_r + \epsilon_w*(C_t - c_r)
            (\forall n \in N_r)
                w_n := w_n + \epsilon_n * (x_t - w_i)
                c_n := c_n + \epsilon_n*(C_t - c_i)
        """
        self.weights[r] += self.e_w * (xt - self.weights[r])
        self.contexts[r] += self.e_w * (self.c_t - self.contexts[r])
        for n in self.model.neighbors(r):
            self.weights[n] += self.e_n * (xt - self.weights[n])
            self.contexts[n] += self.e_n * (self.c_t - self.contexts[n])

    def _increment_edges_age(self, r):
        """
        increment the age of all edges connected with r
            age_{(r,n)} := age_{(r,n)} + 1 (\forall n \in N_r )
        """
        for (u, v) in self.model.edges(r):
            self.model[u][v]['age'] += 1

    def _add_node(self, e=0, w=None, c=None):
        if w is None:
            w = random_sample(self.dimensions)
        else:
            w = np.reshape(w, newshape=self.dimensions)
        if c is None:
            c = random_sample(self.dimensions)
        else:
            c = np.reshape(c, newshape=self.dimensions)
        idx = self.matrix_indices.argmin()
        self.matrix_indices[idx] = True
        self.errors[idx] = e
        self.weights[idx] = w
        self.contexts[idx] = c
        self.model.add_node(idx)
        if self.verbose:
            print('Node {} added.'.format(idx))
        return idx

    def _remove_node(self, idx):
        self.matrix_indices[idx] = False
        self.errors[idx] = np.nan
        self.weights[idx] = self.empty_row
        self.contexts[idx] = self.empty_row
        self.model.remove_node(idx)
        if self.verbose:
            print('Node {} removed.'.format(idx))

    def _add_edge(self, r, s):
        if r == s:
            raise Exception('cannot connect edge to itself')
        if s in self.model.neighbors(r):
            self.model[r][s]['age'] = 0
        else:
            self.model.add_edge(r, s, age=0)

    def _remove_old_edges(self):
        """
        remove old connections E := E \ {(a, b)| age_(a, b) > \gamma}
        """
        for (u, v) in self.model.edges():
            if self.model.edge[u][v]['age'] > self.gamma:
                self.model.remove_edge(u, v)

    def _remove_unconnected_neurons(self):
        """
        """
        for n in self.model.nodes():
            if not self.model.degree(n):
                self._remove_node(n)

    def _create_new_neuron(self):
        """
        create new neuron if t mod \lambda = 0 and |K| < \theta
            a. find neuron q with the greatest counter:
               q := arg max_{n \in K} e_n
            b. find neighbor f of q with f := arg max_{n \in N_q} e_n
            c. initialize new neuron l
                K := K \cup l
                w_l := 1/2 * (w_q + w_f)
                c_l := 1/2 * (c_q + c_f)
                e_l := \delta * (e_f + e_q)
            d. adapt connections: E := (E \ {(q, f)}) \cup {(q, n), (n, f)}
            e. decrease counter of q and f by the factor \delta
                e_q := (1 - \delta) * e_q
                e_f := (1 - \delta) * e_f
        """
        q = np.nanargmax(self.errors)
        # noinspection PyPep8Naming
        N_q = None
        if q:
            # noinspection PyPep8Naming
            N_q = self.model.neighbors(q)
        if N_q:
            f = max(N_q, key=lambda n: self.errors[n])
            # noinspection PyTypeChecker
            l = self._add_node(e=self.delta*(self.errors[q] + self.errors[f]),
                               w=(self.weights[q] + self.weights[f]) / 2,
                               c=(self.contexts[q] + self.contexts[f]) / 2)
            self.model.remove_edge(q, f)
            self._add_edge(q, l)
            self._add_edge(f, l)
            self.errors[q] *= (1 - self.delta)
            self.errors[f] *= (1 - self.delta)

            return l

    def set_params(self, **params):
        self.params.update(params)
        [setattr(self, k, v) for k, v in self.params.iteritems()]
        self.model = nx.Graph()
        self._init_matrices()
        self._add_node()
        self._add_node()
        return self

    def get_params(self, deep=True):
        return self.params

    # noinspection PyPep8Naming
    def fit(self, X):
        X = np.asarray(X)
        verbose = self.verbose
        points_nr = len(X)
        winners = np.zeros(points_nr)
        for t, xt in enumerate(X):
            if verbose and t % (points_nr//10) == 0:
                print("{}% done".format(100*t//points_nr))
            winners[t], _ = self.time_step(xt)

        self.winners = winners
        return self

    # noinspection PyPep8Naming
    def transform(self, X):
        X = np.asarray(X)
        verbose = self.verbose
        points_nr = len(X)
        winners = np.zeros(points_nr)
        for t, xt in enumerate(X):
            if verbose and t % (points_nr//10) == 0:
                print("{}% done".format(100*t//points_nr))
            winners[t], _ = self.time_step(xt)

        return winners

    def find_winner_neurons(self, xt):
        return _find_winner_neurons(xt, self.weights, self.contexts,
                                    self.c_t, self.alpha)
        
    def time_step(self, xt, train=True):
        """
        :param xt: current data point
        :param train: whether to update the current parameters
        """
        # 6. find winner r and second winner s
        xt = np.reshape(xt, newshape=self.dimensions)
        winners = _find_winner_neurons(xt, self.weights, self.contexts,
                                       self.c_t, self.alpha)
        r_dist, r, _, s = winners

        # 7. Ct+1 := (1 - \beta)*w_r + \beta*c_r
        c_t1 = (1 - self.beta) * self.weights[r] + self.beta * self.contexts[r]

        if train:
            # 8. connect r with s: E := E \cup {(r, s)}
            # 9. age(r;s) := 0
            self._add_edge(r, s)

            # 10. increment counter of r: e_r := e_r + 1
            self.errors[r] += 1

            # 11. update neuron r and its direct topological neighbors:
            self._update_neighbors(r, xt)

            # 12. increment the age of all edges connected with r
            self._increment_edges_age(r)

            # 13. remove old connections E := E \ {(a, b)| age_(a, b) > \gamma}
            self._remove_old_edges()

            # 14. delete all nodes with no connections.
            self._remove_unconnected_neurons()

            # 15. create new neuron if t mod \lambda = 0 and |K| < \theta
            if self.t % self.lmbda == 0 and len(self.model.nodes()) < self.theta:
                self._create_new_neuron()

            # 16. decrease counter of all neurons by the factor \eta:
            #    e_n := \eta * e_n (\forall n \in K)
            self.errors *= self.eta

        # 7. Ct+1 := (1 - \beta)*w_r + \beta*c_r
        self.c_t = c_t1

        # 17. t := t + 1
        self.t += 1
        
        return r, r_dist


def main():
    # noinspection PyPep8Naming
    import Oger as og
    import pylab
    signal = og.datasets.mackey_glass(sample_len=1500,
                                      n_samples=1,
                                      seed=50)[0][0].flatten()
    print(signal)
    signal = signal + np.abs(signal.min())
    print(signal)
    # 2. initialize neuron set K with 2 neurons with counter e := 0 and random
    #    weight and context vectors
    # 3. initialize connections set E \in K * K := \empty;
    # 4. initialize global temporal context C1 := 0
    mgng = MGNG(lmbda=6)
    # 1. time variable t := 1
    # 5. read / draw input signal xt
    # 18. if more input signals available go to step 5 else terminate
    for t, xt in enumerate(signal, 1):
        mgng.time_step(xt)
        if t % 1500 == 0:
            print('training: %i%%' % (t / 1500))

    errors = [[] for _ in range(30)]
    for t, xt in enumerate(signal, 1):
        if t % 150 == 0:
            print('calculating errors: %i%%' % (t / 150))
        _, n, _, _ = mgng.find_winner_neurons(xt)
        for i in range(min(30, t)):
            # noinspection PyTypeChecker
            error = np.mean((mgng.weights[n] - signal[t - i - 1]) ** 2)
            # noinspection PyTypeChecker
            errors[i].append(error)

    summary = [0] * 30
    for i in range(30):
        summary[i] = np.sum(errors[i]) / len(errors[i])

    pylab.subplot(2, 1, 1)
    pylab.plot(range(30), summary)

    pylab.subplot(2, 1, 2)
    pylab.plot(range(len(mgng.model.nodes())),
               [n[1]['w'] for n in mgng.model.nodes(data=True)])
    pylab.show()


if __name__ == '__main__':
    main()
