from __future__ import division


from collections import namedtuple
import numpy as np
import scipy as sp

from nsim.scene.noise import ArNoiseGen


# noinspection PyPep8Naming
def nexp(X):
    return np.exp(-X)


NeuronConfig = namedtuple('NeuronConfig', ['firing_rate', 'window_scale'])


# noinspection PyUnusedLocal
def generate_waveforms(channels_nr=4, data_len=10000, window_size=42,
                       ar_params_nr=10, ar_params_func=nexp,
                       neuron_params=(NeuronConfig(0.001, 4),
                                      NeuronConfig(0.0005, 2))):
    neurons = [dict() for _ in neuron_params]
    neurons_nr = len(neurons)
    for i, n in enumerate(neurons):
        firing_rate = neuron_params[i].firing_rate
        window_scale = neuron_params[i].window_scale
        # single channel waveform prototype
        # two phases of sine for the basic features
        waveform_proto = np.zeros(window_size)
        waveform_proto[:window_size//2] = sp.sin(
            sp.linspace(0, 3/2*sp.pi, window_size//2)
        )
        waveform_proto[window_size//2:] = 2 * sp.sin(
            sp.linspace(3/2, 3*sp.pi, window_size//2)
        )
        # hanning window levels the edges to zero and gets
        # the overall relative scale right to generate more
        # prototypes you can just get creative with the windows
        # and/or combine more harmonics before windowing
        waveform_proto *= sp.hanning(window_size) * window_scale
        fire_sequence = np.random.poisson(
            lam=firing_rate, size=data_len-window_size
        )
        n['waveform'] = waveform_proto
        n['fire_seq'] = np.argwhere(fire_sequence != 0).flatten()

    channel_distr = np.random.random(channels_nr*neurons_nr)
    channel_distr = channel_distr.reshape(neurons_nr, channels_nr)
    proto_channels = np.zeros((neurons_nr, channels_nr, window_size))
    for n in range(neurons_nr):
        waveform = neurons[n]['waveform']
        for c in range(channels_nr):
            proto_channels[n, c] = waveform * channel_distr[n, c]

    data = np.zeros((neurons_nr, channels_nr, data_len))
    for n in range(neurons_nr):
        idx = neurons[n]['fire_seq'][:, np.newaxis]
        idx = np.repeat(idx, window_size, axis=1)
        idx += np.arange(window_size)
        for c in range(channels_nr):
            try:
                data[n, c, idx] = proto_channels[n, c]
            except IndexError:
                print n, c, idx
                raise

    noise_cov = np.random.random(size=channels_nr**2)
    noise_cov = noise_cov.reshape(channels_nr, channels_nr)
    ar_params = ar_params_func(np.arange(1, channels_nr + 1))
    ar_params = ar_params[:, np.newaxis]
    ar_params = np.repeat(ar_params, channels_nr, axis=1).T
    noise_gen = ArNoiseGen(noise_params=(ar_params, noise_cov))

    # now we have all components of the generative model.
    data = data.sum(axis=0).T.copy()
    data += noise_gen.query(size=data_len)

    return data, neurons
