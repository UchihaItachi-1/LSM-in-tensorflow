#11.11.20
#

# 5.11.20
# using speech_lsm
# Iin in line 69 will work as long as dealing with 2d matrices and not high d tensors
# 19.10.20
# lien 25 rng,


import numpy as np
#import tensorflow as tf
from matplotlib import pyplot as plt
from math import ceil

#print(tf.version)


class reservoir:
    def __init__(self, runtime_para, res_para, Gin, Gres, neuron_model, synapse_model):
        # dic of runtime and res para , Gin,Gres tenors .. neuron and synapse models have more parameters
        # self.synapse_model = synapse_model
        # self.neuron_model = neuron_model
        self.Gres = Gres
        self.Gin = Gin

        # runtime para
        self.dt = runtime_para['dt']
        self.T = runtime_para['T']
        self.t = np.arange(0, self.T, self.dt)

        # res para
        self.Nin = res_para['Nin']  # 5;       %?   no of input neurons
        self.Nres = res_para['Nres']  # 10;     %?
        ##rng(4);        %???? ask
        self.pinp = res_para['pinp']  # 0.2;    %?
        self.pres = res_para[
            'pres']  # 0.5; %? some sort of thershold on revervoir weights- why? - perhaps to make them 0and 1 like gin?(-1,1)
        self.pinh = res_para['pinh']  # 0.2;    %?  decides what fraction of res neurons are inhibitory
        self.fout = res_para['fout']  # 4;      %? no of res neurons each input neuron is connected to
        self.pinhi = res_para['pinhi']  # 0.25;  %? perhaps each ip neuron has this much inhibitory connection and 1-this exhibitory

        # synapse para
        self.Iwave = synapse_model['Iwave']  # rest dont seem imp
        # 1x A <- A depends on tau1, tau2 etc.. property of synapse.

        # neuron para
        self.tau = neuron_model['tau']
        self.Cb = neuron_model['Cb']
        self.Vth = neuron_model['Vth']
        self.Vrst = neuron_model['Vrst']
        self.RPS = neuron_model['RPS']

        # future parameters
        self.memV = 'memv'
        self.Iin_all = 'Iinall'
        self.res_spike_all = 'resspikeall'

    def run_input(self, input_spike):  # num_channels x num_timesteps matrix

        memV = np.zeros((self.Nres, len(self.t)))  # %perhaps will store all potentials at all times
        V = np.zeros((self.Nres, 1))
        Ibuffer = np.zeros((self.Nin + self.Nres, len(self.Iwave)))  # % why i wave?
        res_spikes = np.zeros((self.Nres, 1))
        inres_spikes = np.zeros((self.Nin + self.Nres, 1))
        res_spikes_all = np.zeros((self.Nres, len(self.t)))  # % contains all res spikes at all times
        Itotal = np.zeros((self.Nin + self.Nres, 1))
        Itotal_all = np.zeros((self.Nin + self.Nres, len(self.t)))
        RP = np.zeros((self.Nres, 1))  # %??
        Iin_all = np.zeros((self.Nres, len(self.t)))  # % ? input current depending on inspkes above??
        Iin = np.zeros((self.Nres, 1))

        Gnet = np.hstack([self.Gin, self.Gres])  # Nres x (nin + nres)
        for i in range(0, len(self.t) - 1):
            memV[:, i] = V  # Nres x num_timesteps <- Nresx1
            res_spikes_all[:, i] = res_spikes  # Nres x num_timesteps <- Nresx1
            inres_spikes = np.vstack([input_spike[:, i], res_spikes])  # Nin+Nres x1 <- Nin+Nres x1
            Itotal_all[:, i] = Itotal  # Nin+Nres x num_timesteps <- Nin+Nres x1
            Iin_all[:, i] = Iin  # Nres x num_timesteps <- Nresx1

            Ibuffer = np.roll(Ibuffer, -1, axis=1)
            Ibuffer[:, -1] = np.zeros((self.Nin + self.Nres, 1))
            Ibuffer = Ibuffer + np.multiply(np.tile(inres_spikes, (1, self.Iwave.shape[-1])),
                                            self.Iwave)  # nres+nin x iwave
            Itotal = Ibuffer[:, 0]
            Iin = np.dot(Gnet, Itotal)
            np.clip(Iin, 0, 10 ** (-9), out=Iin)

            RP -= 1
            np.clip(RP, 0, out=RP)
            V = V * (1 - (self.dt / self.tau)) + Iin * (self.dt / self.Cb)  # tau, Cb - neuron para
            V = np.where(RP > 0, 0, V)

            res_spikes = V > self.Vth  # from neuron
            V[res_spikes] = self.Vrst
            RP[res_spikes] = self.RPS

        self.Iin_all = Iin_all
        self.memV = memV
        self.res_spike_all = res_spikes_all

        return res_spikes_all  # reservoir_spike_pattern

    def plot_reservoir_char(self):
        if self.memV.isalpha() : return " reservoir not generated"

        plt.plot(self.t, self.memV[1, :])
        plt.xlabel("time")
        plt.ylabel(" memV for a neuron")
        plt.title("MemV vs time", loc='right')
        plt.show()

        plt.plot(self.t, self.Iin_all[1, :])
        plt.xlabel("time")
        plt.ylabel(" input currnet for a neuron")
        plt.title("Iin vs time", loc='right')
        plt.show()

        x, y = np.nonzero(self.res_spike_all)
        plt.plot(y * self.dt, x, 'ro')
        plt.ylim(1 - 0.1, self.Nres + 0.1)
        plt.xlabel("time")
        plt.ylabel(" bool(res neuron spiked) for each neuron")
        plt.title("times when each res neuron spiked", loc='right')
        plt.show()
        return


if __name__ == '__main__':
    from parameters import neuron, synapse, runtime, reservoir, gin, gres
    from input_spikes import spikes
    res_trail = reservoir(runtime(), reservoir(), gin(), gres(), neuron(), synapse())   #each of these return a dictionary/excpt gin gres
    res_trail.run_input(spikes())
    res_trail.plot_reservoir_char()
