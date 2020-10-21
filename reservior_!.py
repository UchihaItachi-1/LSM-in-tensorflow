# 19.10.20
#lien 25 rng,
import numpy as np
import tensorflow as tf
print(tf.version)


class reservoir:
    def __init__(self , runtime_para, res_para, Gin, Gres, neuron_model, synapse_model):
        # dic of runtime and res para , Gin,Gres tenors .. neuron and synapse models have more parameters
        # self.synapse_model = synapse_model
        # self.neuron_model = neuron_model
        self.Gres = Gres
        self.Gin = Gin

        #runtime para
        self.dt = runtime_para['dt']
        self.T = runtime_para['T']
        self.t = np.arange(0 , self.T, self.dt)

        #res para
        self.Nin =  res_para['Nin']  #5;       %?   no of input neurons
        self.Nres = res_para['Nres']  # 10;     %?
        ##rng(4);        %???? ask
        self.pinp = res_para['pinp']   #0.2;    %?
        self.pres = res_para['pres']     #0.5; %? some sort of thershold on revervoir weights- why? - perhaps to make them 0and 1 like gin?(-1,1)
        self.pinh = res_para['pinh']   #0.2;    %?  decides what fraction of res neurons are inhibitory
        self.fout = res_para['fout']   #4;      %? no of res neurons each input neuron is connected to
        self.pinhi = res_para['pinhi']   #0.25;  %? perhaps each ip neuron has this much inhibitory connection and 1-this exhibitory

        #synapse para
        self.Iwave = synapse_model['Iwave'] #rest dont seem imp


    def run_input(self, input_spike):
        reservoir_spike_pattern = 0
        memV = np.zeros((self.Nres, len(self.t)))  # %perhaps will store all potentials at all times
        V = np.zeros((self.Nres, 1))
        Ibuffer = np.zeros((self.Nin + self.Nres, len(self.Iwave)))  # % why i wave?
        res_spikes = np.zeros((self.Nres, 1))
        inres_spikes = np.zeros((self.Nin + self.Nres, 1))
        res_spikes_all = np.zeros((self.Nres, len(self.t)))  # % contains all res spikes at all times
        Itotal = np.zeros((self.Nin + self.Nres, 1))
        Itotal_all = np.zeros((self.Nin + self.Nres, len(self.t)))
        RP = np.zeros((self.Nres, 1))  # %??
        Iin_all = np.zeros((self.Nres, len(self.t)))  # % ? input current depending oninspkes above??
        Iin = np.zeros((self.Nres, 1))

        Gnet = np.hstack([self.Gin, self.Gres])
        for i in range(0, len(self.t)-1):
            memV[:, i] = V
            


        return reservoir_spike_pattern


if __name__ == '__main__':
    0



