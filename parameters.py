# 10.11.20
# contains functions that build parameters
from math import ceil
import numpy as np
from matplotlib import pyplot as plt


def runtime():
    run = {}
    run['dt'] = 0.1e-6
    run['T'] = 1e-3
    run['t'] = np.linspace(0, 10e-3, num=1000, endpoint= False)
    return {}

def neuron():
    run = runtime()
    dt = run['dt']
    neur_par = {'Vth': 200e-3, 'Imin': 3e-9, 'Iat100kHz': 7e-9}
    neur_par['taubyC'] = neur_par['Vth'] / neur_par['Imin']
    neur_par['tau'] = 18e-6  # % 1 / 100e3 / log(Iat100kHz / (Iat100kHz - Imin))
    neur_par['Cb'] = 270e-15  # % tau / taubyC
    neur_par['Vrst'] = 0
    neur_par['RefPer'] = 0.1e-6
    neur_par['RPS'] = ceil(neur_par['RefPer'] / dt)
    return neur_par

def synapse():
    run = runtime()
    dt = run['dt']

    tau1 = 1e-6
    tau2 = 0.5e-6
    I0 = 30e-9
    ds = 0.1e-6
    DS = ceil(ds / dt)
    ts = np.arange(0, (ds + 6*tau1 + tau2), dt)
    print('len ts 67', len(ts))
    Iwave = I0 * (np.exp(-(ts - ds) / tau1) - np.exp(-(ts - ds) / tau2)) # % ???
    Iwave[1: ceil(ds / dt)] = 0

    syn = {}
    syn['Iwave']= Iwave
    # figure(3);
    # plot(ts, Iwave);
    return syn

def reservoir():
    res = {}
    res['Nin'] = 5
    res['Nres'] = 10
    res['pinp'] = 0.2
    res['pres'] = 0.5
    res['pinh'] = 0.2
    res['fout'] = 4
    res['pinhi'] = 0.25

    return res

def gin():

    network = reservoir()
    Nres =  network['Nres']
    Nin = network['Nin']
    fout = network['fout']
    pinhi= network['pinhi']

    Gin = np.zeros((Nres, Nin)) #; % arr
    for i in range(Nin):
        indices = np.random.randint(0, Nres, fout)  #% indices = randperm(Nres, fout); % mychange
        weights =  np.vstack((np.ones((int(fout * pinhi), 1)),np.ones((int(fout * (1 - pinhi)), 1)) ))  #np.vstack
        for ind, j in enumerate(indices):
            Gin[j][i] = weights[ind]
    # % for each column take at random 4 rows without repition and do something
    # % something - an
    # matrix
    # of - 1
    # s
    # followed
    # by
    # ones, perhaps
    # inhibitory and exhibtory
    return Gin

def plot_gin(Gin):
    x, y = np.nonzero(Gin)
    plt.plot(y, x, 'ro')              #find a way to colour code -> given arrays: i,j arr[i,j] = 1 or -1, colourcode and plot acccordingly
    plt.ylim(0 - 0.2, len(Gin)-1+ 0.2)
    plt.xlabel("Nin")
    plt.ylabel("Nres connected with each Nin")
    plt.title("[i][j] = dot means i connected to j in reservoir", loc='right')
    plt.show()
    return


input_mat = gin()
plot_gin(input_mat)