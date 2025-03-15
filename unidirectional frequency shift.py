# -*- coding: utf-8 -*-
"""
Created on 2023/1/30
@author: Monica Xu
time dependent hamiltonion to for magnon achieve bloch oscillation.
for i in range(n_max-1):
    hami[i,i+1] = g * exp( i* (omega_{n+1} - omega_{n} - Omega) t )
    hami[i+1,i] = g * exp( -i* (omega_{n+1} - omega_{n} - Omega) t )
  
能级 omega_n = omega0 + (n * n) * Omega_prime ,n>=0 这里omega0是n=0的能级
magnon含时间布洛赫振荡哈密顿量；初始波包人为用高斯函数构造。

所有频率使用模拟值                                                                                                                                              。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from qutip import *
from qutip import basis
import sys
import os

# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def initialization():
    global omega_n
    f = open(r"./raw_data/frequency.txt")
    line = f.readline()
    data_order = []
    while line:
        a = list(map(float, line.split(' ')))
        b = a[0]  # i.e. a[3:4] is to copy the fourth line
        data_order.append(b)
        line = f.readline()
    f.close()
    data_order = np.array(data_order)
    omega_n = data_order


def eigenstates(H):
    _, states = H.eigenstates()
    return states

def prepare_gaussian_initial_state(H, x0=50, k0=0.1, sigma=0.01):
    states = eigenstates(H)
    N = states[0].shape[0]
    x = np.arange(0, N)
    profile = np.exp(-(((x)-x0)/(N*sigma))**2 + 1j * k0 * (x-x0))
    profile /= np.linalg.norm(profile)
    return Qobj(profile)


def project(states, psi):
    return(np.array([(abs(psi[:])**2)[i, 0] for i in range(psi.shape[0])])) 


def _ket(n, N):
    return basis(N, n)

def _bra(n, N):
    return basis(N, n).dag()

def period(time, kappa, Omega, g):
    g_fuc = g
    fuc = Omega
    if  time < wait:
        g_fuc = 0
        fuc = 0
    elif time>=wait :
        fuc = kappa * (time-wait)
        fuc = Omega + fuc
    return g_fuc, fuc 


def hamiltonian(Omega, g, N, kappa):
    initialization()

    def make_Hn01_args(n):
        def Hn01_args(time, args):
            g_fuc,fuc = period(time,kappa,Omega, g)
            return g_fuc *np.exp(1j * (omega_n[n+1]*(2*np.pi) - omega_n[n]*(2*np.pi) - fuc) * (time-wait))    # nearest neighbor hopping
        return Hn01_args

    def make_Hn10_args(n):
        def Hn10_args(time, args):
            g_fuc,fuc = period(time,kappa,Omega, g)
            return g_fuc *np.exp( -1j * (omega_n[n]*(2*np.pi) - omega_n[n-1]*(2*np.pi) - fuc) * (time-wait))    # nearest neighbor hopping
        return Hn10_args

    H_array=[ ]
    for n in range(0, N-1):
        Hn01_args = make_Hn01_args(n)
        Hn01 = 1 *_ket(n, N)*_bra(n + 1, N)
        Hn01 = [Hn01, Hn01_args]
        H_array.append(Hn01)
  
    for n in range(1, N): 
        Hn10_args = make_Hn10_args(n)
        Hn10 = 1 *_ket(n, N)*_bra(n - 1, N)
        Hn10 = [Hn10, Hn10_args]
        H_array.append(Hn10)
    return H_array
# ========above is system===================

params = dict(
    sigma = 0.01,
    n = 15,
    t_end = 950,
    n_times = 951,
)  #950,951
# ========above is parameters===================
def lz_grid_time_evolution(kappa0, sigma=0.01, t_end=1, n=10, n_times=201):
    global wait
    wait = 0
    k0 = 0 #14 -1,-8,-14,-20
    deltaK = 0.3
    g = -0.0429*deltaK          # hopping 
    Omega = 0.135*(2*np.pi)     # driving frequency: Omega = 0.135*(2*np.pi)
    kappa = kappa0 *(2*np.pi)*10**(-6)          # Omega(t) = Omega + kappa * t        
    N = 2*n + 1
    t_list = np.linspace(0, t_end, n_times)

    # compute Hamiltonian
    H = hamiltonian(Omega, g, N, kappa)
    Hfun = QobjEvo(H)

    # prepare initial state
    psi0 = prepare_gaussian_initial_state(Hfun(0.0), 20, k0, sigma)

    # run time evolution
    global i, data1
    i = 0
    data1 = np.zeros((len(t_list), N))
    def e_ops(time, psi):
        global i, data1

        print('time', time)
        states = eigenstates(Hfun(time))    # diagonalize time dependent H(time)
        data1[i, :] = project(states, psi)
        i += 1
    options = Options(nsteps=100)
    sesolve(H, psi0, t_list, e_ops=e_ops, options=options)
      
 
def plot_style(xlabel='x', ylabel='y'):
    size =10
    font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 10}
    plt.rc('font', **font)
    plt.xlabel(xlabel, font={'family':'Times New Roman', 'size':18}) #
    plt.ylabel(ylabel, font={'family':'Times New Roman', 'size':18})
    plt.xticks(size=size)
    plt.yticks(size=size)

##===============================================================================  
lz_grid_time_evolution(kappa0=25, **params)
''''if kappa0 = -25, the system will be driven to higher resonant mode.
    if kappa0 = 0, then Omega(t) is a constant. When Omega = 0.135*(2*np.pi), 
                    Bloch oscillation can be achieved.'''
N =dimy=(params['n']*2+1)
t_end = params['t_end']
n_times = params['n_times']
ns = np.arange(0,N,1)

fig, ax = plt.subplots(figsize=(1.7, 3.8), dpi=300)
plot_style(xlabel='frequency mode', ylabel='Time (ns)')
x1_list = np.linspace(50, t_end+50, n_times)
im = ax.pcolormesh(ns[10:31], x1_list, data1[:, 10:31],cmap=plt.cm.magma_r,alpha=0.7,shading='auto') # shading='gouraud'

cbar_ax = fig.add_axes([0.95, 0.125, 0.04, 0.755])  # x, y, width, height (0-1 scale)
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=0.5, pad=2 )
plt.show()