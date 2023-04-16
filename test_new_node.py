# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 11:44:35 2023

@author: selin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:55:43 2022

@author: selin & Gubn
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:37:58 2022

@author: selin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
from PIDsim import PID # PID controller implementation

T1 = 85

# PID controller parameters
SP = 75 # set point
PV = T1 # measured process variable
MV = 0 # manipulated variable, valve 0...1, 0 -> only input 1, 1 -> only input 2

# PID controller setup
PID_controller = PID(Kp = 0.005, Ki = 0.00000, Kd=0, MVrange=(0.0,1.0), DirectAction = True)

# Ki too big -> oscillation
#PID_controller = PID(Kp = 0.01, Ki = 0.002, Kd=0, MVrange=(0.0,1.0), DirectAction = True)

# disturbance
T_dist = 50

L = 0.1
B = 0.1

i = 3
j = 3

l = L/i
bl = B/j
d = 1
              # m length of the cubic room
Spcm_updown = l*d           # m² surface of the pcm 
Spcm_leftright = bl*d

Tmelt = 55
n0 = np.array([0,1,2,3,4,5,6,7,8,9])
# Physical properties

# ===================

Water = {'Density': 997,                      # kg/m³
       'Specific heat': 4186}               # J/(kg·K)
# pd.DataFrame.from_dict(air, orient='index', columns=['air'])
Water = pd.DataFrame(Water, index=['Water'])

PCM = {'Conductivity': [2, 2, 2],  # W/(m·K)
        'Density': [1280, 1280, 1280],        # kg/m³
        'Specific_heat': [3000, 2100, 265000/4],  # J/(kg·K)
        'Width': [1, 1, 1],
        # 'Surface': [Spcm_b, Spcm_b, Spcm_b], # m²
        'Slices': [3, 3, 3]}                # number of  slices
PCM_Data = pd.DataFrame(PCM, index=['Liquid', 'Solid', 'Phasechange'])

h = 7750  # W/(m²⋅K)

# Conduction
#G_crl = PCM_Data.Conductivity['Solid'] / l * Spcm_l
G_cud = PCM_Data.Conductivity['Liquid'] / bl * Spcm_updown
G_crl = float(PCM_Data.Conductivity['Liquid'] / l * Spcm_leftright)

# Convection
Gw_ud = h*Spcm_updown
Gw_rl = h*Spcm_leftright
# Combindes Convection/Conduction
G1 = Gw_ud + G_cud*2


#Capacities
m = PCM_Data.Density['Liquid']*Spcm_leftright*bl
Cpcm_solid = m*PCM_Data.Specific_heat['Solid']
Cpcm_liquid = m*PCM_Data.Specific_heat['Liquid']
Cpcm_pc = m*PCM_Data.Specific_heat['Phasechange']

A = np.zeros([14, 10])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3,0], A[4,1], A[5,2], A[6,3] , A[7,4],A[8,3], A[9,4], A[10,5], A[11,6],A[12,8], A[12,7], A[13,2]= -1,-1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1
A[3,3], A[4,4], A[5,5], A[6,4],  A[8,6], A[9,7], A[10,8], A[12,8], A[11,7], A[7,5], A[13, 9]= 1,1,1, 1, 1, 1, 1, 1, 1, 1, 1

np.set_printoptions(suppress=False)
A_Data = pd.DataFrame(A)

G = np.array([Gw_rl, Gw_rl, Gw_rl, G1, G1, G1, G_crl, G_crl, G_cud, G_cud, G_cud, G_crl, G_crl, Gw_rl])
#G = np.array([Gw*2, Gwpcm, Gwpcm, G1, G1, G1, G_crl, G_crl, G_cud, G_cud, G_cud, G_crl, G_crl])
#G = np.array([Gw*2, Gw, Gw, G1, G1, G1, G_cud, G_cud,  G_cud,  G_cud, G_cud, G_cud, G_cud])
G= np.diag(G)
np.set_printoptions(precision=3, threshold=16, suppress=True)
pd.set_option("display.precision", 1)
G_Data = pd.DataFrame(G)

C_solid = np.array([0, 0, 0, Cpcm_solid, Cpcm_solid, Cpcm_solid, Cpcm_solid, Cpcm_solid, Cpcm_solid, 0])
C_solid = np.diag(C_solid)
# Uncomment next line to put 'Air' and 'Glass' capacities to zero 
# C = np.diag([0, C['Concrete'], 0, C['Insulation'], 0, 0, 0, 0])
pd.set_option("display.precision", 3)
C_solid_data = pd.DataFrame(C_solid)

C_liquid = np.array([0, 0, 0, Cpcm_liquid, Cpcm_liquid, Cpcm_liquid,  Cpcm_liquid, Cpcm_liquid, Cpcm_liquid, 0])
C_liquid = np.diag(C_liquid)
# Uncomment next line to put 'Air' and 'Glass' capacities to zero 
# C = np.diag([0, C['Concrete'], 0, C['Insulation'], 0, 0, 0, 0])
pd.set_option("display.precision", 3)
C_liquid_data = pd.DataFrame(C_liquid)

Cpcm_pc = np.array([0, 0, 0, Cpcm_pc, Cpcm_pc, Cpcm_pc, Cpcm_pc, Cpcm_pc, Cpcm_pc, 0])
Cpcm_pc = np.diag(Cpcm_pc)
# Uncomment next line to put 'Air' and 'Glass' capacities to zero 
# C = np.diag([0, C['Concrete'], 0, C['Insulation'], 0, 0, 0, 0])
pd.set_option("display.precision", 3)
C_pc_data = pd.DataFrame(Cpcm_pc)

b = np.zeros(14)  
b[[0]] = 1 
print(b)   # branches
f = np.zeros(10)         # nodes
y = np.zeros(10)
for i in range(0, len(n0)):         # nodes
    y[[n0]] = 1              # nodes (temperatures) of interest

[As_solid, Bs_solid, Cs_solid, Ds_solid] = dm4bem.tc2ss(A, G, b, C_solid, f, y)
print('As = \n', As_solid, '\n')
print('Bs = \n', Bs_solid, '\n')
print('Cs = \n', Cs_solid, '\n')
print('Ds = \n', Ds_solid, '\n')

[As_liquid, Bs_liquid, Cs_liquid, Ds_liquid] = dm4bem.tc2ss(A, G, b, C_liquid, f, y)

[As_pc, Bs_pc, Cs_pc, Ds_pc] = dm4bem.tc2ss(A, G, b, Cpcm_pc, f, y)

# steady state
b = np.zeros(14)        # temperature sources
b[[0]] = T1      # outdoor temperature

f = np.zeros(10)         # flow-rate sources

θ = np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print(f'θ = {θ} °C')

# State-space representation

bT = np.array([T1])     # [To, To, To, Tisp]
fQ = np.array([0])         # [Φo, Φi, Qa, Φa]
u = np.hstack([bT])
print(f'u = {u}')

yss = (-Cs_solid @ np.linalg.inv(As_solid) @ Bs_solid + Ds_solid) @ u
print(f'yss = {yss} °C')

λ = np.linalg.eig(As_solid)[0]    # eigenvalues of matrix As
print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
dtmax = min(-2. / λ)
print(f'Maximum time step: {dtmax:.2f} s = {dtmax / 60:.2f} min')

dt = 12    # seconds
print(f'dt = {dt} s = {dt / 60:.0f} min')

t_resp = 4 * max(-1 / λ)
print('Time constants: \n', -1 / λ, 's \n')
print(f'Settling time: {t_resp:.0f} s = {t_resp / 60:.1f} min \
= {t_resp / (3600):.2f} h = {t_resp / (3600 * 24):.2f} days')

duration = 126000        # seconds, larger than response time
n = int(np.floor(duration / dt))    # number of time steps
t = np.arange(0, n * dt, dt)        # time vector for n time steps

print(f'Duration = {duration} s')
print(f'Number of time steps = {n}')
pd.DataFrame(t, columns=['time'])

# initialize input temperature
T_mix = T1
u = np.zeros([1, n])      # Tisp = 20 for n time steps
u[:, 0] = T_mix

n_s = As_solid.shape[0]                      # number of state variables
θ_exp = 20*np.ones([n_s, t.shape[0]])    # explicit Euler in time t
θ_imp = 20*np.ones([n_s, t.shape[0]])    # implicit Euler in time t

# initialize output matrices
y_exp = np.zeros([len(n0), t.shape[0]])
y_imp = np.zeros([len(n0), t.shape[0]])
y_exp[:, 0] = Cs_solid @ θ_exp[:, 0] + Ds_solid @  u[:, 0]
y_imp[:, 0] = Cs_solid @ θ_imp[:, 0] + Ds_solid @  u[:, 0]

# initialize PID controller
MV = PID_controller.update(t[0], SP, PV, MV)

Q = np.zeros([14,10500])

I = np.eye(n_s)                        # identity matrix
p = 8 # last PCM node
for k in range(n - 1):
    if θ_exp[p-3, k] < Tmelt-4:
        θ_exp[:, k + 1] = (I + dt * As_solid) @\
            θ_exp[:, k] + dt * Bs_solid @ u[:, k]
        θ_imp[:, k + 1] = np.linalg.inv(I - dt * As_solid) @\
            (θ_imp[:, k] + dt * Bs_solid @ u[:, k])
    if θ_exp[p-3, k] > Tmelt+4:
        θ_exp[:, k + 1] = (I + dt * As_liquid) @\
            θ_exp[:, k] + dt * Bs_liquid @ u[:, k]
        θ_imp[:, k + 1] = np.linalg.inv(I - dt * As_liquid) @\
            (θ_imp[:, k] + dt * Bs_liquid @ u[:, k])
    else: 
        θ_exp[:, k + 1] = (I + dt * As_pc) @\
            θ_exp[:, k] + dt * Bs_pc @ u[:, k]
        θ_imp[:, k + 1] = np.linalg.inv(I - dt * As_pc) @\
            (θ_imp[:, k] + dt * Bs_pc @ u[:, k])
    
    y_exp[:, k + 1] = Cs_solid @ θ_exp[:, k + 1] + Ds_solid @  u[:, k]
    y_imp[:, k + 1] = Cs_solid @ θ_imp[:, k + 1] + Ds_solid @  u[:, k]
    
    # extract T2_a
    T2_a = y_exp[2, k + 1]
    
    # compute T2_b
    #T2_b = T2_a - T_dist
    T2_b = 40
    
    # compute mix temperature (valve)
    T_mix = (1 - MV) * T1 + MV * T2_b
    u[:, k + 1] = T_mix
    PV = T_mix
    
    # update PID controller
    MV = PID_controller.update(t[k + 1], SP, PV, MV)
    Q[:, k] =  G @ (-A @  y_exp[:,k+1] + np.transpose(b))
    
#y_exp_2 = Cs_solid @ θ_exp + Ds_solid @  u
#y_imp_2 = Cs_solid @ θ_imp + Ds_solid @  u

# PID controller plots
PID_controller.plot()

fig, ax = plt.subplots()
ax.plot(t / 3600, y_exp.T)
ax.set(xlabel='Time [h]',
       ylabel='$T_i$ [°C]',
       title='Step input: To')
ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.show()
plt.grid()

################### Charging Power ##################################################
f_0 = 85* np.ones([len(y_exp), 1])
#z = y_exp[0, :]-y_exp[3,:]*G1
#Q = (y_exp[3, :]-y_exp[6, :]+y_exp[4, :]-y_exp[7, :]+y_exp[5, :]-y_exp[8, :])*G_cud/1000
#Q =  G @ (-A @  y_exp + b)
#Q = (y_exp[0, :]-y_exp[1, :] + y_exp[1, :]-y_exp[2, :])*Gw/1000
#Q = (y_exp[0, :]-y_exp[1, :])*G0
#Q = (T1-y_exp[0, :])*G1
fig, ax = plt.subplots()
ax.plot(t / 3600, Q)
plt.grid()
Integral = np.trapz(Q, t)