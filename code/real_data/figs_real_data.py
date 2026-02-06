#!/usr/bin/env python
# coding: utf-8
# This is code for generating the Figure3 in the real data analysis section of the paper. 

import numpy as np
import pandas as pd
import os

from scipy.interpolate import BSpline
from scipy.linalg import lstsq

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import GPy
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import time

time_00 = time.time()

# This dataset should be accessed through MIMIC-III database. We cannot share the data due to privacy issues. 
# The data description is in data/README.md
df = pd.read_csv('')
file1 = open('state_features.txt', 'r')

Lines = file1.readlines()
state_names = [line.strip() for line in Lines]
state_names.remove('re_admission')

state_names_cont = state_names.copy()
state_names_cont.remove('gender')
state_names_cont.remove('mechvent')
state_names_cont.remove('SIRS')

df_state = df[state_names_cont]
df_action = df[['iv_input','vaso_input']]
df_reward = -df[['SOFA']]

time_id = np.array(df['bloc'])
id1 = np.where((time_id == 4)==True)[0] - 3
id2 = np.where((time_id == 4)==True)[0] - 2
id3 = np.where((time_id == 4)==True)[0] - 1
id4 = np.where((time_id == 4)==True)[0] 

s1 = df_state.to_numpy()[id1,:]
s2 = df_state.to_numpy()[id2,:]
s3 = df_state.to_numpy()[id3,:]
s4 = df_state.to_numpy()[id4,:]
a1 = df_action.to_numpy()[id1,:]
a2 = df_action.to_numpy()[id2,:]
a3 = df_action.to_numpy()[id3,:]

dim_s = len(state_names_cont)
dim_a = 2

s1 = minmax_scale(s1, feature_range=(0,1), axis=0)
s2 = minmax_scale(s2, feature_range=(0,1), axis=0)
s3 = minmax_scale(s3, feature_range=(0,1), axis=0)
s4 = minmax_scale(s4, feature_range=(0,1), axis=0)


k_degree= 3
m_interior= 3
knots= np.concatenate( (np.zeros(k_degree), np.linspace(0,1, m_interior+2), np.ones(k_degree)) )
L = k_degree + m_interior + 1

def generate_design_matrix(sh, ah=None):
    # sh.shape is (m, sh_length)
    design_matrix_s = np.concatenate( [ np.array([BSpline(t=knots, c=np.eye(1,L,i)[0], k=k_degree)(sh[:,j])
                                                for i in range(L)]).T
                                     for j in np.arange(sh.shape[1]) ], axis=1 )
    # shape is (m, L*sh_length)
    if ah is not None:
        design_matrix_a = np.concatenate( [ 1*np.array([ah[:,j]==i for i in range(5)]).T 
                                           for j in range(ah.shape[1]) ], axis=1)
        # shape is (m, 5*ah_length)
        design_matrix = np.concatenate( (design_matrix_s, design_matrix_a), axis=1)
    else:
        design_matrix = design_matrix_s
    design_matrix = np.maximum(design_matrix, np.zeros(design_matrix.shape))
    # it is supposed to be nonnegative, but it may contain negative values with very small absolute value due to computation error
    return design_matrix


def get_policy_prob(k, theta, h, greedy = False):
    # h has m rows, return a matrix of shape (m,25)
    # theta has 25 columns
    # h.shape[1] = k*dim_s + (k-1)*2
    # theta.shape[0] = b.shape[1] = L*sh_length + 5*ah_length = L*k*dim_s + 5*(k-1)*2
    m = h.shape[0]
    
    if k==1:
        s_history = h
        b = generate_design_matrix(s_history)
    
    if k==2:
        # h=(s1,a1,s2), columns: dim_s + 2 + dim_s
        a_history = h[:, [dim_s,dim_s+1]]
        s_history = np.delete(h, [dim_s,dim_s+1], axis=1)
        b = generate_design_matrix(s_history, a_history)
    
    if k==3:
        # h=(s1,a1,s2,a2,s3), columns: dim_s + 2 + dim_s + 2 + dim_s
        a_history = h[:, [dim_s,dim_s+1, 2*dim_s+2, 2*dim_s+3]]
        s_history = np.delete(h, [dim_s,dim_s+1, 2*dim_s+2, 2*dim_s+3], axis=1)
        b = generate_design_matrix(s_history, a_history)
        
    #b = np.maximum(b, np.zeros(b.shape))
    # f: Q 
    f = (b+ 1e-100) @ theta   # shape is (m,25)
    if greedy == False:
        exp_f = np.exp( f- np.max(f, axis=1)[:,np.newaxis] )
        prob = exp_f / np.sum(exp_f, axis=1)[:,np.newaxis]   # shape is (m,25)
    if greedy == True:
        prob = 1*( np.arange(25) == np.argmax(f, axis=1)[:,np.newaxis] )
    return prob


h1 = s1
h2 = np.concatenate((s1,a1,s2),axis=1)
h3 = np.concatenate((h2,a2,s3),axis=1)
c_list = np.array([0, 1, 5, 10, 20, 50, 100])
action_list = np.zeros((50, 3, 25))   # 50seed, 3k. fix: c=5, t=20
cwd = os.getcwd()

cc=20
print('c={}:'.format(cc))
c_index = np.argmax(c_list == cc)
t=20
print('t={}:'.format(t))
for seed_index in range(50):
    job_index = seed_index*7 + c_index
    theta1 = np.load( os.path.join(cwd+'', str(int(job_index)).zfill(3)+'_theta1_t'+str(int(t)).zfill(2)+''+'.npy') )
    theta2 = np.load( os.path.join(cwd+'', str(int(job_index)).zfill(3)+'_theta2_t'+str(int(t)).zfill(2)+''+'.npy') )
    theta3 = np.load( os.path.join(cwd+'', str(int(job_index)).zfill(3)+'_theta3_t'+str(int(t)).zfill(2)+''+'.npy') )
    action_list[seed_index, 0,:] = np.sum( get_policy_prob(1, theta1, h1) , axis=0)
    action_list[seed_index, 1,:] = np.sum( get_policy_prob(2, theta2, h2) , axis=0)
    action_list[seed_index, 2,:] = np.sum( get_policy_prob(3, theta3, h3) , axis=0)

print('k=1')
c5_t20_action_stage1 = np.mean( action_list[:,0,:] , axis=0).reshape(5,5)
print( c5_t20_action_stage1 )
np.save('', c5_t20_action_stage1)

print('k=2')
c5_t20_action_stage2 = np.mean( action_list[:,1,:] , axis=0).reshape(5,5)
print( c5_t20_action_stage2 )
np.save('', c5_t20_action_stage2)

print('k=3')
c5_t20_action_stage3 = np.mean( action_list[:,2,:] , axis=0).reshape(5,5)
print( c5_t20_action_stage3 )
np.save('', c5_t20_action_stage3)
print('\n')


# DTR-Q Action
qlearn_action_list = np.zeros((50,3,25))  
print('q learning:')
for seed_index in range(50):
    job_index = seed_index*4 + c_index
    theta1 = np.load( os.path.join(cwd+'/save', str(int(job_index)).zfill(3)+'_theta1_q'+''+'.npy') )
    theta2 = np.load( os.path.join(cwd+'/save', str(int(job_index)).zfill(3)+'_theta2_q'+''+'.npy') )
    theta3 = np.load( os.path.join(cwd+'/save', str(int(job_index)).zfill(3)+'_theta3_q'+''+'.npy') )
    qlearn_action_list[seed_index, 0,:] = np.sum( get_policy_prob(1, theta1, h1, greedy=True) , axis=0)
    qlearn_action_list[seed_index, 1,:] = np.sum( get_policy_prob(2, theta2, h2, greedy=True) , axis=0)
    qlearn_action_list[seed_index, 2,:] = np.sum( get_policy_prob(3, theta3, h3, greedy=True) , axis=0)

print('k=1')
qlearn_action_stage1 = np.mean( qlearn_action_list[:,0,:] , axis=0).reshape(5,5)
print( qlearn_action_stage1 )
np.save('', qlearn_action_stage1)

print('k=2')
qlearn_action_stage2 = np.mean( qlearn_action_list[:,1,:] , axis=0).reshape(5,5)
print( qlearn_action_stage2 )
np.save('', qlearn_action_stage2)

print('k=3')
qlearn_action_stage3 = np.mean( qlearn_action_list[:,2,:] , axis=0).reshape(5,5)
print( qlearn_action_stage3 )
np.save('', qlearn_action_stage3)
print('\n')

# Physician Action 
print('physician:')
a1_to25 = a1[:,0]*5+a1[:,1]
a2_to25 = a2[:,0]*5+a2[:,1]
a3_to25 = a3[:,0]*5+a3[:,1]

print('k=1')
physician_action_stage1 = np.sum( np.arange(25) == a1_to25[:,np.newaxis] , axis=0).reshape(5,5)
print( physician_action_stage1 )
np.save('', physician_action_stage1)

print('k=2')
physician_action_stage2 = np.sum( np.arange(25) == a2_to25[:,np.newaxis] , axis=0).reshape(5,5)
print( physician_action_stage2 )
np.save('', physician_action_stage2)

print('k=3')
physician_action_stage3 = np.sum( np.arange(25) == a3_to25[:,np.newaxis] , axis=0).reshape(5,5)
print( physician_action_stage3 )
np.save('', physician_action_stage3)
print('\n')
                
time_01 = time.time()
print('Finished. Total time of the job: {:.3f} minutes'.format( (time_01-time_00)/60 ) )
print('finished')



### Plot

value_list = np.zeros((50, 5, 21))
qlearn_list = np.zeros(50)
cwd = os.getcwd()


selected_c_indices = [0, 2, 3, 4, 5]

for seed_index in range(50):
    for new_c_index, original_c_index in enumerate(selected_c_indices):
        job_index = seed_index * 7 + original_c_index  
        filepath = os.path.join(cwd, '', f"{job_index:03d}.npy")
        value_list[seed_index, new_c_index, :] = np.load(filepath)

for seed_index in range(50):
    for c_index_q in range(4):
        job_index = seed_index * 4 + c_index_q
        qlearn_list[seed_index] = np.load(
        os.path.join(cwd, '', str(int(job_index)).zfill(3) + '_OPE_qlearn' + '' + '.npy')
    )

c_list = np.array([0, 5, 10, 20, 50])
qlearn_average = np.mean(qlearn_list)

plt.figure(facecolor='w')
for jc, cc in enumerate(c_list):
    plt.plot(range(21), np.mean(value_list[:, jc, :], axis=0), label=f'c={cc}')

plt.axhline(y=qlearn_average, linestyle='--', color='tab:purple', label='DTR-Q')

    

plt.xlabel('iterations')
plt.ylabel('policy value')
plt.xticks([0, 5, 10, 15, 20])
plt.ylim(2.0, 2.4)
plt.legend()
plt.savefig('', dpi=200)

print('finished')





