#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import GPy
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import time
from tqdm import trange
from code.POLAR import *


param_seedtenth_index = sys.argv.index('--seedtenth')
param_seedunit_index = sys.argv.index('--seedunit')
param_c_index = sys.argv.index('--c')
seed_tenth = int(sys.argv[param_seedtenth_index + 1])
seed_unit = int(sys.argv[param_seedunit_index + 1])
seed = int(seed_tenth*10 + seed_unit)
c = float(sys.argv[param_c_index + 1])
### seed, c are given by the shell script (p_split=0.8, eta=20, n_MC=100, m=1000, T=20)
print('seed={}, c={}'.format(seed, c))


time_00 = time.time()

# create state and action 

df = pd.read_csv('sepsis_processed_state_action.csv')
file1 = open('data/state_features.txt', 'r')
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
r1 = df_reward.to_numpy().flatten()[id1]
r2 = df_reward.to_numpy().flatten()[id2]
r3 = df_reward.to_numpy().flatten()[id3]

dim_s = len(state_names_cont)
dim_a = 2

s1 = minmax_scale(s1, feature_range=(0,1), axis=0)
s2 = minmax_scale(s2, feature_range=(0,1), axis=0)
s3 = minmax_scale(s3, feature_range=(0,1), axis=0)
s4 = minmax_scale(s4, feature_range=(0,1), axis=0)
r1 = minmax_scale(r1)
r2 = minmax_scale(r2)
r3 = minmax_scale(r3)


k_degree= 3
m_interior= 3
knots= np.concatenate( (np.zeros(k_degree), np.linspace(0,1, m_interior+2), np.ones(k_degree)) )
L = k_degree + m_interior + 1



def estimate_P(s1,a1,s2,a2,s3,a3,s4, r1, r2, r3, k, kernel_1, kernel_2, kernel_3):
    h1 = s1
    h2 = np.concatenate((s1,a1,s2),axis=1)
    h3 = np.concatenate((h2,a2,s3),axis=1)
    Z1 = KMeans(n_clusters=k).fit( np.concatenate((h1,a1),axis=1) ).cluster_centers_
    Z2 = KMeans(n_clusters=k).fit( np.concatenate((h2,a2),axis=1) ).cluster_centers_
    Z3 = KMeans(n_clusters=k).fit( np.concatenate((h3,a3),axis=1) ).cluster_centers_
    P1_hat = GPy.models.SparseGPRegression(np.concatenate((h1,a1),axis=1), s2, Z=Z1, kernel = kernel_1)
    P2_hat = GPy.models.SparseGPRegression(np.concatenate((h2,a2),axis=1), s3, Z=Z2, kernel = kernel_2)
    P3_hat = GPy.models.SparseGPRegression(np.concatenate((h3,a3),axis=1), s4, Z=Z3, kernel = kernel_3)
    
    
    P1_hat_r = GPy.models.SparseGPRegression(np.concatenate((h1,a1),axis=1), r1.reshape(-1, 1), Z=Z1, kernel = kernel_1)
    P2_hat_r = GPy.models.SparseGPRegression(np.concatenate((h2,a2),axis=1), r2.reshape(-1, 1), Z=Z2, kernel = kernel_2)
    P3_hat_r = GPy.models.SparseGPRegression(np.concatenate((h3,a3),axis=1), r3.reshape(-1, 1), Z=Z3, kernel = kernel_3)
    
    return [P1_hat,P2_hat,P3_hat, P1_hat_r, P2_hat_r, P3_hat_r]
 



def r_tilde(k,h,a, P1_hat,P2_hat,P3_hat, P1_hat_r,P2_hat_r,P3_hat_r,c_pessimism):
    # consider m samples of (h,a), return an array of shape (m,)
    if k==1:
        r1_hat = P1_hat_r.predict_noiseless( np.concatenate((h,a),axis=1) )[0]
        modified_reward = r1_hat -c_pessimism * P1_hat.predict_noiseless( np.concatenate((h,a),axis=1) )[1]
        
    if k==2:
        r2_hat = P2_hat_r.predict_noiseless( np.concatenate((h,a),axis=1) )[0]
        modified_reward = r2_hat -c_pessimism * P2_hat.predict_noiseless( np.concatenate((h,a),axis=1) )[1]
        
        
    if k==3:
        # r_hat = P3_hat.predict_noiseless( np.concatenate((h,a),axis=1) )[0]
        # modified_reward = r_hat -c_pessimism * P3_hat.predict_noiseless( np.concatenate((h,a),axis=1) )[1]
        r3_hat = P3_hat_r.predict_noiseless( np.concatenate((h,a),axis=1) )[0]
        modified_reward = r3_hat -c_pessimism * P3_hat.predict_noiseless( np.concatenate((h,a),axis=1) )[1]
        
    return modified_reward.reshape(-1,)



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
    f = (b+ 1e-100) @ theta   # shape is (m,25)
    if greedy == False:
        exp_f = np.exp( f- np.max(f, axis=1)[:,np.newaxis] )
        prob = exp_f / np.sum(exp_f, axis=1)[:,np.newaxis]   # shape is (m,25)
    if greedy == True:
        prob = 1*( np.arange(25) == np.argmax(f, axis=1)[:,np.newaxis] )
    return prob

def get_action(k, theta, h, greedy = False):
    # h has m rows, return an array of shape (m,)
    # theta has 25 columns
    # h.shape[1] = k*dim_s + (k-1)*2
    # theta.shape[0] = L*sh_length + 5*ah_length = L*k*dim_s + 5*(k-1)*2
    m = h.shape[0]
    prob = get_policy_prob(k, theta, h, greedy)
     
    u = np.random.uniform(0,1,size=m)
    a = np.argmax( np.cumsum(prob, axis=1) > u.reshape(m,1) , axis = 1)   # range from 0 to 24, (m,)
    #if greedy == True:
        #a = np.argmax(prob, axis=1)   # range from 0 to 24, shape is (m,)
        
    return np.array([a//5, a%5]).T   # shape is (m,2)




def compute_Q_function(k,h,a, P1, P2, theta_2, theta_3, r_function, n_MC):
    # P1, P2 are fitted GPy.models.SparseGPRegression
    # compute Q function for m samples of (h,a), return an array of shape (m,)
    # r_function takes value of (k,h,a) and return an (m,) array
    m = h.shape[0]
    if k==3:
        return r_function(k,h,a)
    if k==2:
        h2,a2 = h,a
        r2_tilde = r_function(2,h2,a2)   # (m,)
        # for each i, we need n_MC trajectory samples started from (h[i],a[i])
        h2_sample = np.repeat(h2, n_MC, axis=0)   # m*n_MC rows
        a2_sample = np.repeat(a2, n_MC, axis=0)
        s3_sample = P2.predict_noiseless( np.concatenate((h2_sample, a2_sample), axis=1) )[0]
        h3_sample = np.concatenate( (h2_sample, a2_sample, s3_sample) , axis=1)
        a3_sample = get_action(3, theta_3, h3_sample)
        r3_sample = r_function(3, h3_sample, a3_sample)   # (m*n_MC,)
        Q = r2_tilde + np.mean( r3_sample.reshape(m,n_MC) , axis=1)   # (m,)
        return Q
    if k==1:
        h1,a1 = h,a
        r1_tilde= r_function(1,h1,a1)   # (m,)
        # for each i, we need n_MC trajectory samples started from (h[i],a[i])
        h1_sample = np.repeat(h1, n_MC, axis=0)   # m*n_MC rows
        a1_sample = np.repeat(a1, n_MC, axis=0)
        s2_sample = P1.predict_noiseless( np.concatenate((h1_sample, a1_sample), axis=1) )[0]
        h2_sample = np.concatenate( (h1_sample, a1_sample, s2_sample) , axis=1) 
        a2_sample = get_action(2, theta_2, h2_sample)
        Q2_sample = compute_Q_function(2, h2_sample, a2_sample, P1,P2, theta_2, theta_3, r_function, n_MC=1)
        Q = r1_tilde + np.mean( Q2_sample.reshape(m,n_MC) , axis=1)   # (m,)
        return Q


def generate_sample_for_regression(m):
    s1_list= np.random.uniform(0,1, size=(m,dim_s))
    s2_list= np.random.uniform(0,1, size=(m,dim_s))
    s3_list= np.random.uniform(0,1, size=(m,dim_s))
    a1_list= np.random.choice(range(5), size=(m,dim_a))
    a2_list= np.random.choice(range(5), size=(m,dim_a))
    sh2_list = np.concatenate( (s1_list, s2_list), axis=1 )   # 'state history'
    sh3_list = np.concatenate( (s1_list, s2_list, s3_list), axis=1 )
    ah2_list = np.concatenate( (a1_list, a2_list), axis=1 )
    
    design_matrix_1 = generate_design_matrix(s1_list)
    design_matrix_2 = generate_design_matrix(sh2_list, a1_list)
    design_matrix_3 = generate_design_matrix(sh3_list, ah2_list)
    
    h1_list = s1_list
    h2_list = np.concatenate( (s1_list, a1_list, s2_list), axis=1 )
    h3_list = np.concatenate( (s1_list, a1_list, s2_list, a2_list, s3_list), axis=1 )
    
    return [h1_list, h2_list, h3_list, design_matrix_1, design_matrix_2, design_matrix_3]



def one_iteration_real_data(theta_1, theta_2, theta_3,
                            s1, a1, s2, a2, s3, a3,
                            P1_hat, P2_hat,
                            eta, r_tilde,
                            n_MC=10):

    m = s1.shape[0]

    design_matrix_1 = generate_design_matrix(s1) 
    design_matrix_2 = generate_design_matrix(np.concatenate((s1, s2), axis=1), a1)
    design_matrix_3 = generate_design_matrix(np.concatenate((s1, s2, s3), axis=1),
                                              np.concatenate((a1, a2), axis=1))

    h1_list = s1
    h2_list = np.concatenate((s1, a1, s2), axis=1)
    h3_list = np.concatenate((h2_list, a2, s3), axis=1)

    r1_tilde = r_tilde(1, h1_list, a1)  
    r2_tilde = r_tilde(2, h2_list, a2)  
    r3_tilde = r_tilde(3, h3_list, a3)  

    Q3_real = r3_tilde.copy()           
    Q2_real = r2_tilde + Q3_real        
    Q1_real = r1_tilde + Q2_real       

    N_synth = max(1, int(np.floor(0.1 * m)))

    theta_1_new = theta_1.copy()
    theta_2_new = theta_2.copy()
    theta_3_new = theta_3.copy()

    a1_to25 = a1[:, 0] * 5 + a1[:, 1]
    a2_to25 = a2[:, 0] * 5 + a2[:, 1]
    a3_to25 = a3[:, 0] * 5 + a3[:, 1]

    dim_s = s1.shape[1]
    dim_a = 2

    for a_index in range(25):
        act1 = np.array([[a_index // 5, a_index % 5]])

        mask1 = (a1_to25 == a_index)
        real_design_1 = design_matrix_1[mask1]                  # shape (n1, feature_dim1)
        real_Q1 = Q1_real[mask1]                                # shape (n1,)

        s1_synth = np.random.uniform(0, 1, size=(N_synth, dim_s))
        a1_synth = np.tile(act1, (N_synth, 1))                  # shape (N_synth, 2)

        Q1_synth = compute_Q_function(
            1,
            s1_synth,
            a1_synth,
            P1_hat, P2_hat,
            theta_2, theta_3,
            r_tilde,
            n_MC
        )                                                       # shape (N_synth,)
        
        design_synth_1 = generate_design_matrix(s1_synth)      # shape (N_synth, feature_dim1)

        if real_design_1.shape[0] > 0:
            design_comb_1 = np.vstack((real_design_1, design_synth_1))  # shape (n1 + N_synth, feat_dim1)
            Q_comb_1 = np.concatenate((real_Q1, Q1_synth))              # shape (n1 + N_synth,)
        else:
            design_comb_1 = design_synth_1
            Q_comb_1 = Q1_synth

        theta_1_inc = lstsq(design_comb_1, Q_comb_1)[0]        # shape (feat_dim1,)
        theta_1_new[:, a_index] = theta_1[:, a_index] + eta * theta_1_inc

        act2 = np.array([[a_index // 5, a_index % 5]])

        mask2 = (a2_to25 == a_index)
        real_design_2 = design_matrix_2[mask2]                  # shape (n2, feature_dim2)
        real_Q2 = Q2_real[mask2]                                # shape (n2,)


        s1_synth2 = np.random.uniform(0, 1, size=(N_synth, dim_s))
        s2_synth2 = np.random.uniform(0, 1, size=(N_synth, dim_s))
        a1_synth2 = np.random.choice(5, size=(N_synth, dim_a)) 
        
        a2_synth2 = np.tile(act2, (N_synth, 1))                  

        h2_synth = np.concatenate((s1_synth2, a1_synth2, s2_synth2), axis=1)  # shape (N_synth, dim_h2)

        Q2_synth = compute_Q_function(
            2,
            h2_synth,
            a2_synth2,
            P1_hat, P2_hat,
            theta_2, theta_3,
            r_tilde,
            n_MC
        )  # shape (N_synth,)

        sh2_synth = np.concatenate((s1_synth2, s2_synth2), axis=1)
        design_synth_2 = generate_design_matrix(sh2_synth, a1_synth2)  # shape (N_synth, feat_dim2)

        if real_design_2.shape[0] > 0:
            design_comb_2 = np.vstack((real_design_2, design_synth_2))
            Q_comb_2 = np.concatenate((real_Q2, Q2_synth))
        else:
            design_comb_2 = design_synth_2
            Q_comb_2 = Q2_synth

        theta_2_inc = lstsq(design_comb_2, Q_comb_2)[0]
        theta_2_new[:, a_index] = theta_2[:, a_index] + eta * theta_2_inc

        act3 = np.array([[a_index // 5, a_index % 5]])

        mask3 = (a3_to25 == a_index)
        real_design_3 = design_matrix_3[mask3]                  # shape (n3, feature_dim3)
        real_Q3 = Q3_real[mask3]                                # shape (n3,)

        s1_synth3 = np.random.uniform(0, 1, size=(N_synth, dim_s))
        s2_synth3 = np.random.uniform(0, 1, size=(N_synth, dim_s))
        s3_synth3 = np.random.uniform(0, 1, size=(N_synth, dim_s))
        a1_synth3 = np.random.choice(5, size=(N_synth, dim_a))
        a2_synth3 = np.random.choice(5, size=(N_synth, dim_a))
        a3_synth3 = np.tile(act3, (N_synth, 1))                  # shape (N_synth, 2)

        h3_synth = np.concatenate(
            (s1_synth3, a1_synth3, s2_synth3, a2_synth3, s3_synth3), axis=1
        )  # shape (N_synth, dim_h3)

        Q3_synth = r_tilde(3, h3_synth, a3_synth3)               # shape (N_synth,)

        sh3_synth = np.concatenate((s1_synth3, s2_synth3, s3_synth3), axis=1)
        ah2_synth = np.concatenate((a1_synth3, a2_synth3), axis=1)
        design_synth_3 = generate_design_matrix(sh3_synth, ah2_synth)  # shape (N_synth, feat_dim3)

        if real_design_3.shape[0] > 0:
            design_comb_3 = np.vstack((real_design_3, design_synth_3))
            Q_comb_3 = np.concatenate((real_Q3, Q3_synth))
        else:
            design_comb_3 = design_synth_3
            Q_comb_3 = Q3_synth

        theta_3_inc = lstsq(design_comb_3, Q_comb_3)[0]
        theta_3_new[:, a_index] = theta_3[:, a_index] + eta * theta_3_inc

    return theta_1_new, theta_2_new, theta_3_new




def update_action(a_index, h1_list, h2_list, h3_list, design_matrix_1, design_matrix_2, design_matrix_3,
                  P1_hat, P2_hat, theta_2, theta_3, r_tilde, eta, n_MC, m):
    a = np.array([a_index // 5, a_index % 5]).reshape(1, 2)

    Q1_list = compute_Q_function(1, h1_list, np.repeat(a, m, axis=0), P1_hat, P2_hat, theta_2, theta_3, r_tilde, n_MC)
    theta_1_increment = lstsq(design_matrix_1, Q1_list)[0]

    Q2_list = compute_Q_function(2, h2_list, np.repeat(a, m, axis=0), P1_hat, P2_hat, theta_2, theta_3, r_tilde, n_MC)
    theta_2_increment = lstsq(design_matrix_2, Q2_list)[0]

    Q3_list = compute_Q_function(3, h3_list, np.repeat(a, m, axis=0), P1_hat, P2_hat, theta_2, theta_3, r_tilde, n_MC)
    theta_3_increment = lstsq(design_matrix_3, Q3_list)[0]

    return theta_1_increment, theta_2_increment, theta_3_increment


def standard_Q_learning(s1, a1, s2, a2, s3, a3, r1, r2, r3):
    # offline data: (s1, a1, s2, a2, s3, a3, r)
    n = s1.shape[0]
    sh2 = np.concatenate( (s1, s2), axis=1 )   # 'state history'
    sh3 = np.concatenate( (s1, s2, s3), axis=1 )
    ah2 = np.concatenate( (a1, a2), axis=1 )
    
    design_matrix_1 = generate_design_matrix(s1)
    design_matrix_2 = generate_design_matrix(sh2, a1)
    design_matrix_3 = generate_design_matrix(sh3, ah2)
    design_matrix_1 = np.maximum(design_matrix_1, np.zeros(design_matrix_1.shape))
    design_matrix_2 = np.maximum(design_matrix_2, np.zeros(design_matrix_2.shape))
    design_matrix_3 = np.maximum(design_matrix_3, np.zeros(design_matrix_3.shape))
    
    a1_to25 = a1[:,0]*5+a1[:,1]
    a2_to25 = a2[:,0]*5+a2[:,1]
    a3_to25 = a3[:,0]*5+a3[:,1]
    
    theta_1 = np.zeros((L*1*dim_s, 25))
    theta_2 = np.zeros((L*2*dim_s + 10, 25))
    theta_3 = np.zeros((L*3*dim_s + 20, 25))
    
    stage1_missing = np.zeros(25)
    stage2_missing = np.zeros(25)
    stage3_missing = np.zeros(25)
    ##### idea:
    # if act3 is not missing, then Q3(h3, act3) is learned
    # if act3 is missing, set theta3 s.t. Q3(h3, act3) = -inf
    
    # compute theta3    
    for act3 in range(25):
        if np.sum( a3_to25==act3 )>0:
            stage3_missing[act3] = 0   # not missing
            theta_3[:,act3] = lstsq(design_matrix_3[a3_to25==act3,:], r3[a3_to25==act3])[0]
        else:
            stage3_missing[act3] = 1   # missing
            theta_3[:,act3] = -np.inf*np.ones(L*3*dim_s + 20)   # set to -inf
                    
    # compute V3^hat
    Q3_hat = (design_matrix_3 + 1e-100) @ theta_3   # shape is n*25
    V3_hat = np.max(Q3_hat, axis=1)
    
    # compute theta2
    for act2 in range(25):
        if np.sum( a2_to25==act2 )>0:
            stage2_missing[act2] = 0   # not missing
            theta_2[:,act2] = lstsq(design_matrix_2[a2_to25==act2,:], V3_hat[a2_to25==act2] + r2[a2_to25==act2])[0]
        else:
            stage2_missing[act2] = 1   # missing
            theta_2[:,act2] = -np.inf*np.ones(L*2*dim_s + 10)
    
    # compute V2^hat
    Q2_hat= (design_matrix_2 + 1e-100) @ theta_2 
    V2_hat = np.max(Q2_hat, axis=1) 
    
    # compute theta1
    for act1 in range(25):
        if np.sum(a1_to25==act1)>0:
            stage1_missing[act1] = 0   # not missing
            theta_1[:, act1] = lstsq(design_matrix_1[a1_to25==act1,:], V2_hat[a1_to25==act1] + r1[a1_to25==act1])[0]
        else:
            stage1_missing[act1] = 1   # missing
            theta_1[:, act1] = -np.inf*np.ones(L*1*dim_s)

    return [theta_1, theta_2, theta_3]


def estimate_behavior_RF(s1,a1,s2,a2,s3,a3):
    h1 = s1
    h2 = np.concatenate((s1,a1,s2),axis=1)
    h3 = np.concatenate((h2,a2,s3),axis=1)
    pi_b_1_hat = RandomForestClassifier(random_state=0).fit(h1, (a1[:,0]*5+a1[:,1]) )
    pi_b_2_hat = RandomForestClassifier(random_state=0).fit(h2, (a2[:,0]*5+a2[:,1]) )
    pi_b_3_hat = RandomForestClassifier(random_state=0).fit(h3, (a3[:,0]*5+a3[:,1]) )
    return [pi_b_1_hat,pi_b_2_hat,pi_b_3_hat]


### seed, c are given by the shell script (p_split=0.8, eta=20, n_MC=100, m=1000, T=20)

seed_list = np.arange(50)
c_list = np.array([0, 1, 5, 10, 20, 50, 100])

seed_index = np.argmax(seed_list == seed)
c_index = np.argmax(c_list == c)
job_index = seed_index*7 + c_index

np.random.seed(seed)

s1_train,s1_test,a1_train,a1_test,s2_train,s2_test,a2_train,a2_test, s3_train,s3_test,a3_train,a3_test,s4_train,s4_test,r1_train,r1_test,r2_train,r2_test,r3_train,r3_test = train_test_split(s1,a1,s2,a2,s3,a3,s4,r1,r2,r3, test_size= 0.5, random_state= seed)
a1_real_pool = a1_train
a2_real_pool = a2_train
a3_real_pool = a3_train
n_test = s1_test.shape[0]

h1_test = s1_test
h2_test = np.concatenate((s1_test,a1_test,s2_test),axis=1)
h3_test = np.concatenate((h2_test,a2_test,s3_test),axis=1)
a1_test_to25 = a1_test[:,0]*5+a1_test[:,1]
a2_test_to25 = a2_test[:,0]*5+a2_test[:,1]
a3_test_to25 = a3_test[:,0]*5+a3_test[:,1]


pi_b_1_hat,pi_b_2_hat,pi_b_3_hat = estimate_behavior_RF(s1_test,a1_test,s2_test,a2_test,s3_test,a3_test)

def OPE(theta_1, theta_2, theta_3, greedy=False):
    # compute importance weights for each stage
    w1 = get_policy_prob(1, theta_1, h1_test, greedy)[np.arange(n_test), a1_test_to25] / \
         pi_b_1_hat.predict_proba(h1_test)[pi_b_1_hat.classes_ == a1_test_to25[:, np.newaxis]]
    
    w2 = get_policy_prob(2, theta_2, h2_test, greedy)[np.arange(n_test), a2_test_to25] / \
         pi_b_2_hat.predict_proba(h2_test)[pi_b_2_hat.classes_ == a2_test_to25[:, np.newaxis]]
    
    w3 = get_policy_prob(3, theta_3, h3_test, greedy)[np.arange(n_test), a3_test_to25] / \
         pi_b_3_hat.predict_proba(h3_test)[pi_b_3_hat.classes_ == a3_test_to25[:, np.newaxis]]
    
    w_total = w1 * w2 * w3
    r_total = r1_test + r2_test + r3_test
    
    num = np.mean(w_total * r_total)
    deno = np.mean(w_total)
    
    if deno == 0:
        return np.mean(r_total)
    else:
        return num / deno



theta_qlearn = standard_Q_learning(s1_train,a1_train,s2_train,a2_train,s3_train,a3_train,r1_train, r2_train, r3_train)

cwd = os.getcwd()


kernel1 = GPy.kern.Matern52(input_dim=45, variance=0.2, lengthscale=6)
kernel2 = GPy.kern.Matern52(input_dim=90, variance=1, lengthscale=18)
kernel3 = GPy.kern.Matern52(input_dim=135, variance=5, lengthscale=50)

P1_hat,P2_hat,P3_hat, P1_hat_r,P2_hat_r,P3_hat_r = estimate_P(s1_train,a1_train,s2_train,a2_train,s3_train,a3_train,s4_train, r1_train,r2_train,r3_train, 10,kernel1,kernel2,kernel3)

# proposed algorithm
T = 20
theta_1 = np.zeros((L*1*dim_s, 25))
theta_2 = np.zeros((L*2*dim_s + 10, 25))
theta_3 = np.zeros((L*3*dim_s + 20, 25))
OPE_list = np.zeros(T+1)
OPE_list[0] = OPE(theta_1,theta_2,theta_3)


def modified_reward(k,h,a):
    return r_tilde(k,h,a, P1_hat, P2_hat, P3_hat, P1_hat_r,P2_hat_r,P3_hat_r, c_pessimism = c)


for t in trange(T, desc="Policy Iteration"):
    time_1 = time.time()

    theta_1, theta_2, theta_3 = one_iteration_real_data(
        theta_1, theta_2, theta_3,
        s1_train, a1_train, s2_train, a2_train, s3_train, a3_train,
        P1_hat, P2_hat,
        eta=80, r_tilde=modified_reward
    )
    OPE_list[t+1] = OPE(theta_1,theta_2,theta_3)
    
