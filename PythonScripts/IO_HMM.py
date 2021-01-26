#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:23:18 2020

@author: pengwei
"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt 
from scipy.special import softmax
from scipy.special import log_softmax

# z_t: the hiiden state; y_t: the output vector; x_t: the input vector
# Model: Pr(z_t+1 |z_t) = sigmoid(transition[z_t] * x_t) 
#        Pr(y_t|Z_t) = gaussian(emission[z_t] * x_t, 1)

class IO_HMM():
       
    def __init__(self, k, m, n):
        
        
        self.r = 0.1 # learning rate
      
        self.transition = np.random.normal(0, 1, size=(k,k,m)) # tansition 
        self.emission = np.random.normal(0, 1, size=(k,n,m)) # emission
        self.pi = np.random.normal(0, 1, size=(k,m)) # inital states
        

        
    ## Forward: Pr(z_t=i | input, output_0^t)
    def getAlpha(self, k, in_seq, out_seq):
        
        T = in_seq.shape[1]
        alpha = np.full((k, T), -np.inf)
        # time 0 
        for i in range(k):
            temp = log_softmax(self.pi.dot(in_seq[:,0]))
            alpha[i][0] = temp[i] - 0.5 * np.sum((self.emission[i].dot(in_seq[:,0]) - out_seq[:,0])**2)
        # time 1 to T
        for t in range(1, T):
            for i in range(k):
                for j in range(k):# t-1
                    temp =  log_softmax(self.transition[j].dot(in_seq[:,t]))
                    logp_ji = temp[i]
                    alpha[i][t] = np.logaddexp(alpha[i][t], (alpha[j][t-1] + logp_ji))
                logp_iout = - 0.5 * np.sum( (self.emission[i].dot(in_seq[:,t]) - out_seq[:,t])**2 )
                alpha[i][t] += logp_iout
                
        return alpha
        
    ## Backward: Pr(z_t=i| input, output_t+1^T-1)
    def getBeta(self, k,in_seq, out_seq):
        
        T = in_seq.shape[1]
        beta = np.full((k, T), -np.inf)
        # time T
        for i in range(k):
            beta[i][T-1] = 0
        # time 0: T-1
        for t in range(T-2, -1,-1):
            for i in range(k):
                for j in range(k): # t+1
                    temp = log_softmax(self.transition[i].dot(in_seq[:,t+1]))
                    logp_ij = temp[j]
                    logp_jout = - 0.5 * np.sum( (self.emission[j].dot(in_seq[:,t+1]) - out_seq[:,t+1])**2 )
                    beta[i][t] = np.logaddexp(beta[i][t],  logp_ij + beta[j][t+1] + logp_jout)
        return beta 
        
    ##  Pr(z_t=i | input, output)
    def getGamma(self, k, alpha, beta):
        # P(state_t=i|Y)
        T = alpha.shape[1]
        gamma = np.full((k, T), -np.inf)
        L = - np.inf
        for i in range(k):
            L = np.logaddexp(L, alpha[i][T-1] + beta[i][T-1])
            
        for t in range(T):
            for i in range(k):
                gamma[i][t] = (alpha[i][t] + beta[i][t]) - L
                
        return gamma
        
    ## Pr(z_t=i, x_t+1=j | input, output)
    def getXi(self,k, alpha, beta, in_seq, out_seq):
        
        
        T = alpha.shape[1]
        L = - np.inf
        for i in range(k):
            L = np.logaddexp(L, alpha[i][T-1] + beta[i][T-1])

        xi = np.full((k,k,T - 1), -np.inf)
        
        for j in range(k):
            for i in range(k):
                for t in range(1,T):
                    temp = log_softmax(self.transition[j].dot(in_seq[:,t]))
                    phi_ji = temp[i]
                    logp_iout = - 0.5  * np.sum((self.emission[i].dot(in_seq[:,t]) - out_seq[:,t])**2 )
                    
                    xi[j][i][t-1] = logp_iout + alpha[j][t-1] + beta[i][t] + phi_ji  -  L 
        return xi 
    
    # .........................................................................
    # .........................................................................
    # Learn parameters 
    def Generalized_EM(self, k,m,n, InputData, OutputData): 
        
        N = len(InputData)
        epoch = 3
        batchSize = 50 
        numBatch =  N // batchSize + 1
        
        for iter in range(epoch):
            print("epoch = ", iter + 1, ".....................")
            for p in range(numBatch):
                
                print("batch = ", p, ".......")
                gradient_transition = np.zeros((k,k,m))
                gradient_emission = np.zeros((k,n,m))
                gradient_pi = np.zeros((k,m))
            
            
                size = min(N, (p+1) * batchSize) - p * batchSize
                for i in range(p * batchSize,min(N, (p+1) * batchSize)):
                    # .........................................................
                    ####### E-step: obtain the posterior distribution
                    # .........................................................
                    
                    in_seq = InputData[i]
                    out_seq = OutputData[i]
                    T = in_seq.shape[1]

                    alpha = self.getAlpha(k, in_seq, out_seq)
                    beta = self.getBeta(k, in_seq, out_seq)
                    gamma = self.getGamma(k, alpha, beta)
                    xi = self.getXi(k, alpha, beta, in_seq, out_seq)
                    
                    # .........................................................
                    ###### M_step: update the parameters (SGD)
                    # .........................................................
                    
                    # t=0 update pi
                    
                    x = in_seq[:,0].reshape(m,1)
                    prob = softmax(self.pi.dot(x)) # k * 1 
                    A = np.exp(gamma[:,0]).reshape(k,1) * (1 - prob)
                    
                    gradient_pi = A.dot(x.T) 
                 
                    # t = 1: T update transition
                    for t in range(T-1):
                        x = in_seq[:,t].reshape(m,1)
                        for j in range(k):
                            phi = softmax(self.transition[j].dot(x)) # k * 1
                            A = np.exp(xi[j,:,t]).reshape(k,1) * (1 - phi)
                            
                            gradient_transition[j] += A.dot(x.T)
                
                    #  t = 0: T update emission         
                    for t in range(T):
                        x = in_seq[:,t].reshape(m,1)
                        y = out_seq[:,t].reshape(n,1)
                        for i in range(k):
                            mu = self.emission[i].dot(x) # n * 1
                            A = - np.exp(gamma[i][t]) * (y - mu).reshape(n,1) # n * 1
                            
                            gradient_emission[i] += A.dot(x.T) # n*1, 1*m -> n*m
                            
                # update paraameters (SGD)               
                self.pi = self.pi + self.r / size * gradient_pi
                self.transition = self.transition + self.r / size * gradient_transition
                self.emssion = self.emission + self.r / size * gradient_emission
            
        
        return 
    
    # .........................................................................
    # .........................................................................
    # Vertibi Algorithm: obtain the mostlikely path of the hidden state
    
    def getBestPath(self, k, in_seq, out_seq):
        
        
        T = in_seq.shape[1]
        parent = [[]]
        # in log scale
        cur = log_softmax(self.pi.dot(in_seq[:,0])) - \
            0.5 * np.sum((self.emission.dot(in_seq[:,0]) - out_seq[:,0])**2, axis = 1)
        cur = cur.reshape((k,1))
        
        
        for t in range(1,T):
            log_p = log_softmax(self.transition.dot(in_seq[:,t]), axis = 1) # row -> col k * k 
            log_out = - 0.5 * np.sum((self.emission.dot(in_seq[:,t]) - out_seq[:,t])**2, axis = 1)
            
            pos = log_p + log_out.reshape((1,k))
            
            parent.append(np.argmax(cur + pos, axis=0)) # for each column
            
            cur = np.max(cur + pos, axis=0).reshape((k,1))
            
        loc = np.argmax(cur)
        path = [loc]
        for t in range(T-1, 0, -1):
            loc = parent[t][loc]
            path.append(loc)
        
        
        return path[::-1]
    
    
    


# test E-step
'''
k = 2
m = 6
n = 7


inseq = np.random.normal(0,1, size = (m,10))
outseq = np.random.normal(0,1,size= (n,10))
wp = IO_HMM(k=k, m=m , n=n)
alpha = wp.getAlpha(k,inseq, outseq) 
beta = wp.getBeta(k,inseq,outseq)
gamma = wp.getGamma(k, alpha, beta)
Xi = wp.getXi(k, alpha, beta, inseq, outseq)
'''


'''
# Test the whole Algorithm 

N = 100
InputData = []
OutputData = []
for i in range(N):
    inseq = np.random.normal(0,1, size = (m,20))
    outseq =   inseq
    InputData.append(inseq)
    OutputData.append(outseq)

wp = IO_HMM(k, n, m)
wp.Generalized_EM(k,m,n,InputData,OutputData)
    
label = []
for i in range(len(InputData)):
    inseq = InputData[i]
    outseq = OutputData[i]
    lab = wp.getBestPath(k,inseq, outseq)
    label.append(lab)


print(pd.DataFrame(label))

pd.DataFrame(label).iloc[:5,:].T.plot()
'''


# data preparation 

    