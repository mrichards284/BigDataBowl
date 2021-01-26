#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 22:50:31 2020

@author: pengwei

Assign an offensive player to each defensive player using input-output HMM:
If B is defending A, then (x_b - x_a, y_b - y_a, sin(o_b)- sin(o_a), dir_b - dir_a ~ N(mu_a, Sigma_a)).
And assuming that transition probability only depends on the previous state (markov chain).
"""



import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt 
#from scipy.special import softmax
from scipy.special import log_softmax
from scipy.stats import multivariate_normal

# z_t: the hidden state; y_t: the output vector; x_t: the input vector


###############################################################################
# I: Input-Output HMM
###############################################################################

class IO_HMM_human():
       
    def __init__(self, k, m, n):
        
        self.r = 0.1 # learning rate
        # in log form 
        self.transition = log_softmax(np.random.normal(0, 1, size=(k,k)), axis=1) # tansition probability
        self.pi = log_softmax(np.ones(k)) # inital probability
        self.emission = np.zeros((k, n, m)) # determine defending who 
        
        for i in range(k):
            for j in range(n):
                self.emission[i][j][i + j * k] = 1
                
        self.mu = np.ones((k, n))  # mean
        self.Sigma = np.zeros((k,n,n)) # variance 
        for i in range(k):
            for j in range(n):
                self.Sigma[i][j][j] = 2.0
                
        
    ## Forward: Pr(z_t=i | input, output_0^t)
    def getAlpha(self, k, in_seq, out_seq):
        
        T = in_seq.shape[1]
        alpha = np.full((k, T), -np.inf)
        # time 0 
        for i in range(k):
            alpha[i][0] = self.pi[i] + np.log(multivariate_normal.pdf(out_seq[:,0] - 
                 self.emission[i] @ in_seq[:,0], self.mu[i], self.Sigma[i]))
        # time 1 to T
        for t in range(1, T):
            for i in range(k):
                for j in range(k):# t-1
                    
                    logp_ji = self.transition[j][i]
                    alpha[i][t] = np.logaddexp(alpha[i][t], (alpha[j][t-1] + logp_ji))
                logp_iout =  np.log(multivariate_normal.pdf(out_seq[:,t] - self.emission[i] @ in_seq[:,t],
                                                self.mu[i], self.Sigma[i]))
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
                    logp_ij = self.transition[i][j]
                    logp_jout = np.log(multivariate_normal.pdf(out_seq[:,t+1] - self.emission[j] @ in_seq[:,t+1],
                                                               self.mu[j], self.Sigma[j]))
                    
                    beta[i][t] = np.logaddexp(beta[i][t], logp_ij + beta[j][t+1] + logp_jout)
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
                    phi_ji = self.transition[j][i]
                    logp_iout = np.log(multivariate_normal.pdf(out_seq[:,t] - self.emission[i] @ in_seq[:,t],
                                                    self.mu[i], self.Sigma[i]))
                    
                    xi[j][i][t-1] = logp_iout + alpha[j][t-1] + beta[i][t] + phi_ji  -  L 
        return xi 
    
    # .........................................................................
    # .........................................................................
    # Learn parameters 
    def Generalized_EM(self, k,m,n, InputData, OutputData): 
        
        N = len(InputData)
        epoch = 1
        batchSize = 300
        numBatch = N // batchSize + 1
        
        for iter in range(epoch):
            print("epoch = ", iter + 1, ".....................")
            # update batchwise 
            for r in range(20): #range(numBatch):
                print("batch = ", r, ".......")
            
                GAMMA = []
                transition = gamma = np.full((k, k), -np.inf) # log 
                pi = gamma = np.full(k, -np.inf) # log
                dom = gamma = np.full(k, -np.inf) # log, for normalization
                DOM = gamma = np.full(k, -np.inf) # log, for normalization
            
                mu = np.zeros((k, n)) 
                Sigma = np.zeros((k,n,n))
            
                size = min(N, (r+1) * batchSize) - r * batchSize
                for p in range(r * batchSize, min(N, (r+1) * batchSize)):
        
                    
                    in_seq = InputData[p]
                    out_seq = OutputData[p]
                    T = in_seq.shape[1]
                    # .........................................................
                    ####### E-step: obtain the posterior distribution
                    # .........................................................
                    alpha = self.getAlpha(k, in_seq, out_seq)
                    beta = self.getBeta(k, in_seq, out_seq)
                    gamma = self.getGamma(k, alpha, beta)
                    xi = self.getXi(k, alpha, beta, in_seq, out_seq)
                    
                    GAMMA.append(gamma)  # record all gamma
                    # .........................................................
                    ###### M_step: update the parameters (SGD)
                    # .........................................................
                    for i in range(k):
                        pi[i] = np.logaddexp(pi[i],gamma[i][0])
                    
                    for i in range(k):
                        for j in range(k):
                            for t in range(T-1):
                                transition[i][j] = np.logaddexp(transition[i][j], xi[i][j][t])
                        
        
                    for i in range(k):
                        for t in range(T):
                            if t < T - 1:
                                dom[i] = np.logaddexp(dom[i],gamma[i][t])
                                mu[i] += np.exp(gamma[i][t]) * (out_seq[:,t] - self.emission[i] @ in_seq[:,t])
                                DOM[i] = np.logaddexp(DOM[i], gamma[i][t])
                        
                # upate pi, in log      
                self.pi = pi - np.log(size)
                # update transition, in log 
                for i in range(k):
                    for j in range(k):
                        self.transition[i][j] = transition[i][j] - dom[i]
            
                # update mean
                for i in range(k):
                    self.mu[i] = mu[i] / np.exp(DOM[i]) 
                    # update Sigma   
                for i in range(k):
                    for p in range(r * batchSize, min(N, (r+1) * batchSize)):
                        T = InputData[p].shape[1]
                        for t in range(T):
                            x = OutputData[p][:,t] - self.emission[i] @ InputData[p][:,t]
                            
                            Sigma[i] += 1 / np.exp(DOM[i]) * np.exp(GAMMA[p - r*batchSize][i][t]) *  (x - self.mu[i]).reshape(n,1).dot((x - self.mu[i]).reshape(1,n))
                
                self.Sigma = Sigma 
                #### to check if the parameters are becoming stable ######
                ##########################################################
                off_positions = pd.Series(['QB', 'WR', 'WR', 'RB', 'TE','FB'], name='Assignment')
                # initial 
                print(pd.DataFrame(np.round(np.exp(self.pi),2), index = off_positions, columns=['initial_prob']))
                # transiton 
                print(pd.DataFrame(np.round(np.exp(self.transition),2), columns = off_positions , index = off_positions))
                # mu
                print(pd.DataFrame(np.round(self.mu,2), columns = ['x','y','o','dir'], index = off_positions))
                # varaince matrix 
                #for i in range(6):
                #    print(off_positions[i],':')
                #    print(pd.DataFrame(np.round(wp.Sigma[i],2), columns = ['x','y','o'] , index = ['x','y','o']))
                #    print("...............")

        return 
    
    # .........................................................................
    # .........................................................................
    # Viterbi Algorithm: obtain the mostlikely path of the hidden states
    
    def getBestPath(self, k, in_seq, out_seq):
        
        T = in_seq.shape[1]
        parent = [[]] # in log scale
        log_out = np.zeros(k)
        for i in range(k):
            log_out[i] =  np.log(multivariate_normal.pdf(out_seq[:,0] - self.emission[i] @ in_seq[:,0], self.mu[i], self.Sigma[i]))
        cur = self.pi + log_out
        cur = cur.reshape((k,1))
        
        for t in range(1,T):
            log_p = self.transition # row -> col k * k 
            log_out = np.zeros(k)
            
            for i in range(k):
                log_out[i] = np.log(multivariate_normal.pdf(out_seq[:,t] - 
                       self.emission[i] @ in_seq[:,t], self.mu[i], self.Sigma[i]))
            
            pos = log_p + log_out.reshape((1,k))
            
            parent.append(np.argmax(cur + pos, axis = 0)) # for each column
            
            cur = np.max(cur + pos, axis = 0).reshape((k,1))
            
        loc = np.argmax(cur)
        path = [loc]
        for t in range(T-1, 0, -1):
            loc = parent[t][loc]
            path.append(loc)
        
        
        return path[::-1]
    
    



# test E-step
'''
k = 6
m = 18
n = 3


inseq = np.random.normal(0,1, size = (m,10))
outseq = np.random.normal(0,1,size= (n,10))
wp = IO_HMM_human(k=k, m=m , n=n)
alpha = wp.getAlpha(k,inseq, outseq) 
beta = wp.getBeta(k,inseq,outseq)
gamma = wp.getGamma(k, alpha, beta)
Xi = wp.getXi(k, alpha, beta, inseq, outseq)
'''

###############################################################################
# II. Generate training data and label new data 
###############################################################################

def dataPreProcess(Data):
    #Data = pd.read_csv('~/Box/Big Data Bowl/Data/week3.csv', sep = ',')
    Data = Data.set_index(['gameId','playId','time'])
    
    def before(df): # filter before passing
        df = df.reset_index().iloc[:,2:]
        df = df.set_index('time')
        indicator = ['pass_forward','qb_sack','pass_shovel','qb_sack_fumble']
        passed = df.index[df['event'].isin(indicator)].tolist()
    
        if passed:
            passtime = passed[-1]
            return df.loc[:passtime]
        
    def after(df): # filter after passing
        df = df.reset_index().iloc[:,2:]
        df = df.set_index('time')
        indicator = ['pass_forward','qb_sack','pass_shovel','qb_sack_fumble']
        passed = df.index[df['event'].isin(indicator)].tolist()
    
        if passed:
            passtime = passed[-1]
            return df.loc[passtime:]
    
    # data before/after passing 
    data = Data.groupby(['gameId', 'playId']).apply(before)
    data_after = Data.groupby(['gameId', 'playId']).apply(after)
     
    def valid(df): # some players are not on the field all the time
        #name = df.index[0]
        df = df.reset_index().iloc[:,2:]
        df = df.set_index('time')
        appear_times = df.groupby(['jerseyNumber','team']).count()
        appear = appear_times['playDirection'].tolist()
     
        #if appear == [appear[0]] * len(appear):
        #   return df
        if appear == [appear[0]] * len(appear):
           return df
    data = data.groupby(['gameId', 'playId']).apply(valid)
    

   
    
    off_roles =['QB','WR','HB','RB','FB','TE']
    def_roles = ['CB','DB','DE','DL','LB','ILB','MLB','NT','OLB','FS','S','SS']

    # seperate offensive, defensive and footabll tracking data

    offen = data[data['position'].isin(off_roles)]
    defen = data[data['position'].isin(def_roles)]
    ball = data[data['displayName'].isin(['Football'])]

    # filter plays with 6 offensive players 
    ##### defen_size = defen.groupby(['gameId','playId','time']).size()
    ##### defen_size.value_counts()
    ##### de_size = defen_size[defen_size == 7]

    of_size = offen.groupby(['gameId','playId','time']).size()
    #of_size.value_counts()
    of_size = of_size[of_size == 6]

    #### The plays for which the offensize players size = 6 and the defensive players size = 7

    def_1 = defen.loc[of_size.index]#defen.loc[of_size.index & de_size.index]
    of_1 = offen.loc[of_size.index]
    ball_1 = ball.loc[of_size.index]
    

    #### vetorize each play
    of_1 = of_1.reset_index()
    of_1.set_index(['gameId','playId'])
    
    def_1 = def_1.reset_index()
    def_1.set_index(['gameId','playId'])
    ball_1 = ball_1.reset_index()
    ball_1.set_index(['gameId','playId'])
    data_after = data_after.reset_index()
    

    of_2 = of_1.groupby(['gameId','playId']).agg(lambda x: list(x))
    def_2 = def_1.groupby(['gameId','playId']).agg(lambda x: list(x))
    #ball_2 = ball_1.groupby(['gameId','playId']).agg(lambda x: list(x))

    # keep the same index for all data to avoid 'join' later
    of_2 = of_2.sort_index()
    def_2 = def_2.sort_index()
    #ball_2 = ball_2.sort_index()
    of_1 = of_1.sort_index()
    def_1 = def_1.sort_index()
    ball_1 = ball_1.sort_index()
    data_after = data_after.sort_index()


    print("Number of plays = ", len(def_2))
    print("NUmber of plays = ", len(of_2))

    
    return of_1, of_2, def_1, def_2, ball_1, data_after



# helper function 
# generate variables and thus to obtain the input sequence, output sequence

def seperate(of, de): # for a single play 
    
    in_x = np.array(of['x']).reshape((len(of['x']),1)) / 50 # Player position along the long axis of the field, 0 - 120 yards.
    in_y = np.array(of['y']).reshape((len(of['y']),1)) / 50 # Player position along the short axis of the field, 0 - 53.3 yards. 
    in_o = np.sin(np.array(of['o']).reshape((len(of['o']),1))) # Player orientation (deg)
    in_dir = np.sin(np.array(of['dir']).reshape((len(of['dir']),1))) #Angle of player motion (deg)
    
    in_z = np.concatenate((in_x, in_y, in_o, in_dir), axis = 1)
    
    out_x = np.array(de['x']).reshape((len(de['x']),1)) / 50
    out_y = np.array(de['y']).reshape((len(de['y']),1)) / 50
    out_o = np.sin(np.array(de['o']).reshape((len(de['o']),1)))
    out_dir = np.sin(np.array(de['dir']).reshape((len(de['dir']),1)))
        
    out_z = np.concatenate((out_x, out_y, out_o, out_dir), axis = 1)
        
    
    inseq = []
    outseq = []
    T = len(in_z) // 6
    n_f = len(out_z) // T # number of defender 7
    for j in range(T):
        obs_in = in_z[6*j: 6*j+6]
        obs_out = out_z[n_f*j: n_f*j + n_f]
            
            
        inseq.append(obs_in.reshape(6 * 4).tolist())
        # output = palyer p (x,y)
        outseq.append(obs_out.reshape(n_f * 4).tolist())
    
    inseq = np.array(inseq).T
    outseq = np.array(outseq).T
    
    return inseq, outseq
          
# get training data
def loadData(of_2, def_2): # load the training data

    InputData = []
    OutputData = []
    num_play = len(def_2)
    
    for i in range(num_play):
    
        of = of_2.iloc[i]
        de = def_2.iloc[i]
        inseq, outseq = seperate(of, de)
        n_f = outseq.shape[0] // 4
        
        for p in range(n_f):
            InputData.append(inseq)
            OutputData.append(outseq[4*p:4*p+4])
    
    return InputData, OutputData 


# helper function
# label a single play
def getDefensiveAssignment(tagger, of, de):
    # play: already aggragate to a vector 
    
    inseq, outseq = seperate(of, de)
    T = inseq.shape[1]
    n_f = len(outseq) // 4 # number of def players
    
    Label = []
    for p in range(n_f):
        lab = tagger.getBestPath(k,inseq, outseq[4*p: 4*p+4]) # path for player p
        Label.append(lab)
        
    Label = np.array(Label)
          
    # reshape to a long vector
  
    R, T = Label.shape # R = 7, C = T
    jerseyNumber = np.array(of['jerseyNumber']).reshape(T,6).T
    for r in range(R):
        for c in range(T):
            Label[r][c] = jerseyNumber[Label[r][c]][c]
                
        
    Label = np.reshape(Label, -1, order='F') # 7 by T 2D-array --> 7 * T 1D-array
    Label = Label.tolist()
    
    
    df = pd.DataFrame({'time':de['time'],'jerseyNumber':de['jerseyNumber']})
    
    df['tag'] = Label
    df['gameId'] = de.name[0]
    df['playId'] = de.name[1]
    #df.columns = ['time','jerseyNumber','tag','gameId','playId']
    df.set_index(['gameId', 'playId'])
    df = df[['gameId','playId','time','jerseyNumber','tag']]


    return df

# label new data
def labelData(tagger, data):
    
    of_1, of_2, def_1, def_2, ball_1, data_after = dataPreProcess(data)
    
    num_play = len(def_2)
    Tagged = []
    for i in range(num_play):
        of, de = of_2.iloc[i], def_2.iloc[i]
        play_i = getDefensiveAssignment(tagger, of, de)
        play_i = play_i.values.tolist()
        Tagged += play_i
    

    Tag = pd.DataFrame(Tagged)
    Tag.columns = ['gameId','playId','time','jerseyNumber','tag']
    Tag = Tag.set_index(['gameId', 'playId','time'])
    Tag = Tag.sort_index()
    
    # defend who? 
  
    def_1['tag'] = Tag['tag'].tolist()
    dat = pd.concat([of_1,def_1,ball_1, data_after])
    dat = dat.reset_index()
    dat = dat.sort_values(by =['gameId', 'playId','time'])

    
    return dat


#######################################################################################
    
#######################################################################################
#   1. Train the model 
#######################################################################################
    
         
# set seed
np.random.seed(7)

data1 = pd.read_csv('~/Box/Big Data Bowl/Data/week1.csv', sep = ',')
#data2 = pd.read_csv('~/Box/Big Data Bowl/Data/week2.csv', sep = ',')
#data3 = pd.read_csv('~/Box/Big Data Bowl/Data/week3.csv', sep = ',')
# get training dataset 
data = data1
of_1, of_2, def_1, def_2, ball_1, data_after = dataPreProcess(data)
print('num_play =', len(of_2)) # = len(def_2)
InputData, OutputData = loadData(of_2, def_2)

m = InputData[0].shape[0]  # m = 18 # input dimension
n = OutputData[0].shape[0] # n = 4 # output dimension
k = 6                      # 6 hidden state
print(len(InputData))
print(len(OutputData))

# train the model
wp = IO_HMM_human(k=k, m=m, n=n)
wp.Generalized_EM(k,m,n,InputData,OutputData)


# check the model parameters
off_positions = pd.Series(['QB', 'WR', 'WR', 'RB', 'TE','FB'], name='Assignment')
# initial prob
pd.DataFrame(np.round(np.exp(wp.pi),3), index = off_positions, columns=['initial_prob'])
# transition 
pd.DataFrame(np.round(np.exp(wp.transition),3), columns = off_positions , index = off_positions)
# mu
pd.DataFrame(np.round(wp.mu,2), columns = ['x','y','o','dir'], index = off_positions )
# variance matrix 
for i in range(6):
    print(off_positions[i],':')
    print(pd.DataFrame(np.round(wp.Sigma[i],3), columns = ['x','y','o','dir'] , index = ['x','y','o','dir']))
    print("...............")


# save the parameters of the model to 'taggerParameters.npz'
file = 'taggerParameters.npz'
np.savez(file, 
         pi = np.exp(wp.pi),
         transition = np.exp(wp.transition),
         mu = wp.mu,
         variance = wp.Sigma)

para = np.load('taggerParameters.npz')





#######################################################################################
#   2. Label the new data of 1-17 weeks with the model 
####################################################################w###################
    
for i in range(1,18):
    week = 'week' + str(i)
    path = '~/Box/Big Data Bowl/Data/' + week + '.csv'
   
    dataToTag = pd.read_csv(path, sep = ',') 
    
    dat = labelData(wp, dataToTag)
    
    # save the labeled data
    new_path = '~/Box/Big Data Bowl/Data/DataLabeled/' + week + '.csv'
    dat.to_csv(new_path)
    print("done labeling ", week, "........" )



