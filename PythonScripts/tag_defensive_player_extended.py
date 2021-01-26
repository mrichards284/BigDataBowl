#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:04:27 2020

@author: pengwei
"""


import os
import numpy as np
import pandas as pd

from IO_HMM import *

###################################


####################################


data1 = pd.read_csv('~/Box/Big Data Bowl/Data/week1.csv', sep = ',')
data2 = pd.read_csv('~/Box/Big Data Bowl/Data/week2.csv', sep = ',')
data3 = pd.read_csv('~/Box/Big Data Bowl/Data/week2.csv', sep = ',')
# conbine the data of different weeks
# data = pd.concat([data1, data2,data3], ignore_index=True)
data = data1


data = data.set_index(['gameId','playId','time'])


off_roles =['QB','WR','HB','RB','FB','TE']
def_roles = ['CB','DB','DE','DL','LB','ILB','MLB','NT','OLB','FS','S','SS']

# seperate offensive, defensive and footabll tracking data

offen = data[data['position'].isin(off_roles)]
defen = data[data['position'].isin(def_roles)]
ball = data[data['displayName'].isin(['Football'])]


defen_size = defen.groupby(['gameId','playId','time']).size()
# defen_size.value_counts()
de_size = defen_size[defen_size == 7]

of_size = offen.groupby(['gameId','playId','time']).size()
#of_size.value_counts()
of_size = of_size[of_size == 6]

#### The plays for which the offensize players size = 6 and the defensive players size = 7

def_1 = defen.loc[of_size.index & de_size.index]
of_1 = offen.loc[de_size.index & of_size.index]
ball_1 = ball.loc[de_size.index & of_size.index]


#### vetorize each play
of_2 = of_1.groupby(['gameId','playId']).agg(lambda x: list(x))
def_2 = def_1.groupby(['gameId','playId']).agg(lambda x: list(x))
ball_2 = ball_1.groupby(['gameId','playId']).agg(lambda x: list(x))

# keep the same index for all data
of_2 = of_2.sort_index()
def_2 = def_2.sort_index()
ball_2 = ball_2.sort_index()

of_1 = of_1.sort_index()
def_1 = def_1.sort_index()
ball_1 = ball_1.sort_index()


print("Number of plays = ", len(def_2)) # 496
print("NUmber of plays = ", len(of_2))
num_play = len(of_2)





# obtain the input sequence, output sequence

def loadData(): # load the training data for defensive player p

    InputData = []
    OutputData = []
    for i in range(num_play):
    
        in_x = np.array(of_2.iloc[i]['x']).reshape((len(of_2.iloc[i]['x']),1))
        in_y = np.array(of_2.iloc[i]['y']).reshape((len(of_2.iloc[i]['y']),1))
        in_z = np.concatenate((in_x, in_y), axis = 1)
    
        out_x = np.array(def_2.iloc[i]['x']).reshape((len(def_2.iloc[i]['x']),1))
        out_y = np.array(def_2.iloc[i]['y']).reshape((len(def_2.iloc[i]['y']),1))
        out_z = np.concatenate((out_x, out_y), axis = 1)
        
        ball_x = np.array(ball_2.iloc[i]['x']).reshape((len(ball_2.iloc[i]['x']),1))
        ball_y = np.array(ball_2.iloc[i]['y']).reshape((len(ball_2.iloc[i]['y']),1))
        ball_z = np.concatenate((ball_x, ball_y), axis = 1)
    
        inseq = []
        outseq = []
        for i in range(len(in_z) // 6):
            obs_in = in_z[6*i: 6*i+6]
            obs_out = out_z[7*i:7*i+7]
            ball = ball_z[i]
            
            # input = bias term  + ball(x,y) + offens(x1,y1,x2,y2,...x6,y6)
            inseq.append( [1] + ball.tolist() + obs_in.reshape(12).tolist())
            # output = palyer p (x,y)
            outseq.append(obs_out.reshape(14).tolist())
        
        # add one observation to the training dataset
        inseq = np.array(inseq).T
        outseq = np.array(outseq).T
        
        for p in range(7):
            InputData.append(inseq)
            OutputData.append(outseq[2*p:2*p+2])
    
    #print(InputData[0].shape) input_dim * T
    #print(OutputData[0].shape) input dim * T
    print("done getting the training dataset .....")

    return InputData, OutputData 




def getDenfensiveAssignment(k, InputData, OutputData, IOhmm):


    Label = [[] for i in range(num_play)] # The labels for all 7 palyers of all plays
    # obtain label matrix 7 * T
    for i in range(num_play):
        lab = []
        for p in range(7):
        # get path for palyer p
            inseq = InputData[i*7+p]
            outseq = OutputData[i*7+p]
            lab.append(IOhmm.getBestPath(k,inseq,outseq))
 
        lab = np.array(lab)
        R, C = lab.shape # R = 7, C = T
        jerseyNumber = np.array(of_2.iloc[i]['jerseyNumber']).reshape(C,6).T
        for r in range(R):
            for c in range(C):
                if lab[r][c] < 6:
                    lab[r][c] = jerseyNumber[lab[r][c]][c]
                else:
                    lab[r][c] = -1
        
        lab = np.reshape(lab, -1, order='F') # 7 by T 2D-array --> 7 * T 1D-array
        Label[i] = lab

        
    return Label
            
    
np.random.seed(7)
InputData, OutputData = loadData()
m = InputData[0].shape[0]  # m = 15 # input dimension
n = OutputData[0].shape[0] # n = 2 # output dimension
    
# train the model
k = 7 # num states: 6 man +  zone
print('InputData: ', len(InputData))
print('k:', k, ", m:", m, ", n:", n)

# train the model
wp = IO_HMM(k=k, m=m, n=n)
wp.Generalized_EM(k,m,n,InputData,OutputData)
    
# obtai the path
Label0 = getDenfensiveAssignment(k, InputData, OutputData, wp) # vectorize
Label = [lab for play in Label0 for lab in play]


#wp_0 = IO_HMM(k=k, m=m, n=n)
#Label0 = getDenfensiveAssignment(k, InputData, OutputData, wp_0) # vectorize
#Label = [lab for play in Label0 for lab in play]


# defend who? 
def_1['tag'] = Label
dat = pd.concat([of_1,def_1,ball_1])
dat = dat.reset_index()
dat = dat.sort_values(by =['gameId', 'playId','time'])

# obtain the tagged data
dat.to_csv('~/Box/Big Data Bowl/Data/taggedData2.csv', index=False)





# takea look at dat
'''
m = 20000
n = m + 50
np.array(dat.iloc[14 * m:14 * n]['tag']).reshape((n-m), 14)
dat[(dat['gameId'] == 2018090600)& (dat['playId'] ==75 )& (dat['time'] =='2018-09-07T01:07:14.599Z')]['tag']
'''
    