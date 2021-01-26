#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:43:21 2020

@author: pengwei

Assign a label to the defense at each time of a play by input-output hmm 
and compare it with the coverage type given in corages_wk1.csv to see if the 
assignment make any sense.

"""



import os
import rope # autocompletion
import numpy as np
import pandas as pd
from IO_HMM import *
from statistics import mode

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
    for i in range(len(of_2)):
    
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
            
            # input =  ball(x,y) + offens(x1,y1,x2,y2,...x6,y6)
            inseq.append(ball.tolist() + obs_in.reshape(12).tolist())
            # output = palyer p (x,y)
            outseq.append(obs_out.reshape(14).tolist())
        
        # add one observation to the training dataset
        inseq = np.array(inseq).T
        outseq = np.array(outseq).T
        
    
        InputData.append(inseq)
        OutputData.append(outseq)
    
    #print(InputData[0].shape) input_dim * T
    #print(OutputData[0].shape) input dim * T
    print("done getting the training datase .....")

    return InputData, OutputData 




def getDenfensiveAssignment(IO_HMM, InputData, OutputData):

    k = 7 # num states

    Label = [[] for i in range(num_play)] # The labels for all 7 palyers of all plays
    
    
   
    for i in range(len(InputData)):
        inseq = InputData[i]
        outseq = OutputData[i]
        lab = IO_HMM.getBestPath(k,inseq, outseq) # path
        Label[i] = lab
            
        
    return Label
   
         
# set seed
np.random.seed(17)
InputData, OutputData = loadData()
m = InputData[0].shape[0]  # m = 14 # input dimension
n = OutputData[0].shape[0] # n = 14 # output dimension
        
k = 7

wp = IO_HMM(k=k, m=m, n=n)
wp.Generalized_EM(k,m,n,InputData,OutputData)
        

Label0 = getDenfensiveAssignment(wp, InputData, OutputData) # vectorize


# defend who? 
Label = [lab for play in Label0 for lab in play]

#dat_cover = def_1.reset_index()
dat_cover = def_1.groupby(['gameId', 'playId','time']).size().reset_index()
dat_cover = dat_cover.sort_values(by =['gameId', 'playId','time'])

dat_cover['coverageType'] = Label
dat_cover = dat_cover[['gameId', 'playId','time', 'coverageType']]


# obtain the tagged data
#dat_cover.to_csv('~/Box/Big Data Bowl/Data/taggedData_coverage.csv', index=False)
coverages = pd.read_csv('~/Box/Big Data Bowl/Data/coverages_wk1.csv', sep = ',')
result = pd.merge(dat_cover, coverages, how='left', on=['gameId', 'playId'])


result.groupby(["coverageType","coverage"]).size()









