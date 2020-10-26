# Copyright © 2020 Md. Zulfiquar Ali Bhotto, Richard Jones, Stephen Makonin, and Ivan V. Bajic.


import sys
sys.modules[__name__].__dict__.clear()

import numpy as np
import pandas as pd
from scipy.linalg import circulant
import timeit
from EnsemblePredictionNetwork import EPN
from NeuralNetList import stackedlstmp,stackedmlpp


wr = 24
nday = 365
N = nday*wr
L = 30
Ns = (L+1)*wr
epc = 1 # Number of epoch for EPN
vals = np.array([1, -1])
offset = np.array([0,1])
col0 = np.zeros(wr)
col0[offset] = vals
F = np.transpose(circulant(col0))

# Energy consumption data
s = np.array(pd.read_csv('/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/datasamples.csv'))

#Tuning parameters for EPN
paramtr = np.array(pd.read_csv('/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/parameters.csv', header=None, skiprows=None))
#Tuning parameters for LCP
tslot = np.array(pd.read_csv('/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/timeslots.csv', header=None, skiprows=None))

labels = (['UP','LCP','HLP','DLP','RbP','RdP','MMP','KbP','ALM','GMC','LSTM','MLP'])
numdataset = 50
RMSE = np.zeros([numdataset,12])
MAE = np.zeros([numdataset,12])
MAP = np.zeros([numdataset,12])
error4spreadsheet = np.zeros([N,3*numdataset])
error = [np.zeros([N,12]) for i in range(numdataset)]
MAPE = [np.zeros([N,12]) for i in range(numdataset)]
sumerr = [[np.zeros([wr,1]) for i in range(12)] for i in range(numdataset)]
maxst = np.zeros([numdataset,1])
epnruntime = []
lstmruntime = []
mlpruntime = []

for dataset in range(numdataset):
    ri = dataset
    print('dataset # ',ri)
    St = np.zeros([nday,wr])
    St = s[:,ri].reshape([nday,wr])
    S = np.transpose(St)
    maxst[ri] = np.max(S[:,L+1:nday])**2

    delta = [0]*4
    delta[0] = [np.sum(S[int(tslot[ri,1]):int(tslot[ri,2]),i]) for i in range(nday)]
    delta[1] = [np.sum(S[int(tslot[ri,4]):int(tslot[ri,5]),i]) for i in range(nday)]
    delta[2] = [np.sum(S[int(tslot[ri,7]):int(tslot[ri,8]),i]) for i in range(nday)]
    delta[3] = [np.sum(S[int(tslot[ri,10]):int(tslot[ri,11]),i]) for i in range(nday)]
    bnd = np.array([np.mean(delta[i]) for i in range(0,4)]).reshape(-1,1)
    u = np.zeros([4,24])
    u[0,int(tslot[ri,1]):int(tslot[ri,2])] = tslot[ri,3]
    u[1,int(tslot[ri,4]):int(tslot[ri,5])] = tslot[ri,6]
    u[2,int(tslot[ri,7]):int(tslot[ri,8])] = tslot[ri,9]
    u[3,int(tslot[ri,10]):int(tslot[ri,11])] = tslot[ri,12]

    alfa = paramtr[ri,1:6]
    thr = paramtr[ri,6]
    ahv = paramtr[ri,7]

    start = timeit.default_timer()
    hatS_All = EPN(S,u,bnd,L,F,ahv,thr,alfa,epc)
    stop = timeit.default_timer()
    epnruntime.append(stop-start)

    start = timeit.default_timer()
    hatS_lstm = stackedlstmp(S,L)
    hatS_All[10] = np.multiply(hatS_lstm,(hatS_lstm>0))
    stop = timeit.default_timer()
    lstmruntime.append(stop-start)

    start = timeit.default_timer()
    hatS_mlp = stackedmlpp(S,L)
    hatS_All[11] = np.multiply(hatS_mlp,(hatS_mlp>0))
    stop = timeit.default_timer()
    mlpruntime.append(stop-start)
    k = 0
    for j in range(nday):
        for i in range(wr):
            for l in range(12):
                error[ri][k,l] = np.abs(S[i,j]-hatS_All[l][i,j])
                MAPE[ri][k,l] = np.abs(1-hatS_All[l][i,j]/S[i,j])
            k=k+1

    Spwr = 2*np.sum(S[:,L+1:nday])
    for i in range(12):
        RMSE[ri,i] = np.mean(error[ri][Ns:N,i]**2)**0.5
        MAE[ri,i] = np.maximum(0,1-np.sum(error[ri][Ns:N,i])/Spwr)
        MAP[ri,i] = np.ma.masked_invalid(MAPE[ri][Ns:N,i]).mean()
        #print(labels[i],"%.2f" % RMSE[ri,i])

    #error4spreadsheet[:,3*ri:3*(ri+1)] = error[ri][:,9:12]


#df = pd.DataFrame(error4spreadsheet)
#df.to_csv("/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/SampleErrors.csv", index = False)
Hid = pd.Index(labels, name="columns")
df = pd.DataFrame(RMSE, columns=Hid)
df.to_csv("/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/RMSETableIIILag0RQ.csv",
          index = False, float_format='%.2f', sep=',')

Hid = pd.Index(labels, name="columns")
df = pd.DataFrame(MAE, columns=Hid)
df.to_csv("/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/MAETableIIILag0RQ.csv",
          index = False, float_format='%.3f', sep=',')

Hid = pd.Index(labels, name="columns")
df = pd.DataFrame(MAP, columns=Hid)
df.to_csv("/Users/mbhotto/Documents/LatextDocs/AwesenseReportCsv/MAPTableIIILag0RQ.csv",
          index = False, float_format='%.3f', sep=',')

#print(df[['UP','LCP','HLP','DLP','RbP','RdP','MMG','MMH','ALM','GMC','LSTM','MLP']])

#plterror = [np.zeros([wr,1]) for i in range(12)]
#for dataset in range(numdataset):
#    ri = dataset
#    nerror = error[ri]/maxst[ri]
    #pltEngyPdtn(nerror,hatS_All,S)
