# Copyright © 2020 Md. Zulfiquar Ali Bhotto, Richard Jones, Stephen Makonin, and Ivan V. Bajic.


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from numpy.random import seed

seed(2)
#from theano import set_random_seed
#set_random_seed(2)
def stackedlstmp(S,L):
    M,N = S.shape
    Dj = np.zeros([M,L,1])
    hatS = np.zeros([M,N])
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(L, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(Dj, hatS[:,0], epochs=10, verbose=0)

    for j in range(N):
        #print(j)
        #x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(Dj, verbose=0)
         # fit model
        hatS[:,j] = yhat[:,0]
        model.fit(Dj, S[:,j], epochs=10, verbose=0)
        # demonstrate prediction
        Dj[:,1:L,0] = Dj[:,0:L-1,0]
        Dj[:,0,0] = S[:,j]
    return(hatS)

def stackedmlpp(S,L):
    M,N = S.shape
    Dj = np.zeros([M,L])
    hatS = np.zeros([M,N])

    # define model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=L))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(Dj, hatS[:,0], epochs=1, verbose=0)
    for j in range(N):
        # demonstrate prediction
        yhat = model.predict(Dj, verbose=0)
        hatS[:,j] = yhat[:,0]
        model.fit(Dj, S[:,j], epochs=1, verbose=0)
        # demonstrate prediction
        Dj[:,1:L] = Dj[:,0:L-1]
        Dj[:,0] = S[:,j]
    return(hatS)
