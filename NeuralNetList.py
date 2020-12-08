import numpy as np
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from numpy.random import seed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import make_pipeline

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

def supvecreg(S,L):
    M,N = S.shape
    Dj = np.zeros([M,L])
    hatS = np.zeros([M,N])

    # define model
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    yS = np.ravel(hatS[:,0])
    svr_rbf.fit(Dj, yS)

    for j in range(N):
        yhat = svr_rbf.predict(Dj)
        yvec = yhat.reshape(-1,1)
        hatS[:,j] = yvec[:,0]
        yS = np.ravel(S[:,j])
        svr_rbf.fit(Dj, yS)
        # demonstrate prediction
        Dj[:,1:L] = Dj[:,0:L-1]
        Dj[:,0] = S[:,j]
    return(hatS)

def ensmtreereg(S,L):
    M,N = S.shape
    Dj = np.zeros([M,L])
    hatS = np.zeros([M,N])

    # define model
    n_estimator = 24
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                          random_state=2)
    rt_lm = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(rt, rt_lm)    
    yS = np.ravel(hatS[:,0])
    yS[0] = 1
    ySt = yS.astype('int')
    pipeline.fit(Dj, ySt)

    for j in range(N):       
        yhat = pipeline.predict(Dj)
        yvec = yhat.reshape(-1,1)
        hatS[:,j] = yvec[:,0]
        yS = np.ravel(S[:,j])
        ySt = yS.astype('int')
        if all(x==ySt[0] for x in ySt):
            ySt[0]= int(ySt[0]+2)
        pipeline.fit(Dj, ySt)
        # demonstrate prediction
        Dj[:,1:L] = Dj[:,0:L-1]
        Dj[:,0] = S[:,j]
    return(hatS)
    
        
