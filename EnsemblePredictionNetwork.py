import numpy as np

def EPN(S,u,bnd,L,F,cpara,thr,alfa,epoch):
    M,N = S.shape
    hatS_all = [np.zeros([M,N]) for i in range(14)]
    Dj = np.zeros([M,L])
    hatS = np.zeros([M,1])
    
    w1 = np.zeros([L,1])

    w2= np.zeros([L,1])
    reg = 1e-6*np.eye(len(bnd))

    w3 = np.zeros([L,1])
    invf = np.linalg.inv(F+1e-12*np.eye(M))

    w4 = np.zeros([L,1]) 
    Dpd = np.zeros([M,1])
    Dpj = np.zeros([M,L])

    w5 =  np.zeros([L,1])

    w6 =  np.zeros([L,1])

    w7 = np.zeros([L,1])
    c7 = 1/M*np.ones([M,1])    
    u7  = np.ones([M,1])
    Z7 = np.eye(M)-1/M*np.matmul(u7,np.transpose(u7))

    w8 = np.zeros([L,1]) 
    Z8 = 1e-10*np.eye(M)

    w9 = np.zeros([L,1])
    bw9 = np.eye(M)

    
    a10 = [1/9*np.ones([9,1]) for i in range(M)]      
    u10  = np.ones([9,1])
    Z10 = np.eye(9)-1/9*np.matmul(u10,np.transpose(u10))
    lgd = np.zeros([9,1])

    for j in range(N):
        # Prediction in Neuron 1
        djw = np.matmul(Dj,w1).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        wp = w1
        for ep in range(epoch):
            djw = np.matmul(Dj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]).reshape(-1,1)
            Dt = np.transpose(Dj)
            Dte = np.matmul(Dt,e)
            DDte = np.matmul(Dj,Dte)
            mud = epoch*(np.matmul(np.transpose(DDte),DDte)+1e-6)
            mun = np.matmul(np.transpose(Dte),Dte)
            mu = mun/mud
            wp = wp +mu*Dte
        w1 = wp
        h1s = np.multiply(hatS,(hatS>0))
        hatS_all[0][:,j] = h1s[:,0]
        
        
        # Prediction in Neuron 2
        B = np.matmul(u,Dj)
        Bt = np.transpose(B)
        BBt = np.linalg.inv(np.matmul(B,Bt)+reg)
        Bu = np.matmul(Bt,BBt)
        Z = np.eye(L)- np.matmul(Bu,B)
        C = np.matmul(Bu,bnd)
        djw = np.matmul(Dj,w2).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        wp = w2
        for ep in range(epoch):
            djw = np.matmul(Dj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]).reshape(-1,1)
            Dt = np.transpose(Dj)
            Dte = np.matmul(Dt,e)
            ZD = np.matmul(Z,Dte)
            DZD = np.matmul(Dj,ZD)
            mud = epoch*(np.matmul(np.transpose(DZD),DZD)+1e-6)
            tZ = np.matmul(Z,wp)+C
            ez = S[:,j].reshape(-1,1)-np.matmul(Dj,tZ)
            Dtz = np.matmul(Dt,ez)
            mun = np.matmul(np.transpose(ZD),Dtz)
            mu = mun/mud
            wp = tZ +mu*ZD
        w2 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[1][:,j] = hgs[:,0]

        # Prediction in Neuron 3
        djw = np.matmul(Dj,w3).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        wp = w3
        for ep in range(epoch):
            djw = np.matmul(Dj,wp).reshape(-1,1)
            fd = np.matmul(F,djw).reshape(-1,1)
            SnS = np.sign(fd).reshape(-1,1)
            MgS = np.maximum(np.abs(fd)-cpara,0).reshape(-1,1)
            fS = np.matmul(invf,np.multiply(MgS,SnS).reshape(-1,1))
            e = (S[:,j]-djw[:,0]).reshape(-1,1) 
            ez = (S[:,j]-fS[:,0]).reshape(-1,1)
            Dt = np.transpose(Dj)
            qj = np.matmul(Dt,ez)
            Dqj = np.matmul(Dj,qj).reshape(-1,1)
            den = epoch*(np.sum(np.multiply(Dqj,Dqj))+1e-4)
            num = np.sum(np.multiply(e,Dqj))
            mu = num/den
            wp = wp + mu*qj
        w3 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[2][:,j] = hgs[:,0]

        # Prediction in Neuron 4
        djw = np.matmul(Dpj,w4).reshape(-1,1)
        hatS[:,0] = djw[:,0]+Dpd[:,0]
        wp = w4
        for ep in range(epoch):
            djw = np.matmul(Dpj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]-Dpd[:,0]).reshape(-1,1) 
            Dt = np.transpose(Dpj)
            qj = np.matmul(Dt,e)
            Dqj = np.matmul(Dpj,qj).reshape(-1,1)
            den = epoch*(np.sum(np.multiply(Dqj,Dqj))+1e-4)
            num = np.sum(np.multiply(e,Dqj))
            mu = num/den
            wp = wp + mu*qj
        w4 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[3][:,j] = hgs[:,0]        

        
        # Prediction in Neuron 5
        djw = np.matmul(Dj,w5).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        wp = w5
        for ep in range(epoch):
            djw = np.matmul(Dj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]).reshape(-1,1)
            Dt = np.transpose(Dj)
            Dte = np.matmul(Dt,np.sign(e))
            DDte = np.matmul(Dj,Dte)
            Z = np.matmul(Dt,e)
            mud = epoch*(np.matmul(np.transpose(DDte),DDte)+1e-6)
            mun = np.matmul(np.transpose(Z),Dte)
            mu = alfa[0]*mun/mud
            wp = wp + mu*Dte
        w5 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[4][:,j] = hgs[:,0]

        # Prediction in Neuron 6
        djw = np.matmul(Dpj,w6).reshape(-1,1)
        hatS[:,0] = djw[:,0]+Dpd[:,0]
        wp = w6
        for ep in range(epoch):
            djw = np.matmul(Dpj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]-Dpd[:,0]).reshape(-1,1) 
            Dt = np.transpose(Dpj)
            qj = np.matmul(Dt,np.sign(e))
            Dqj = np.matmul(Dpj,qj).reshape(-1,1)
            den = epoch*(np.sum(np.multiply(Dqj,Dqj))+1e-4)
            num = np.sum(np.multiply(e,Dqj))
            mu = alfa[1]*num/den
            wp = wp + mu*qj
        w6 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[5][:,j] = hgs[:,0]

        # Prediction in Neuron 7
        djw = np.matmul(Dj,w7).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        wp = w7
        cp = c7
        for ep in range(epoch):
            djw = np.matmul(Dj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]).reshape(-1,1)
            ez = np.sum(np.multiply(cp,e))
            q = np.matmul(Z7,e)*ez
            mu = np.min(cp[:,0])/(epoch*(np.max(np.abs(q))+1e-8))
            Dt = np.transpose(Dj)
            Dtc = np.matmul(Dt,cp)            
            den = epoch*(np.matmul(np.matmul(np.transpose(cp),Dj),Dtc))
            cp = cp + mu*q
            wp = wp + alfa[2]/(den+1e-8)*Dtc*ez
        w7 = wp
        c7 = cp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[6][:,j] = hgs[:,0]

        # Prediction in Neuron 8
        djw = np.matmul(Dj,w8).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        wp = w8
        for ep in range(epoch):
            djw = np.matmul(Dj,wp).reshape(-1,1)
            e = (S[:,j]-djw[:,0]).reshape(-1,1)
            eet = np.matmul(e,np.transpose(e))
            Z8 = 0.9*Z8 + eet
            We = 3*Z8-eet
            Dt = np.transpose(Dj)
            Dte = np.matmul(Dt,e)
            Dze = np.matmul(Dt,np.matmul(We,e))
            DDte = np.matmul(Dj,Dze)
            mud = epoch*(np.matmul(np.transpose(DDte),DDte)+1e-6)
            mun = np.matmul(np.transpose(Dte),Dze)
            mu = thr*mun/mud
            wp = wp +mu*Dze
        w8 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[7][:,j] = hgs[:,0]
        
        # Prediction in Neuron 9
        y1 = np.matmul(Dj,w9)
        y2 = np.matmul(bw9,Dj)
        djw = np.matmul(bw9,y1).reshape(-1,1)
        hatS[:,0] = djw[:,0]
        bwp = bw9
        wp = w9
        for ep in range(epoch):
            y1 = np.matmul(Dj,wp)
            y2 = np.matmul(bwp,Dj)
            djw = np.matmul(bwp,y1).reshape(-1,1)
            e = (S[:,j]-djw[:,0]).reshape(-1,1) 
            y2t = np.transpose(y2)
            q1 = np.matmul(e,np.transpose(y1))
            q2 = np.matmul(y2t,e)
            wp = wp + alfa[3]/epoch*q2
            #mu = 0.0001/60
            bwp = bwp + alfa[3]/epoch*q1
        bw9 = bwp
        w9 = wp
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[8][:,j] = hgs[:,0]
        
        # Prediction in Neuron 1 of Layer 2
        for i in range(M):
            dq = 1
            for l in range(9):
                dq = dq*hatS_all[l][i,j]**a10[i][l,0]
                lgd[l,0] = np.log(hatS_all[l][i,j]+1e-8)
            hatS[i,0] = dq
            ap = a10[i][:,0].reshape(-1,1)
            for ep in range(epoch):
                dq = 1
                for l in range(9):
                    dq = dq*hatS_all[l][i,j]**ap[l,0]
                cd = np.matmul(Z10,lgd)
                eq = np.log((S[i,j]+1e-8)/(dq+1e-8))
                #den = np.sum(np.multiply(cd,lgd))+1e-8
                ng = (cd[:,0]*eq).reshape(-1,1)
                mu = alfa[4]*np.min(ap[:,0])/(np.max(np.abs(ng))+1e-8)
                ap = ap+mu/epoch*ng
            a10[i][:,0] = ap[:,0]
        hgs = np.multiply(hatS,(hatS>0))
        hatS_all[9][:,j] = hgs[:,0]
        
        Dj[:,1:L] = Dj[:,0:L-1]
        Dj[:,0] = S[:,j]

        Dpj[:,1:L] = Dpj[:,0:L-1]
        Dpj[:,0] = S[:,j]-Dpd[:,0]
        Dpd[:,0] = S[:,j]
 
    return(hatS_all)
