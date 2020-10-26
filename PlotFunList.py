# Copyright © 2020 Md. Zulfiquar Ali Bhotto, Richard Jones, Stephen Makonin, and Ivan V. Bajic.


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc


def pltEngyPdtn(error, hatS_all,S):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    M,N = hatS_all[0].shape
    r,number = error.shape
    t = np.arange(N*M)
    cmap = plt.get_cmap('gnuplot')
    #colors = [cmap(i) for i in np.linspace(0, 1, number)]
    colors = ['darkgray', 'sienna', 'purple', 'olive', 'cyan','pink','red','green','dodgerblue','blue','coral',
              'turquoise']
    labels = (['UP','LCP','HLP','DLP','RbP','RdP','MMP', 'KbP', 'ALM','GMC','LSTM','MLP'])
    StyleList = [(0, (2,1)), (0,(1,1)), (0, ()),  (0, (4, 4)),   (0, (4, 1)),
               (0, ()),  (0, (5, 4, 1, 4)), (0, (3, 1, 1, 1)), (0, ()),
                (0, ()), (0, (2, 4, 1, 4, 1, 4)),  (0, (3, 1, 1, 1, 1, 1))]
    # List of Dash styles, each as integers in the format:
    #(first line length, first space length, second line length, second space length...)

    fig, (ax) = plt.subplots(1, 1)
    for i in range(10):
        ax.plot(t, 10*np.log10(error[:,i]), color=colors[i], label=labels[i],linewidth=2,linestyle=StyleList[i])
    ax.legend(loc='upper right')
    plt.xlabel('time, Hr')
    plt.ylabel('NSE, dB')
    plt.grid()
    plt.show()

    t = np.arange(M)
    lw = [4,2,2,3,2,6,2,2,2,2,2,2]
    fig, (ax) = plt.subplots(1, 1, sharex=True)
    ax.plot(t,S[:,198],'darkblue',label='Gt',linewidth=4)
    for i in range(12):
        ax.plot(t, hatS_all[i][:,198], color=colors[i], label=labels[i],linewidth=lw[i],linestyle=StyleList[i])
    ax.legend(loc='upper right')
    ax.grid('on')
    plt.xlabel('time, Hr')
    plt.ylabel('Energy')
    plt.show()

    t = np.arange(M)
    fig, (ax) = plt.subplots(1, 1, sharex=True)
    ax.plot(t,S[:,199],'darkblue',label='Gt',linewidth=4)
    for i in range(12):
        ax.plot(t, hatS_all[i][:,199], color=colors[i], label=labels[i],linewidth=lw[i],linestyle=StyleList[i])
    ax.legend(loc='upper right')
    ax.grid('on')
    plt.xlabel('time, Hr')
    plt.ylabel('Energy')
    plt.show()

    t = np.arange(M)
    fig, (ax) = plt.subplots(1, 1, sharex=True)
    ax.plot(t,S[:,200],'darkblue',label='Gt',linewidth=4)
    for i in range(12):
        ax.plot(t, hatS_all[i][:,200], color=colors[i], label=labels[i],linewidth=lw[i],linestyle=StyleList[i])
    ax.legend(loc='upper right')
    ax.grid('on')
    plt.xlabel('time, Hr')
    plt.ylabel('Energy')
    plt.show()

def pltmixa(mixp,N):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    t = np.arange(N)
    #cmap = plt.get_cmap('gnuplot')
    #colors = [cmap(i) for i in np.linspace(0, 1, number)]
    colors = ['darkgray', 'sienna', 'purple', 'olive', 'cyan','pink','red','green','dodgerblue','blue','coral',
              'turquoise']
    labels = (['UP','LCP','HLP','DLP','RbP','RdP','MMP','KbP','ALM','GMC','LSTM','MLP'])
    StyleList = [(0, (2,1)), (0,(1,1)), (0, ()),  (0, (4, 4)),   (0, (4, 1)),
               (0, ()),  (0, (5, 4, 1, 4)), (0, (3, 1, 1, 1)), (0, ()),
                (0, ()), (0, (2, 4, 1, 4, 1, 4)),  (0, (3, 1, 1, 1, 1, 1))]
    lw = [4,2,2,3,2,6,2,2,2,2,2,2]
    fig, (ax) = plt.subplots(1, 1)
    for i in range(9):
        ax.plot(t, mixp[i,:], color=colors[i], label=labels[i],linewidth=lw[i],linestyle=StyleList[i])
    ax.legend(loc='upper right')
    plt.xlabel('time, Hr')
    plt.ylabel('$a_t$')
    plt.grid()
    #plt.savefig('/Users/mbhotto/Documents/LatextDocs/AwesenseReportFig/'+figname % index, dpi=300)
    plt.show()

def pltEngyMad(error, maxst,nday,figname,index):
    t = np.arange(24)
    #cmap = plt.get_cmap('gnuplot')
    #colors = [cmap(i) for i in np.linspace(0, 1, number)]
    colors = ['darkgray', 'sienna', 'purple', 'olive', 'cyan','pink','red','green','dodgerblue','blue','coral',
              'turquoise']
    labels = (['UP','LCP','HLP','DLP','RbP','RdP','MMP','KbP','ALM','GMC','LSTM','MLP'])
    fig, (ax) = plt.subplots(1, 1)
    for i in range(9,12):
        ax.plot(t, 10*np.log10(error[i][:,0]/(nday*maxst)), color=colors[i], label=labels[i],linewidth=2)
    ax.legend(loc='upper right')
    plt.xlabel('time, Hr')
    plt.ylabel('NMSE, dB')
    plt.grid()
    plt.savefig('/Users/mbhotto/Documents/LatextDocs/AwesenseReportFig/'+figname % index, dpi=300)
    #plt.show()
