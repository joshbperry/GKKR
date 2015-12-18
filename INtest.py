import FitImport as imp
import numpy as np
import matplotlib.pyplot as plt
import random

from math import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error

FIGROW = 1
FIGCOL = 2
error = 0.5
GSFOLDS = 3
FOLDS = 5
NPTS = 25


def MethodSelect(Xt,XT,Yt,YT):
    return FitGKRR(Xt,XT,Yt,YT)

#calls the functions in the GKRR Functions section to calculate rms with GKRR.
#To use a different algorithm (Linear regresion, NN, etc) with this test,
# replace this function and the GKRR methods section with a section that
#performs your operation once and returns the rms.


##--------GKRR Functions---------#

def FitGKRR(Xt,XT,Yt,YT):
    regr = setBestParameters(len(Xt))
    regr.fit(Xt,Yt)
    return GetRMSE(YT,GetPrediction(XT,regr))

def setBestParameters(L,ker=0,npts=NPTS,f=GSFOLDS):
    if f > L:
        f = L-2
    return GridSearchCV(KernelRidge(kernel='rbf'), cv=f,
                          param_grid={"alpha": np.logspace(-6,3,npts),
                                      "gamma": np.logspace(-10,0,npts)})

def GetPrediction(X,regr):
    return regr.predict(X)

def GetRMSE(Y,YP):
    return sqrt(mean_squared_error(Y,YP))

##--------Data Methods---------#

def normalizeData(X):
    for m in range(len(X[0])):
        ma = max(X[:,m])
        mi = min(X[:,m])
        for n in range(len(X)):
            if ma != mi: X[n][m] = ((X[n][m] - mi)/(ma-mi))
            else: X[n][m] = 0.5
    return X

def LONHostElements(lo,X,Y,label,q):
    loindex = []
    lolabel = []
    Itlabel = []
    retelements = []
    count = 0
    found = len(label[1])
    for n in range(len(label[1])):
        if label[1][n][0] == lo:
            loindex.append(n)
            lolabel.append(label[1][n][1])
            count += 1
            if n < found:
                found = n
    if count == 0:
        print("Element Not Found")
        sys.exit()
    for n in range(q):
        ret = random.randint(0,len(loindex)-1)
        loindex.pop(ret)
        l = lolabel.pop(ret)
        Itlabel.append(l)
    XT = np.array((X[loindex]))
    YT = np.array((Y[loindex]))
    Xtemp = []
    Ytemp = []
    for n in range(len(X)):
            if loindex.count(n) != 1:
                  Xtemp.append(X[n])
                  Ytemp.append(Y[n])
    Xt = np.array([[float(0) for n in range(len(X[0]))]
                   for n in range(len(X) - len(loindex))])
    Yt = np.array([[float(0) for n in range(1)]
                   for n in range(len(Y) - len(loindex))])
    for m in range(len(Xtemp)):
        Yt[m] = Ytemp[m]
        for n in range(len(Xtemp[0])):
            Xt[m][n] = Xtemp[m][n]
    return Xt,XT,Yt,YT,lolabel,Itlabel,found

##--------Plot Methods---------#

def SetRowColumn(r,c):
    global FIGROW
    global FIGCOL
    FIGROW = r
    FIGCOL = c
    return

def LON(x,ave,al,cu,ni,pd,pt,aveSD,alSD,cuSD,niSD,pdSD,ptSD):
    M = []
    m = []

    for n in range(len(x)):
        M.append(max(ave[n],al[n],cu[n],ni[n],pd[n],pt[n]))
        m.append(min(ave[n],al[n],cu[n],ni[n],pd[n],pt[n]))
   
    plt.errorbar(x,ave,aveSD,fmt='none',ecolor='gold')
    plt.plot(x,ave,'gold',label="Average")
    plt.errorbar(x,al,alSD,fmt='none',ecolor='r')
    plt.plot(x,al,'r',label = "Al")
    plt.errorbar(x,cu,cuSD,fmt='none',ecolor='b')
    plt.plot(x,cu,'b',label = "Cu")
    plt.errorbar(x,ni,niSD,fmt='none',ecolor='g')
    plt.plot(x,ni,'g',label = "Ni")
    plt.errorbar(x,pd,pdSD,fmt='none',ecolor='orange')
    plt.plot(x,pd,'orange',label = "Pd")
    plt.errorbar(x,pt,ptSD,fmt='none',ecolor='purple')
    plt.plot(x,pt,'purple',label = "Pt")
    plt.grid()
    setAxes(x,list([max(M),min(m)]),"Average RMS vs Impurities Trained On",
            "Number Of Impurities Trained on","RMS")


def setAxes(x,y,s1="Title",s2="X data",s3= "Ydata",legend = True):
    plt.title(s1)
    ax = plt.gca()
    if min(y) < 0: #set x-axis
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('data',0))
        plt.text(0.5, -0.025, s2, ha='center', va='center',
                 transform=ax.transAxes)
    else:
        plt.text(0.5, -0.1, s2, ha='center', va='center',
                 transform=ax.transAxes)

    if min(x) < 0: #set y-axis
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_position(('data',0))
        plt.text(-0.05, 0.5, s3, ha='center', va='center',
                 rotation = 'vertical', transform=ax.transAxes)
    else:
        plt.text(-0.12, 0.5, s3, ha='center', va='center',
                 rotation = 'vertical', transform=ax.transAxes)
    if legend: plt.legend(loc=0)
    return

##--------Main Method---------#
#For a different method simply replace MethodSelect with a different function
# that calculates and returns rms error with the desired Method.


def INtest(n,r):
    lo = ['Al','Cu','Ni','Pd','Pt']
    testel = 5
    testn = n+1
    reps = r
    allrms = []
    X,Y,label = imp.FullImport()
    X = normalizeData(X)
    for n in range(testn):
        loNrms = []
        for m in range(testel):
            host = lo[m]
            for o in range(reps):
                Xt,XT,Yt,YT,_,_,_ = LONHostElements(host,X,Y,label,n)
                
                rms = MethodSelect(Xt,XT,Yt,YT)
                
                loNrms.append(rms)
        allrms.append(loNrms)
        print("completed",n,"impurities included")
        
    allaverms = []
    alaverms = []
    cuaverms = []
    niaverms = []
    pdaverms = []
    ptaverms = []
    
#changes start here

    allstd = []
    alstd = []
    custd = []
    nistd = []
    pdstd = []
    ptstd = []
    for n in range(len(allrms)):
        allaverms.append(round(np.mean(allrms[n]),3))
        allstd.append(round(np.std(allrms[n]),3))
        alaverms.append(np.mean(allrms[n][0:reps]))
        alstd.append(np.std(allrms[n][0:reps]))
        cuaverms.append(np.mean(allrms[n][reps:2*reps]))
        custd.append(np.std(allrms[n][reps:2*reps]))
        niaverms.append(np.mean(allrms[n][2*reps:3*reps]))
        nistd.append(np.std(allrms[n][2*reps:3*reps]))
        pdaverms.append(np.mean(allrms[n][3*reps:4*reps]))
        pdstd.append(np.std(allrms[n][3*reps:4*reps]))
        ptaverms.append(np.mean(allrms[n][4*reps:5*reps]))
        ptstd.append(np.std(allrms[n][4*reps:5*reps]))
    x = np.arange(0,testn,1)
    SetRowColumn(1,1)
    LON(x,allaverms,alaverms,cuaverms,niaverms,pdaverms,ptaverms,
        allstd,alstd,custd,nistd,pdstd,ptstd)
    setAxes(x,list([max(allaverms),min(allaverms)]),
            "Average RMS vs Impurities Trained On",
            "Number Of Impurities Trained on","RMS")
    print()
    print("Tested Elements: ",testel,
          "\nHighest Number of Impurities Included: ",testn-1,
          "\nIterations per number included per element: ",reps)
    print("\nAvg")
    for n in range(len(allaverms)):
        print(allaverms[n]," ± ",allstd[n])
    print("\nAl")
    for n in range(len(allaverms)):
        print(alaverms[n]," ± ",alstd[n])
    print("\nCu")
    for n in range(len(allaverms)):
        print(cuaverms[n]," ± ",custd[n])
    print("\nNi")
    for n in range(len(allaverms)):
        print(niaverms[n]," ± ",nistd[n])
    print("\nPd")
    for n in range(len(allaverms)):
        print(pdaverms[n]," ± ",pdstd[n])
    print("\nPt")
    for n in range(len(allaverms)):
        print(ptaverms[n]," ± ",ptstd[n])
    plt.show()    


# n = Highest number of Host-Impuirties to include (9 in official test)
# r = iterations per Host per impurities included (50 in official test)

if __name__ == '__main__':
    INtest(5,5)

