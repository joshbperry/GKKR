import FitImport as imp
import matplotlib.pyplot as plt
import pylab
import numpy as np
from math import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

error = 0.5

#Normalize input data
def normalizeData(X):
    for m in range(len(X[0])):
        ma = max(X[:,m])
        mi = min(X[:,m])
        for n in range(len(X)):
            if ma != mi: X[n][m] = ((X[n][m] - mi)/(ma-mi))
            else: X[n][m] = 0.5
    return X


#Set the grid over which the GKRR is performed
def setBestParameters(L,ker=0,npts=25,f=3):
    if f > L:
        f = L-2
    return GridSearchCV(KernelRidge(kernel='rbf'),
                          param_grid={"alpha": np.logspace(-6,3,npts),
                                      "gamma": np.logspace(-10,0,npts)}, cv=f)

# Several methods for Calculating and printing the error in predictions
def GetErrorInY(Y,YP,p=True):
    Yer = Y-YP
    Yme = np.mean(abs(Yer))
    Yma = max(abs(Yer))
    Ymi = min(abs(Yer))
    if p == True:
        print("The mean error in Y:    ",Yme)
        print("The max error in Y:     ",str(float(Yma)))
        print("The min error in Y:     ",str(float(Ymi)))
    return Yer, Yme, Yma, Ymi

def GetRMSE(Y,YP):
    return sqrt(mean_squared_error(Y,YP))

def GetAllPredictions(X,Xt,XT,regr):
    return regr.predict(X), regr.predict(Xt), regr.predict(XT)


##Create a Model and get RMS and predictions
def createModelAllData(regr,X,Y):
    regr.fit(X,Y)
    YP = regr.predict(X)
    rms = GetRMSE(Y,YP)
    return regr,YP,rms


#Plotting methods
def SetRowColumn(r,c):
    global FIGROW
    global FIGCOL
    FIGROW = r
    FIGCOL = c
    return

def simplemultiDPlot(Y,Yt,Ytp,YT,YTP,r=2,c=2,i=1,j=2,k=3):
    SetRowColumn(r,c)
    errorPlot(Yt, Ytp, dt = 1,  n=i, s = "calculated vs Y actual(Training)",
              xl='Actual Values',yl="Calclulated Values")
    errorPlot(YT, YTP, dt = 2, n=j, s="Y calculated vs Y actual(Testing)",
              xl='Actual Values',yl="Calclulated Values")
    errorPlotCombined(Yt,Ytp,YT,YTP,n=k,s="Y calculated vs Y actual(Combined)",
              xl='Actual Values',yl="Calclulated Values")

def errorPlot(Y,Yp,dt = 0,n=1,s="Y calculated vs Y actual",
              xl='Actual Values',yl="Calclulated Values",legend = True):
    truemin = min(min(Y),min(Yp))
    truemax = max(max(Y),max(Yp))
    if truemin < 0: T = np.linspace(1.1*truemin,1.1*truemax)
    else:           T = np.linspace(truemin-1.1*truemin,1.1*truemax)
    
    plt.subplot(FIGROW,FIGCOL,n)
    plt.grid()

    if dt== 1:      plt.plot(Y,Yp,'og',label = "Train Data")
    elif dt == 2:   plt.plot(Y,Yp,'or',label = "Test Data")
    else:           plt.plot(Y,Yp,'o')

    plt.plot(T,T,'-',color='b', label = "± 0 units")
    plt.plot(T,T+error,'--b', label = "± "+str(error)+" unit")
    plt.plot(T,T-error,'--b')
    setAxes(Y,Yp,s,xl,yl,legend)
    pylab.xlim(xmin= truemin-0.1*abs(truemin), xmax = 1.1 * truemax)
    pylab.ylim(ymin= truemin-0.1*abs(truemin), ymax = 1.1 * truemax)
    return

def errorPlotCombined(Y1,Yp1,Y2,Yp2, n=1,s="Y calculated vs Y actual",
              xl='Actual Values',yl="Calclulated Values",legend = True):
    truemin = min(min(min(Y1),min(Yp1)),min(min(Y2),min(Yp2)))
    truemax = max(max(max(Y1),max(Yp1)),max(max(Y2),max(Yp2)))
    if truemin < 0: T = np.linspace(1.1*truemin,1.1*truemax)
    else:        T = np.linspace(truemin-1.1*truemin,1.1*truemax)
    
    
    plt.subplot(FIGROW,FIGCOL,n)

    plt.plot(Y1,Yp1,'og',label = "Train Data")
    plt.plot(Y2,Yp2,'or',label = "Test Data")

    plt.plot(T,T,'-',color='b', label = "± 0 units")
    plt.plot(T,T+error,'--b', label = "± "+str(error)+" unit")
    plt.plot(T,T-error,'--b')
    setAxes([min(Y1),min(Y2)],[min(Yp1),min(Yp2)],s,xl,yl,legend)
    pylab.xlim(xmin= truemin-0.1*truemin, xmax = 1.1 * truemax)
    pylab.ylim(ymin= truemin-0.1*truemin, ymax = 1.1 * truemax)
    plt.grid()
    return

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


##Main Methods

def OfficialTest(I=50,npts=25, s=1):
        X,Y,L = imp.FullImport()    ##Get Data in usable format
        X = normalizeData(X)        ##Normalize the data
        AllMse = []                 ##Holds the calculated RMS error
        AllXt = []                  ##Holds the Xt sets
        AllYt = []                  ##Holds the Yt sets
        AllXT = []                  ##Holds the XT sets
        AllYT = []                  ##Holds the YT sets
        maxError = []               ##Holds the max eror for each sets

        #Loops I times, splits data into training and testing with training being
        #80%, then fits the data and calculates the RMS error saving values to the
        #arrays listed above
        
        for n in range(I):
            if int(n+1)%10 == 0:    print("iteration "+str(n+1)+"/"+str(I))

            ##GKRR-specific Code Starts Here
            Xt,XT,Yt,YT = cross_validation.train_test_split(X, Y, test_size = 0.2)
            regr = setBestParameters(len(Xt),0,npts)
            regr.fit(Xt,Yt)
            YTP = regr.predict(XT)
            rms = sqrt(mean_squared_error(YT,YTP))
            _,_,ymax,_ = GetErrorInY(YT,YTP,False)
            ##GKRR-specific Code Ends Here
            
            AllMse.append(rms)
            AllXt.append(Xt)
            AllYt.append(Yt)
            AllXT.append(XT)
            AllYT.append(YT)
            maxError.append(ymax)

        #Finds the max RMS error and recreates model from dataset at the correct
        # position (m). Prints the fit information
        MMSE = max(AllMse)
        m = AllMse.index(MMSE)
        regr = setBestParameters(len(Xt),0,npts)
        Mregr,_,_ = createModelAllData(regr,AllXt[m],AllYt[m])
        print("\nThe maximum RMS is      ", MMSE)
        MYP, MYtp, MYTP = GetAllPredictions(X,AllXt[m],AllXT[m],Mregr)
        GetErrorInY(Y,MYP)
        print("The alpha parameter is ",str(float(Mregr.best_params_['alpha'])))
        print("The gamma parameter is ",str(float(Mregr.best_params_['gamma'])))    

        #Finds the min RMS error and recreates model from dataset at the correct
        # position (n). Prints the fit information
        nMSE = min(AllMse)
        n = AllMse.index(nMSE)
        regr = setBestParameters(len(Xt),0,npts)
        nregr,_,_ = createModelAllData(regr,AllXt[n],AllYt[n])
        print("\nThe minimum RMS is      ", nMSE)
        nYP, nYtp, nYTP = GetAllPredictions(X,AllXt[n],AllXT[n],nregr)
        GetErrorInY(Y,nYP)
        print("The alpha parameter is ",str(float(nregr.best_params_['alpha'])))
        print("The gamma parameter is ",str(float(nregr.best_params_['gamma'])))

        #Calculates average RMS and max Error
        print("\nThe average RMS:        ",np.mean(AllMse),"±",np.std(AllMse))
        AME = np.mean(maxError)
        print("The average max error:  ",AME)
        MME = max(maxError)
        print("The maximum error:      ",str(float(MME)))

        #Takes care of plotting the best and worst fits
        fig = plt.figure()
        simplemultiDPlot(Y,AllYt[m],MYtp,AllYT[m],MYTP,2,3,1,2,3)
        simplemultiDPlot(Y,AllYt[n],nYtp,AllYT[n],nYTP,2,3,4,5,6)
        ax = fig.add_axes([0,0,1,1])
        plt.text(0.07, 0.725, "Worst Fit", ha='center', va='center',
                 rotation = 'vertical', size = 18,transform=ax.transAxes)
        plt.text(0.07, 0.275, "Best Fit", ha='center', va='center',
                 rotation = 'vertical', size = 18, transform=ax.transAxes)
        ax.set_axis_off()
        plt.show()


#The input variable is the number of iterations to run (usually 200)
if __name__ == '__main__':
    OfficialTest(50); 
