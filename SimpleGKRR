import FitImport as imp
import numpy as np
from math import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

#Set the grid over which the GKRR is performed to find best parameters
#Tests the two parameters over the range of 10^-6 to 10^3 (alpha) and
#10^-10 to 10^0 (gamma).Does numfolds of cross validation
def setBestParameters(length,npts=25,numfolds=3):
    if numfolds > length:
        numfolds = length-2
    return GridSearchCV(KernelRidge(kernel='rbf'),
                          param_grid={"alpha": np.logspace(-6,3,npts),
                                      "gamma": np.logspace(-10,0,npts)},
                            cv=numfolds)

#Returns the Root Mean Square Error between the two values
def GetRMSE(Y,YP):
    return sqrt(mean_squared_error(Y,YP))

#Runs GKRR I times selecting a new training/testing split each time to test on
#with 80% being used to train and 20% being used to test. The rms are all saved
#to an array called allrms.
def SimpleGKRR(X,Y,I):
    allrms = []
    for reps in range(I):
        Xt,XT,Yt,YT = cross_validation.train_test_split(X, Y, test_size = 0.2)
        regr = setBestParameters(len(Xt))
        regr.fit(Xt,Yt)
        YTP = regr.predict(XT)
        rms = GetRMSE(YT,YTP)
        allrms.append(rms)
        print("Run "+str(reps+1)+"/"+str(I)+" complete!")
    print();
    for tests in range(I):
        print("Run "+str(tests+1)+" had rms "+str(allrms[tests]))


if __name__ == '__main__':

#Do something here to get your X and Y data where X is a 2D array with each
#column being a descriptor and each row being a data point and Y is the a
#column array with a value which is the target value for each data point.
#For example:   A + B = C
#
#       X:              Y:
#       1   1           2
#       1   2           3
#       2   1           3
#       1   3           4

#Sample Data
##    X = [[1,1],[1,2],[2,1],[1,3]]
##    Y = [[2],[3],[3],[4]]
    SimpleGKRR(X,Y,5)
