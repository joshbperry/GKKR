import FitImport as imp
import numpy as np
from math import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

GSFOLDS = 3
FOLDS = 5
NPTS = 25


#Method Select function that allows you tyo use a differnet learning method
# (NN,decision trees, etc) Simply replace GKRR Methods section with functions
#that run your method and have method select return the rms. MethodSelect
#provides the X to test in TestSet and the target values in Y.

def MethodSelect(TestSet,Y):
    return SplitFitGKRR(TestSet,Y)

##--------GKRR METHODS---------#

def GetPrediction(X,regr):
    return regr.predict(X)

def GetRMSE(Y,YP):
    return sqrt(mean_squared_error(Y,YP))

def SplitFitGKRR(X,Y):
    Xt,XT,Yt,YT = cross_validation.train_test_split(
        X, Y, test_size = 0.2)
    regr = setBestParameters(len(Xt))
    regr.fit(Xt,Yt)
    return GetRMSE(YT,GetPrediction(XT,regr))

def setBestParameters(L,ker=0,npts=NPTS,f=GSFOLDS):
    if f > L:
        f = L-2
    return GridSearchCV(KernelRidge(kernel='rbf'), cv=f,
                          param_grid={"alpha": np.logspace(-6,3,npts),
                                      "gamma": np.logspace(-10,0,npts)})


#--------Descriptor Methods---------#

#Gets Descriptor from position n and returns it.
#If a label Array from fitimport is passed in as well gets grabs the label that
#corresponds to the descriptor returned 

def GODesc(X,n,label=None):
    if label != None:
        L = label[0][n]
    else:
        L = "None"
    X = X[:,n]    
    return list(X),L



##---Main-----##

def FWDS(i = 50):
    X,Y,L = imp.FullImport(1)
    T = len(X[0])       #Number Of Descriptors
    BestDesc = []       #Holds Best Descriptors
    
    while ((len(BestDesc) != 2) and len(BestDesc) < T) :
        #continue until found 30 Best Descriptors or until no more Descr to add
        D = np.arange(T)
        TestRMS = []
        
        for z in range(T):                  #Test Each descriptor
            #print("Testing Descr ",z,"",L[0][z])
            TestSet = []
            setRMS = []
            for n in range(len(BestDesc)):  #Add already known bests
                desc,_ = GODesc(X,BestDesc[n])
                TestSet.append(desc)
            
            tryfit = True          #add step new descriptors?
            if BestDesc.count(z) == 0:  #Test if already one of best before try
                desc,_ = GODesc(X,z)
                TestSet.append(desc)
            else: tryfit = False

            print(TestSet)
                
            if tryfit == True:              #If not best, test with current best
                TestSet = np.swapaxes(TestSet,0,1)
                for n in range(i):
                    rms = MethodSelect(TestSet,Y)
                    setRMS.append(rms)
                TestRMS.append(np.mean(setRMS))
                print("Descr ",z,"",L[0][z]+" has rms",np.mean(setRMS))
                
            else:                           #Skip if already a best descriptor
                TestRMS.append(10)
                ##print("Descr ",z,"",L[0][z]+" already added")

        #Sorts and prints sorted list by worst Descriptor
        print("\n");
        TestRMS, D = (list(t) for t in zip(*sorted(zip(TestRMS, D))))
        for num in range(len(TestRMS)):
            print("Descr ",D[num],"",L[0][D[num]]+" has rms",TestRMS[num])

        #Add best Descriptor to list
        BestDesc.append(D[t])
        print("Desc",D[t],"("+str(L[0][D[t]])+
              ") added as descriptor number",len(BestDesc),"\n\n")

    #When done print best Descriptors in order
    print("Best Descs are:")
    for n in range(len(BestDesc)-1):
        print(L[0][BestDesc[n]])

#FWS take 1 argument which is the number of iterations to test each Descriptor
#each time will be with a different training and testing set.

if __name__ == '__main__':
    FWS(1)
