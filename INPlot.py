import numpy as np
import matplotlib.pyplot as plt
import random,os

from math import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


##  Imports from files of the form:
##  Avg
##  ##,##,##
##  Al
##  ##,##,##
##  Cu
##  ##,##,##
##  ...
##  AvgSD
##  ##,##,##
##  ALSD
##  ##,##,##
##  ...
##
##  Where each row is a label with the row following it a series of #'s that
##  correspond to the rms average value found for the remaining impurities when
##  increasing humber of host-impurity pairs are placed into the training set.
##  So the first number is the average RMS with 0 impurities, followed by
##  average rms with 1 impurity and so on.
##  Immediately following the last RMS values the same pattern is followed with
##  the data's standard devitaion values in the same order as the rms.
##
##  Note: All rows of numbers must include the same number of values and each
##  each row must have a label (or be a blank line)
##
##
##  Only needed change is to change the global variable FILENAME to the name of
##  of the file containing the data



FILENAME = "INResults.txt"
FIGROW = 1
FIGCOL = 2

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


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

def setAxes(x,y,s1="Title",s2="X data",s3= "Ydata",legend = True):
    plt.title(s1)
    ax = plt.gca()
    plt.xlim(min(x)-0.5,max(x)+0.5)  #set axis limits
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

def INPlot():
    data = []
    testel = []
    f = open(FILENAME,"r")
    for line in f:
        test = ["Invalid"]
        entries = line
        if "," in entries:
                test = entries.split(",")
                testfloats = []
                if(isFloat(test[0])):
                    for l in range(len(test)):
                        testfloats.append(float(test[l]))
                    data.append(testfloats)
                else:
                    testel.append(test)
    f.close()
    allaverms = data[0]
    alaverms = data[1]
    cuaverms = data[2]
    niaverms = data[3]
    pdaverms = data[4]
    ptaverms = data[5]
    
    ##Changes start here (12/10/15)
    
    allstd = data[6]
    alstd = data[7]
    custd = data[8]
    nistd = data[9]
    pdstd = data[10]
    ptstd = data[11]
    x = np.arange(0,len(data[0]),1)
    SetRowColumn(1,1)
    LON(x,allaverms,alaverms,cuaverms,niaverms,pdaverms,ptaverms,
        allstd,alstd,custd,nistd,pdstd,ptstd)
    setAxes(x,list([max(allaverms),min(allaverms)]),
            "Average RMS vs Impurities Trained On",
            "Number Of Impurities Trained on","RMS")
    plt.show()

if __name__ == '__main__':
    INPlot()
