import sys
import numpy as np

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def importData(nameOfFile):
    entries = []
    allData = []
    labels = []
    try:
        f = open(nameOfFile,"r")
        for line in f:
            entries = line.split(",")
            allData.append(entries)
        f.close()
    except FileNotFoundError:
        print("Import Error: FILE",nameOfFile," DOES NOT EXIST")
        sys.exit()
    print("Import Successful")
    labels = getLabels(allData)
    allData = convertToArray(allData)
    return allData, labels

#-----DATA IMPORT METHODS----------#

def convertToArray(temp):
    data = makeEven(temp)
    i = len(data)
    j = len(data[0])
    new = np.zeros((i,j))
    try:
        for n in range(i):
            for m in range(j):
                new[n][m] = data[n][m]
    except ValueError:
        print("Import Error: Non-number Value Found In Data")
        print(data[n][m],"is at",n," ",m)
        sys.exit()
    return new


def makeEven(temp):
    temp = removeLabels(temp)
    minlen = len(temp[0])
    for n in range(len(temp)):
        templen = len(temp[n])
        if templen < minlen:
            minlen = templen
    for n in range(len(temp)):
        if len(temp[n]) > minlen:
            for m in range(len(temp[n]) - minlen):
                temp[n].pop(-1)
    return temp
                

def removeLabels(temp):
    if isFloat(temp[0][0]) == False:
        if isFloat(temp[0][1]) == False:
            temp.pop(0)
            temp = removeLabels(temp)
        else:
            if isFloat(temp[1][0]) == False:
                for n in range(len(temp)):
                    temp[n].pop(0)
    return temp

def getX(data):
    x = np.zeros((len(data),len(data[0])-1))
    for n in range(0,len(data)):
        for i in range(0,len(data[0])-1):
            x[n][i] = data[n][i]
    return x


def getY(data):
    y = np.zeros((len(data),1))
    for m in range(0,len(data)):
        y[m][0] = data[m][-1]
    return y

#-------LABEL IMPORT METHODS-----------#

def getLabels(data):
    xlabel = getXLabel(data)
    temp,col = getYLabel(data)
    if xlabel[0] != "no labels":
        for n in range(col):
            xlabel[0].pop(0)
    ylabel = []
    count = temp[0].count("-")
    if count >= 1:
        for n in range(len(temp)):
            ylabel.append(temp[n].split("-"))
        return list(xlabel+[ylabel])
    else:
        ylabel = temp
        return list(xlabel+[ylabel])
    

def getXLabel(data,labels = []):
    nonums = True
    if isFloat(data[0][0]) == False:
        for n in range(len(data[0])):
            if isFloat(data[0][n]) == True:
                nonums = False
        if nonums:
            labels[:] = []
            labels.append(data[0])
            data.pop(0)
            labels = getXLabel(data,labels)
    if labels == []:
        labels = ["no labels"]
    return labels

def getYLabel(data,labels = [],col = 0):
    nonums = True
    if isFloat(data[0][0]) == False:
        for n in range(len(data)):
            if isFloat(data[n][0]) == True:
                nonums = False
        if nonums:
            labels[:] = []
            for n in range(len(data)):
                labels.append(data[n][0])
                data[n].pop(0)
            col += 1
            labels,col = getYLabel(data,labels,col)
    if labels == []:
        labels = ["no labels"]
    return labels, col


#-----OVERALLL IMPORT METHOD---------#

def FullImport(): ##Path to Datasets
    data,labels = importData(input("Input Training Data File Name:\n"))
    return getX(data), getY(data), labels
