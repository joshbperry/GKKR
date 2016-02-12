import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.kernel_ridge import KernelRidge
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score

#majorLocator = MultipleLocator(100)
#fig, ax = plt.subplots()

f = open("Reaserch_Data_new.csv")
f.readline()

dataTemp =  np.loadtxt(f, delimiter = ',')

data = dataTemp[:, np.newaxis]
XCu = []
XNi = []
XMn = []
XP  = []
XSi = []
XC  = []
XFl = []
XFx = []
XT  = []
Ydata = []

YFl = []

for n in range(len(data[:, :, 0])):
    if data[n, :, 6] < 3*(10**19):
        
        XCu = np.concatenate((XCu, data[n, :, 13]),  axis=0)
        XNi = np.concatenate((XNi, data[n, :, 14]),  axis=0)
        XMn = np.concatenate((XMn, data[n, :, 15]),  axis=0)
        XP  = np.concatenate((XP,  data[n, :, 16]),  axis=0)
        XSi = np.concatenate((XSi, data[n, :, 17]),  axis=0)
        XC  = np.concatenate((XC,  data[n, :, 18]),  axis=0)
        XFl = np.concatenate((XFl, data[n, :, 19]), axis=0)
        XFx = np.concatenate((XFx, data[n, :, 20]), axis=0)
        XT  = np.concatenate((XT,  data[n, :, 21]),  axis=0)
        Ydata = np.concatenate((Ydata, data[n, :, 9]), axis=0)

        YFl = np.concatenate((YFl, data[n, :, 11]), axis=0)

l = len(Ydata)
#print(l)

XCu = np.reshape(XCu, (l,1))
XNi = np.reshape(XNi, (l,1))
XMn = np.reshape(XMn, (l,1))
XP  = np.reshape(XP,  (l,1))
XSi = np.reshape(XSi, (l,1))
XC  = np.reshape(XC,  (l,1))
XFl = np.reshape(XFl, (l,1))
XFx = np.reshape(XFx, (l,1))
XT  = np.reshape(XT,  (l,1))
Ydata = np.reshape(Ydata, (l,1))

Xdata = np.concatenate((XCu, XNi, XMn, XP, XSi, XC, XFl, XFx, XT), axis=1)

Xdata_test = Xdata
Xdata_train = Xdata
Ydata_train = Ydata
Ydata_test = Ydata

ridge = KernelRidge(alpha= 2.68e-6, coef0=1, degree=3, gamma=1.098541, kernel='rbf', kernel_params=None)

ridge.fit(Xdata_train, Ydata_train)


for n in range(len(data[:, :, 0])):
    if np.abs(ridge.predict(Xdata_test[n]) - Ydata_test[n]) >= 35:
        plt.scatter((Ydata_test[n]),(ridge.predict(Xdata_test[n])), color='red')
        print(n)
    if np.abs(ridge.predict(Xdata_test[n]) - Ydata_test[n]) < 35:
        plt.scatter((Ydata_test[n]),(ridge.predict(Xdata_test[n])), color='black')          
        

# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % ridge.score((Xdata_test), Ydata_test))
# calculate rms
rms = np.sqrt(np.mean(((ridge.predict(Xdata_test)) - Ydata_test) ** 2))
mse = np.mean(np.abs((ridge.predict(Xdata_test)) - Ydata_test))
r2 = r2_score(Ydata_test, ridge.predict(Xdata_test))  

print('RMS:', rms, 'MSE:', mse, 'r2', r2)


# Plot outputs
#plt.scatter((Ydata_test),(ridge.predict(Xdata_test)), color='black')
#plt.scatter(YFl,(ridge.predict(Xdata_test)-Ydata_test), color='black')

plt.plot((-50, 700), (-50, 700), color='blue', linewidth=3)
#plt.plot((16.5, 21.5), (20, 20), color='blue', linewidth=1)
#plt.plot((16.5, 21.5), (0, 0), color='blue', linewidth=3)
#plt.plot((16.5, 21.5), (-20, -20), color='blue', linewidth=1)

#ax.xaxis.set_major_locator(majorLocator)
plt.show()

f.close()
