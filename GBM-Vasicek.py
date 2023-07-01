import matplotlib.pyplot as plt
import numpy as np
import math

#GBM
S_0, mu, sigma, dt, steps = 100, 0.1, 0.2, 0.01, 50
stockprices= [S_0]*51
stockpricescont = [S_0]*51
phi = np.random.normal(0, 1, 51)
for i in range(1, steps+1):
    stockprices[i] = stockprices[i-1]*(1 + mu*dt + sigma*phi[i]*math.sqrt(dt))
timeaxis = [j*dt for j in range(0,51)]
plt.figure()
plt.plot(timeaxis, stockprices)
plt.title('Stock price evolution')
plt.xlabel('time')
plt.ylabel('stock price')
plt.show()


#Vasicek
interestrate_0, reversionrate, meanrate = 0.04, 2, 0.05
interestrates = [interestrate_0]*51
for i in range(1, steps+1):
    interestrates[i] = (interestrates[i-1] + reversionrate*(meanrate -
                        interestrates[i-1]) + sigma*phi[i]*math.sqrt(dt))
plt.figure()
plt.plot(timeaxis, interestrates)
plt.title('interest rate evolution')
plt.xlabel('time')
plt.ylabel('interest rate')
plt.show()
