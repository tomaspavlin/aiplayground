import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.random.normal(0, 1, (100))
#data = np.random.rand(10)
##data = np.array([2, 2, 2.1, 5])

#print(data)

def gaussian(x, sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

def kernel_density(x, data, h):
    #print(np.abs(data - x))
    return np.sum(gaussian(np.abs(data - x), h)) / len(data)

def get_data():
    file = open("compound.dat")
    file.readline()

    ret = [float(l.split("\t")[1]) for l in file.readlines()]
    file.close()
    return np.array(ret)

data = get_data()
data = np.log(data + 1)

a = np.linspace(-3, 10, 1000)
#b = gaussian(a, 0.01)
c = [kernel_density(x, data, 0.05) for x in a]



k2, p = stats.normaltest(data)
print("p={}, k1={}".format(p, k2))


#plt.hist(data, bins=50)
plt.plot(a, c)


plt.show()
