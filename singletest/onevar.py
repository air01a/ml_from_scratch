import numpy as np
import matplotlib.pyplot as plt


def f(X):
    return 10*np.cos(X)

def g(X):
    return 10 * np.exp(-X ** 2)

X = np.linspace(-10,10,num=400)
Y = f(X)
Z = g(X)
plt.plot(X,Y)
plt.plot(X,Z)


plt.title("Cos and gauss")
plt.axis('equal')
plt.grid()
plt.show()