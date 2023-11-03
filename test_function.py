import numpy as np

from network import Network
from layers import FC


def f(x):
    y = x**2 / (100*100)
    return y

x_train = np.array([])
y_train = np.array([])
# training data
for x in range(100):
    x_train = np.append(x_train,[[x]])
    y_train = np.append(y_train,[[f(x)]])

print(x_train)
print(y_train)
# network
net = Network('mse')
net.add(FC(10, 1,'tanh'))
net.add(FC(10, 10,'tanh'))
net.add(FC(10, 10,'tanh'))
net.add(FC(10, 10,'tanh'))
net.add(FC(1, 10,'id'))

net.fit(x_train, y_train, epochs=10000, learning_rate=0.0001)

# test
out = net.predict(x_train)
print(out)
for index, i in enumerate(out):
    print(i, y_train[index])
    