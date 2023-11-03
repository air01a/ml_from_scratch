import numpy as np

from network import Network
from layers import FC


# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network('mse')
net.add(FC(3, 2,'relu'))
net.add(FC(1, 3,'id'))

net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)