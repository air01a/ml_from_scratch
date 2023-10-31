
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd


w = 2
L = []
step = 0.1
epochs = 30

for i in range (epochs):
  w = w - step * 2 * w
  L.append(w * w)

plt.plot(L)
print ("final value of L=", L[-1], ", W=", w)
plt.show()

X = [1.0, 2.0, 3.0]
y = [1.2, 1.8, 3.4]

plt.scatter(X,y)
plt.scatter(X,y)
plt.plot(X[:3], y[:3], 'r-', lw=2)
plt.show()

a = 1
b = 0

def gradient_descent(_X, _y, _learningrate=0.06, _epochs=5):
    trace = pd.DataFrame(columns=['a', 'b', 'mse'])
    X = np.array(_X)
    y = np.array(_y)
    a, b = 0.2, 0.5 
    iter_a, iter_b, mse = [], [], []
    N = len(X) 
    
    for i in range(_epochs):
        delta = y - (a*X + b)
        
        # Updating a and b
        a = a -_learningrate * (-2 * X.dot(delta).sum() / N)
        b = b -_learningrate * (-2 * delta.sum() / N)

        trace = trace._append(pd.DataFrame(data=[[a, b, mean_squared_error(y, (a*X + b))]], 
                                          columns=['a', 'b', 'mse'], 
                                          index=['epoch ' + str(i+1)]))

    return a, b, trace

def displayResult(_a, _b, _trace):
    plt.figure( figsize=(30,5))

    plt.subplot(1, 4, 1)
    plt.grid(True)
    plt.title("Distribution & line result")
    plt.scatter(X,y)
    plt.plot([X[0], X[2]], [_a * X[0] + _b, _a * X[2] + _b], 'r-', lw=2)
    plt.show()
    
    plt.subplot(1, 4, 2)
    plt.title("Iterations (Coeff. a) per epochs")
    plt.plot(_trace['a'])
    plt.show()
    
    plt.subplot(1, 4, 3)
    plt.title("Iterations (Coeff. b) per epochs")
    plt.plot(_trace['b'])
    plt.show()

    plt.subplot(1, 4, 4)
    plt.title("MSE")
    plt.plot(_trace['mse'])

    print (_trace)

a, b, trace = gradient_descent(X, y, _epochs=3)
displayResult(a, b, trace)

a, b, trace = gradient_descent(X, y, _epochs=10)
displayResult(a, b, trace)

a, b, trace = gradient_descent(X, y, _epochs=50)
displayResult(a, b, trace)