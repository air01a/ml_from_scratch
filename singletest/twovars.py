import numpy as np
import matplotlib.pyplot as plt

n=100
F = "np.sin(np.sqrt(X**2 + Y ** 2))"


def f(X,Y):
    return eval(F)


VX = np.linspace(-5,5,n)
VY = np.linspace(5,-5,n)
X,Y = np.meshgrid(VX, VY)

Z = f(X,Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(40,-30)
ax.plot_surface(X,Y,Z)
plt.title(F)
plt.show()


plt.title("F(x,y)=cste")
plt.contour(X,Y,Z,30)
plt.show()