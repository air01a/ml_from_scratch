import numpy as np


class Activation:
    def sigma(self,x):
        return 1/(1+np.exp(-x))

    def sigma_prime(self,x):
        return self.sigma(x)(1-self.sigma(x))

    def heavyside(self,x):
        return np.where(x >= 0, 1, 0)
        
    def heavyside_prime(self,x):
        return np.zeros_like(x)

    def relu(self,x):
        return np.maximum(0, x)


    def relu_prime(self,x):
        return np.where(x < 0, 0, 1)

    def sinusoide(self,x):
        return np.sin(self,x)

    def sinusoide_prime(self,x):
        return np.cos(x)


    def cardinalsinusoide(self,x):
        if x==0:
            return 1
        return np.sin(x)/x

    def cardinalsinusoide_prime(self,x):
        if x==0:
            return 0
        return np.cos(x)/x - np.sin(x)/x

    def tanh(self,x):
        return np.tanh(x)

    def tanh_prime(self,x):
        return 1-np.tanh(x)**2

    def id(self,x):
        return x
    
    def id_prime(self,x):
        return 1

    def __init__(self):
        self.activation_function = {'id':self.id,'tanh':self.tanh,'sigma':self.sigma,'heavyside':self.heavyside, 'relu':self.relu,'sinusoide':self.sinusoide,'cardinalsinusoide':self.cardinalsinusoide}
        self.prime_function =  {'id':self.id_prime, 'tanh':self.tanh_prime, 'sigma':self.sigma,'heavyside':self.heavyside, 'relu':self.relu,'sinusoide':self.sinusoide,'cardinalsinusoide':self.cardinalsinusoide}

    def get_activation_function(self, func):
        if func in self.activation_function.keys():
            return self.activation_function[func]
        
    def get_activation_prime_function(self, func):
        if func in self.prime_function.keys():
            return self.prime_function[func]
        
    