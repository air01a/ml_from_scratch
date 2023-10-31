import numpy as np

class Loss:
    # loss function and its derivative
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size
    

    def absmean(self, y_true, y_pred):
        return np.mean(np.abs(y_true-y_pred, 2))

    def absmean_prime(self, y_true, y_pred):
        return (y_pred-y_true)/np.abs(y_pred-y_true)/y_true.size
    

    def __init__(self):
        self.loss_function = {'mse':self.mse}
        self.prime_loss = {'mse':self.mse_prime}
    
    def get_loss_function(self, func):
        if func in self.loss_function.keys():
            return self.loss_function[func]
        
    def get_loss_prime_function(self, func):
        if func in self.prime_loss.keys():
            return self.prime_loss[func]
        
    

    