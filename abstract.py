class Layer:
    def __init__(self):
        self.activation = None
        self.number_of_neurons = 0
        self.bias = None
        self.weights = None

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError


