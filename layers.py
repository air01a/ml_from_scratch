from abstract import Layer
from numpy import random, dot, zeros, sum, maximum, zeros_like, unravel_index, argmax

from activation import Activation
from scipy import signal
from skimage import measure

class FC(Layer):

    def __init__(self, number_neurons, imput_dim, activation):
        self.activation_function = activation
        self.activation = Activation()
        self.activation_function = self.activation.get_activation_function(activation)
        self.prime_activation_function = self.activation.get_activation_prime_function(activation)
        self.weights = random.rand(imput_dim, number_neurons) - 0.5
        self.activation = activation
        self.number_of_neurons = number_neurons
        self.bias = random.rand(1, number_neurons) - 0.5

    def forward(self, input):
        self.input = input
        output = dot(input, self.weights) + self.bias
        self.activation_function_input=output
        self.output = self.activation_function(output)
        return self.output
    
    def backward(self, output_error, learning_rate):
        current_output_error = self.prime_activation_function(self.activation_function_input)*output_error
        input_error = dot(current_output_error, self.weights.T)
        weights_error = dot(self.input.T, current_output_error)



        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * current_output_error

        return input_error

class CONV(Layer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output_depth
    def __init__(self, input_shape, kernel_shape, layer_depth,activation):
        self.activation_function = activation
        self.activation = Activation()
        self.activation_function = self.activation.get_activation_function(activation)
        self.prime_activation_function = self.activation.get_activation_prime_function(activation)

        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth)
        self.weights = random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = random.rand(layer_depth) - 0.5


    def forward(self, input):
        self.input = input
        self.output = zeros(self.output_shape)
        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k]
        self.activation_function_input=self.output
        return self.activation_function(self.output)


    def backward(self, output_error, learning_rate):

        current_output_error = self.prime_activation_function(self.activation_function_input)*output_error

        in_error = zeros(self.input_shape)
        dWeights = zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(current_output_error[:,:,k], self.weights[:,:,d,k], 'full')
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], current_output_error[:,:,k], 'valid')
            dBias[k] = self.layer_depth * sum(current_output_error[:,:,k])

        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias
        return in_error
    
class FLATTEN(Layer):

    def forward(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)
    
class MAXPOOLING(Layer):

    def __init__(self, pool_size=(2,2)):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        return measure.block_reduce(input, self.pool_size,max)

    def backward(self, output_error, learning_rate):
        input = self.input
        batch_size, input_height, input_width, num_channels = input.shape
        output_height, output_width = output_error.shape[1], output_error.shape[2]
        d_input = zeros_like(input)


        for i in range(output_height):
            for j in range(output_width):
                # Extraire la fenêtre de max-pooling correspondante dans l'entrée
                window = input[:, i*self.pool_size[0]:(i+1)*self.pool_size[0], j*self.pool_size[1]:(j+1)*self.pool_size[1], :]
                
                # Calculer les indices du maximum dans la fenêtre
                max_indices = argmax(window, axis=(1, 2))
                
                # Calculer les gradients en attribuant d_output aux positions des maxima
                for batch in range(batch_size):
                    for channel in range(num_channels):
                        max_index = max_indices[batch, channel]
                        max_row, max_col = unravel_index(max_index, (self.pool_size[0], self.pool_size[1]))
                        d_input[batch, i*self.pool_size[0]+max_row, j*self.pool_size[1]+max_col, channel] = output_error[batch, i, j, channel]
        return d_input


