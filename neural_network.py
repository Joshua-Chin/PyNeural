import random
import numpy as np

class network():

    def __init__(self,
                 input_neurons,
                 hidden_neurons,
                 output_neurons):
        self.input_size = input_neurons
        self.hidden_size = hidden_neurons
        self.output_size = output_neurons
        self.weights1 = np.random.rand(input_neurons, hidden_neurons)
        self.weights2 = np.random.rand(hidden_neurons, output_neurons)

    def activate(self, input, debug=True):
        if len(input) != self.input_size:
            raise ValueError('length of input must equal %s'%self.input)
        input = np.array(input, dtype=np.float64)
        hidden = self.sigmoid(np.sum(self.weights1 * input.reshape(self.input_size, 1), axis=0)/self.input_size)
        out = self.sigmoid(np.sum(self.weights2 * hidden.reshape(self.hidden_size, 1), axis=0)/self.hidden_size)
        if debug:
            self.input = input
            self.hidden = hidden
            self.out = out
        return out

    def copy(self):
        return object.__new__

    def test(self, input, expected_out):
        return self.error(self.activate(input))

    def error(self, array):
        return np.sqrt(np.sum(np.square(array)))
    
    def sigmoid(self, array):
        return 1 / (1 + np.exp(-array))
        
    def randomize_weights(self):
        self.weights1 = np.random.rand(self.weights1.shape)
        self.weights2 = np.random.rand(self.weights2.shape)
