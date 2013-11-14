import numpy as np
import itertools
import random
import math

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def logistic(x):
    return np.tanh(x)
logistic.derivative = lambda x : (1-np.square(x))

class NeuralNetwork:

    def __init__(self, weights, bias=None, sigmoid=None):
        self.weights = weights
        if sigmoid is None:
            sigmoid = logistic
        self.sigmoid = sigmoid
        self.bias = bias

    @classmethod
    def fromlayers(cls, sizes, bias=True, sigmoid=None):
        weights = [2*np.matrix(np.random.rand(b,a))-1
                   for a,b in pairwise(sizes)]
        if bias:
            bias = [2*np.matrix(np.random.rand(b,1))-1
                    for a,b in pairwise(sizes)]
            return cls(weights, bias, sigmoid)
        return cls(weights, sigmoid)

    def copy(self):
        return NeuralNetwork(
            [np.matrix(x) for x in self.weights],
            ([np.matrix(x) for x in self.bias]
             if self.bias is not None else None),
            self.sigmoid
            )

    def mutate(self, scale=1):
        for weight_matrix in self.weights:
            weight_matrix += (
                2*scale *
                (np.random.rand(*weight_matrix.shape)-0.5))
        if self.bias is not None:
            for bias in self.bias:
                bias += (
                    2*scale *
                    (np.random.rand(*bias.shape)-0.5))

    def __call__(self, potentials, trace=False):
        potentials = np.matrix(potentials, dtype=np.float64
                               ).reshape(len(potentials), 1)

        if trace:
            trace = [potentials]
        
        for index, weight_matrix in enumerate(self.weights):
            potentials = weight_matrix * potentials
            if self.bias is not None:
                potentials += self.bias[index]
            potentials = self.sigmoid(potentials)
            if trace:
                trace.append(potentials)
            
        if trace:
            return trace
        return potentials

    def __repr__(self):
        return ("weights:\n"+
                "\n".join(map(repr,self.weights))+"\n"
                "bias:\n"+
                "\n".join(map(repr, self.bias)))

def backpropagate(network, tests, iterations=50):
    for _ in range(iterations):
        traces = [(network(test[0], True), test[1]) for test in tests]
        errors =[sum(expected-trace[-1] for trace,expected in traces)]
        #trace = np.sum(trace[0] for trace in traces)
        print("\n".join(map(str,traces)))
        print('error')
        print(errors)
        #print(trace)
        for index, weights in reversed(
            list(enumerate(network.weights))):
            
            print(index)
            error = errors[-1]
            

if True:
    global network
    tests = [((0,0),0),((0,1),1),((1,0),1),((1,1),0)]
    network = NeuralNetwork.fromlayers([2,3,1])
    backpropagate(network, tests, 1)
