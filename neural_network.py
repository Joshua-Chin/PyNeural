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

    def __call__(self, potentials):
        potentials = np.matrix(potentials, dtype=np.float64
                               ).reshape(len(potentials), 1)

        trace = []
        trace.append(potentials)
        
        for index, weight_matrix in enumerate(self.weights):
            potentials = weight_matrix * potentials
            if self.bias is not None:
                potentials += self.bias[index]
            potentials = self.sigmoid(potentials)
            trace.append(potentials)
            
        self.trace = trace
        return potentials

    def __repr__(self):
        return ("weights:\n"+
                "\n".join(map(repr,self.weights))+"\n"
                "bias:\n"+
                "\n".join(map(repr, self.bias)))
    
def train(network, tests, iterations=50):
    def error(model):
        return sum(rms(model(potentials), output)
                   for potentials, output in tests
                   )/len(tests)
    temp = anneal(network, error, iterations)
    print(error(temp)); return temp

def rms(actual, expected):
    return np.sum(np.square(expected - actual))

def anneal(model, error, iterations=50):

    best = model.copy()
    
    min_err = error(model)
    prev_err = min_err
    
    for i in range(1, iterations):
        neighbor = model.copy()
        neighbor.mutate()
        err = error(neighbor)
        if err < prev_err or accept(err-prev_err, i):
            model, prev_err = neighbor, err
            if err < min_err:
                best, min_err = neighbor, err

    return best

def accept(delta, i):
    return math.exp(-delta/.95**i) > random.random()

if __name__ == '__main__':
    global x
    x = NeuralNetwork.fromlayers([2,5,1])
    x = train(x, [((1,1),0), ((0,0),0), ((1,0), 1), ((0,1),1)], 500)
    print([x([a,b]) for a in [0,1] for b in [0,1]])
