import numpy as np

def backpropagate(network, tests, iterations=500, step_size=.05, momentum=.01, debug=False):

    #convert tests into numpy matrices
    tests = [(np.matrix(inputs, dtype=np.float64).reshape(len(inputs), 1),
            np.matrix(expected, dtype=np.float64).reshape(len(expected), 1))
            for inputs, expected in tests]

    #keep track of old values for momentum
    weight_delta_old = [np.zeros(matrix.shape) for matrix in network.weights]
    bias_delta_old = [np.zeros(matrix.shape) for matrix in network.bias]
    
    for epoch in range(iterations):

        #accumulate the weight and bias deltas
        weight_delta = [np.zeros(matrix.shape) for matrix in network.weights]
        bias_delta = [np.zeros(matrix.shape) for matrix in network.bias]

        #optional error counter
        error_counter = 0
        
        #iterate over the tests
        for potentials, expected in tests:

            #input the potentials into the network
            #calling the network with trace == True returns a list of matrices,
            #representing the potentials of each layer 
            trace = network(potentials, trace=True)
            errors = [expected - trace[-1]]

            if debug:
                error_counter += float(np.sum(np.abs(errors)))
            
            #iterate over the layers backwards
            for weight_matrix, layer in reversed(list(zip(network.weights, trace))):
                #compute the error vector for a layer
                errors.append(np.multiply(weight_matrix.transpose()*errors[-1],
                                          network.sigmoid.derivative(layer)))
            
            #remove the input layer
            errors.pop()
            errors.reverse()

            #compute the deltas for bias and weight
            for index, error in enumerate(errors):
                bias_delta[index] += error
                weight_delta[index] += error * trace[index].transpose()

        if debug:
            print('epoch %s: error=%s'%(epoch, error_counter))

        #apply the deltas
        for index, delta in enumerate(weight_delta):
            network.weights[index] += step_size * delta + momentum * weight_delta_old[index]
        for index, delta in enumerate(bias_delta):
            network.bias[index] += step_size * delta + momentum * bias_delta_old[index]

        #set current deltas to old
        weight_delta_old = weight_delta
        bias_delta_old = bias_delta
            
    if debug:
        for potentials, expected in tests:
            print("input: %s => output: %s, expected %s"%(
                potentials.tolist(), network(potentials).tolist(), expected.tolist()))
    return network

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

if True:
    from neural_network import *
    global network
    tests = [((0,0),[0]),((0,1),[1]),((1,0),[1]),((1,1),[0])]
    network = NeuralNetwork.fromlayers([2,5,1])
    backpropagate(network, tests, 500, debug=False)
    print("done")
