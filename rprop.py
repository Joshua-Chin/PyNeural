import numpy as np

def irprop(network, tests, iterations=500, debug=False):

    #convert tests into numpy matrices
    tests = [(np.matrix(inputs, dtype=np.float64).reshape(len(inputs), 1),
            np.matrix(expected, dtype=np.float64).reshape(len(expected), 1))
            for inputs, expected in tests]

    #keep track of old values for momentum
    weight_deriv_old = [np.zeros(matrix.shape) for matrix in network.weights]
    bias_deriv_old = [np.zeros(matrix.shape) for matrix in network.bias]

    weight_delta = [np.empty(matrix.shape) for matrix in network.weights]
    bias_delta = [np.empty(matrix.shape) for matrix in network.bias]

    for delta in weight_delta:
        delta.fill(0.0125)
    for delta in bias_delta:
        delta.fill(0.0125)
    
    for epoch in range(iterations):

        #accumulate the weight and bias deltas
        weight_deriv = [np.zeros(matrix.shape) for matrix in network.weights]
        bias_deriv = [np.zeros(matrix.shape) for matrix in network.bias]

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

            #compute the total derivative for bias and weight
            for index, error in enumerate(errors):
                bias_deriv[index] += error
                weight_deriv[index] += error * trace[index].transpose()

        for index, (layer_deriv, old_layer_deriv) in enumerate(zip(weight_deriv, weight_deriv_old)):

            changed_sign = layer_deriv * old_layer_deriv < 0

            #if the derivative changed signs, we jumped over a local minima, so decrease step size
            delta_mult = np.where(changed_sign, 0.5, 1.2)
            weight_delta[index] *= delta_mult #Set Delta to zero latter
            np.clip(weight_delta[index], 0, 50, weight_delta[index])
            layer_deriv[changed_sign] = 0
            
            #apply delta in the proper direction
            network.weights[index] += np.sign(layer_deriv) * weight_delta[index]

        for index, (layer_deriv, old_layer_deriv) in enumerate(zip(bias_deriv, bias_deriv_old)):

            changed_sign = layer_deriv * old_layer_deriv < 0

            #if the derivative changed signs, we jumped over a local minima, so decrease step size
            delta_mult = np.where(changed_sign, 0.5, 1.2)
            bias_delta[index] *= delta_mult #Set Delta to zero latter
            np.clip(bias_delta[index], 0, 50, bias_delta[index])

            layer_deriv[changed_sign] = 0
            
            #apply delta in the proper direction
            network.bias[index] += np.sign(layer_deriv) * bias_delta[index]


        weight_deriv_old = weight_deriv
        bias_deriv_old = bias_deriv                

        if debug and not epoch%int(debug):
            print('epoch %s: error=%s'%(epoch, error_counter))
            
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
    clone = network.copy()
    irprop(network, tests, 10000, debug=1000)
    print("done")
