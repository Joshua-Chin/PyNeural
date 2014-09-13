import math
import numpy as np

def anneal(network, tests, schedule=None, iterations=None, debug=False):

    #convert tests into numpy matrices
    tests = [(np.matrix(inputs, dtype=np.float64).reshape(len(inputs), 1),
            np.matrix(expected, dtype=np.float64).reshape(len(expected), 1))
            for inputs, expected in tests]
    
    if schedule is None:
        schedule = linear_schedule()
    if iterations is None:
        iterations = 10**100 #BIG number
        
    #keep track of the best network
    best_network = network.copy()
    best_error = rms_error(network, tests)

    error = best_error
    
    for index, temperature in enumerate(schedule):
        if 20*index > iterations:
            break
        accepted = 0
        for _ in range(20):
            new_network = network.copy()
            
            new_network.mutate()
            new_error = rms_error(new_network, tests)

            if new_error < error or math.e ** ((error-new_error)/temperature) > random.random():
                accepted += 1
                network = new_network
                error = new_error
                if error <= best_error:
                    best_network = network.copy()
                    best_error = error
                    
        if debug and not index%10:
            print('Epoch: %s; Error: %s'%(index, error))

    if debug:
        print("Best error: %s"%best_error)
    return best_network

def sum_error(network, tests):
    return sum(float(abs(np.sum((network(potentials) - expected))))
               for potentials, expected in tests)

        
def rms_error(network, tests):
    return sum(float(np.sum((network(potentials) - expected)**2))
               for potentials, expected in tests)

def linear_schedule(start=0.015, step=0.0001):
    while start > 0:
        yield start
        start -= step

def exp_schedule(start, factor):
    while True:
        yield start
        start *= factor

if True:
    from neural_network import *
    global network
    tests = [((0,0),[0]),((0,1),[1]),((1,0),[1]),((1,1),[0])]
    network = NeuralNetwork.fromlayers([2,5,1])
    network = anneal(network, tests, iterations=5000, debug=True)
    print("done")
