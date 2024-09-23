import math
from pprint import pp
from random import random

from numpy import dot

input_number = 3

def eqm(weights, samples):
    e = 0
    
    for sample in samples:
        _, (_, Y2j) = forward(weights, sample[:input_number])
        u = Y2j[0]
        d = sample[input_number]
        e += (d - u) ** 2
    return e / len(samples)

def activation_function(n):
    return math.tanh(n)

def forward(weights, inputs):
    inputs = [-1] + inputs
    
    I1j = [dot(neuron_weights, inputs) for neuron_weights in weights[0]]
    Y1j = [-1] + [activation_function(I1i) for I1i in I1j]
    
    I2j = [dot(neuron_weights, Y1j) for neuron_weights in weights[1]]
    Y2j = [activation_function(I2i) for I2i in I2j]
    
    return (I1j, Y1j), (I2j, Y2j)
    

samples = []

with open("datasets/perceptron_multi_layer_TRAIN.txt", "r") as file:
    for line in file:
        if 'x1' in line:
            continue
        samples.append([float(value) for value in line.removesuffix('\n').split("    ")[1:]])
        

pp(samples)

# initialize the layers with random weights
layers = [
    # W1
    [
        [ 
            random() for _ in range(input_number + 1) # (inputs number) + 1 (for the bias) 
        ]
        for _ in range(10) # 10 is the number of neurons of the next layer
    ],
    
    # W2 (intermediate layer)
    [
        [ 
            random() for _ in range(10 + 1) # 10 (neuron number) + 1 (for the bias)
        ]
        for _ in range(1) # 1 is the number of neurons of the output layer
    ]
]

pp(layers)

LEARNING_RATE = 0.1
PRECISION = 0.000001

epoch = 0

while True:
    em = eqm(layers, samples)
    
    print("EM: ", em)
    
    for sample in samples:
        (I1j, Y1j), (I2j, Y2j) = forward(layers, sample[:input_number])
        break

    