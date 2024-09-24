import math
from random import random

from numpy import dot

INPUT_NUMBER = 3
LEARNING_RATE = 0.1
PRECISION = 0.000001

def eqm(weights, samples):
    e = 0
    for sample in samples:
        _, (_, Y2j) = forward(weights, sample[:INPUT_NUMBER])
        u = Y2j[0]
        d = sample[INPUT_NUMBER]
        e += (d - u) ** 2
    return e / len(samples)

def activation_function(n):
    return math.tanh(n)

def derivative_activation_function(n):
    return 1 - math.tanh(n)**2

def forward(weights, inputs):
    inputs = [-1] + inputs
    
    I1j = [dot(neuron_weights, inputs) for neuron_weights in weights[0]]
    Y1j = [-1] + [activation_function(I1i) for I1i in I1j]
    
    I2j = [dot(neuron_weights, Y1j) for neuron_weights in weights[1]]
    Y2j = [activation_function(I2i) for I2i in I2j]
    
    return (I1j, Y1j), (I2j, Y2j)

def backward(layers, I1j, I2j, Y1j, output, expected_output, inputs):
    inputs = [-1] + inputs
    
    grad2j = [(expected_output - output) * derivative_activation_function(i2j) for i2j in I2j] 
    for j in range(len(layers[1])):
        neuron = layers[1][j]
        for i in range(len(neuron)):
            neuron[i] += LEARNING_RATE * grad2j[j] * Y1j[i]
            
    grad1j = [-dot(grad2j, [layers[1][k][j] for k in range(len(layers[1]))])  * derivative_activation_function(I1j[j]) for j in range(len(I1j))] 
    for j in range(len(layers[0])):
        neuron = layers[0][j]
        for i in range(len(neuron)):
            neuron[i] += LEARNING_RATE * grad1j[j] * inputs[i]
    

samples = []

with open("datasets/perceptron_multi_layer_TRAIN.txt", "r") as file:
    for line in file:
        if 'x1' in line:
            continue
        samples.append([float(value) for value in line.removesuffix('\n').split("    ")[1:]])

# initialize the layers with random weights
layers = [
    # W1
    [
        [ 
            random() for _ in range(INPUT_NUMBER + 1) # (inputs number) + 1 (for the bias) 
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

epoch = 0

while True:
    last_em = eqm(layers, samples)
    
    print("EM: ", last_em)
    
    for sample in samples:
        xs = sample[:INPUT_NUMBER]
        d = sample[-1]
        
        (I1j, Y1j), (I2j, Y2j) = forward(layers, xs)
        backward(layers, I1j, I2j, Y1j, Y2j[0], d, xs)
    
    curr_em = eqm(layers, samples)
    
    epoch += 1
    
    if abs(curr_em - last_em) <= PRECISION:
        print("Finishing")
        print("EPOCHS: ", epoch)
        break
    
# Testing the model
print("\n\n---Testing the model---\n")

random_sample = samples[int(random() * len(samples))]

(I1j, Y1j), (I2j, Y2j) = forward(layers, random_sample[:INPUT_NUMBER])

print("Expected: ", random_sample[-1])
print("Result: ", Y2j)




    