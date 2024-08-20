import math
from numpy import dot 
from typing import List

def sigmoid(t: float) -> float:
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs) -> float:
    # weights inclui o termo de viés, o vetor inclui um 1
    return sigmoid(dot(weights, inputs))

def feed_forward(neural_network: List[List[List]],
                 input_vector: List) -> List[List]:
    '''
    Alimenta o vetor de entrada na rede neural.
    Retorna as saídas de todas as camadas (não só a última).
    '''
    outputs: List[List] = []
    
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        
        # Agora, a entrada da próxima camada é a saída desta
        input_vector = output
        
    return output

# XOR 
xor_network =[
    # camada oculta
    [
        [20., 20, -30], # neuronio 'and'
        [20., 20, -10] # neuronio 'or'
    ],
    # camada de saída
    [
        [-60., 60, -30] 
    ]
] 

'''
o feed_forward retorna as saídas de todas as camadas para que [-1] receba
a saída final e para que [0] receba o valor do vetor resultante
'''
assert 0.000 < feed_forward(xor_network, [0, 0])[0] < 0.001
assert 0.999 < feed_forward(xor_network, [1, 0])[0] < 1.000
assert 0.999 < feed_forward(xor_network, [0, 1])[0] < 1.000
assert 0.000 < feed_forward(xor_network, [1, 1])[0] < 0.001

