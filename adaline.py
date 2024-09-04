from numpy import dot
from random import random
import matplotlib.pyplot as plt

input_n = 5

def eqm(weights, amostras):
    e = 0
    for amostra in amostras:
        u = output(weights, amostra[:input_n])
        d = amostra[input_n]
        e += (d - u) ** 2
    return e / len(amostras)
    

def output(weights, x) -> float:
    return dot(weights, x)

amostras = [
    [-1, 0.4329, -1.3719, 0.7022, -0.8535, 1.0000],
    [-1, 0.3024, 0.2286, 0.8630, 2.7909, -1.0000],
    [-1, 0.1349, -0.6445, 1.0530, 0.5687, -1.0000],
    [-1, 0.3374, -1.7163, 0.3670, -0.6283, -1.0000],
    [-1, 1.1434, -0.0485, 0.6637, 1.2606, 1.0000],
    [-1, 1.3749, -0.5071, 0.4464, 1.3009, 1.0000],
    [-1, 0.7221, -0.7587, 0.7681, -0.5592, 1.0000],
    [-1, 0.4403, -0.8072, 0.5154, -0.3129, 1.0000],
    [-1, -0.5231, 0.3548, 0.2538, 1.5776, -1.0000],
    [-1, 0.3255, -2.0000, 0.7112, -1.1209, 1.0000],
    [-1, 0.5824, 1.3915, -0.2291, 4.1735, -1.0000],
    [-1, 0.1340, 0.6081, 0.4450, 3.2230, -1.0000],
    [-1, 0.1480, -0.2988, 0.4778, 0.8649, 1.0000],
    [-1, 0.7359, 0.1869, -0.0872, 2.3584, 1.0000],
    [-1, 0.7115, -1.1469, 0.3394, 0.9573, -1.0000],
    [-1, 0.8251, -1.2840, 0.8452, 1.2382, -1.0000],
    [-1, 0.1569, 0.3712, 0.8825, 1.7633, 1.0000],
    [-1, 0.0033, 0.6835, 0.5389, 2.8249, -1.0000],
    [-1, 0.4243, 0.8313, 0.2634, 3.5855, -1.0000],
    [-1, 1.0490, 0.1326, 0.9138, 1.9792, 1.0000],
    [-1, 1.4276, 0.5331, -0.0145, 3.7286, 1.0000],
    [-1, 0.5971, 1.4865, 0.2904, 4.6069, -1.0000],
    [-1, 0.8475, 2.1479, 0.3179, 5.8235, -1.0000],
    [-1, 1.3967, -0.4171, 0.6443, 1.3927, 1.0000],
    [-1, 0.0044, 1.5378, 0.6099, 4.7755, -1.0000],
    [-1, 0.2201, -0.5668, 0.0515, 0.7829, 1.0000],
    [-1, 0.6300, -1.2480, 0.8591, 0.8093, -1.0000],
    [-1, -0.2479, 0.8960, 0.0547, 1.7381, 1.0000],
    [-1, -0.3088, -0.0929, 0.8659, 1.5483, -1.0000],
    [-1, -0.5180, 1.4974, 0.5453, 2.3993, 1.0000],
    [-1, 0.6833, 0.8266, 0.0829, 2.8864, 1.0000],
    [-1, 0.4353, -1.4066, 0.4207, -0.4879, 1.0000],
    [-1, -0.1069, -3.2329, 0.1856, -2.4572, -1.0000],
    [-1, 0.4662, 0.6261, 0.7304, 3.4370, -1.0000],
    [-1, 0.8298, -1.4089, 0.3119, 1.3235, -1.0000]
]

weights = [random() for _ in range(input_n)] # 4 inputs + 1 para o limiar

print("Initial weights: ", weights)

LEARNING_RATE = 0.0025
PRECISION = 0.000006

epoch = 1

eqms = []

while True:
    last_e = eqm(weights, amostras)
    
    for amostra in amostras:
        u = output(weights, amostra[:input_n])
        
        for i in range(input_n):
            d = amostra[input_n]
            weights[i] += LEARNING_RATE * (d - u) * amostra[i]
            
    epoch += 1
    
    curr_e = eqm(weights, amostras)
    
    eqms.append(curr_e)
    
    if abs(curr_e - last_e) <= PRECISION:
        print("Finishing")
        break
    
print("Final weights: ", weights)
print("Epochs: ", epoch)

# Avaliar com pesos personalizados

print("\n\n---Testing the model---\n")

tests = [
    [-1, 0.9694, 0.6909, 0.4334, 3.4965],
    [-1, 0.5427, 1.3832, 0.6390, 4.0352],
    [-1, 0.6081, -0.9196, 0.5925, 0.1016],
    [-1, -0.1618, 0.4694, 0.2030, 3.0117],
    [-1, 0.1870, -0.2578, 0.6124, 1.7749],
    [-1, 0.4891, -0.5276, 0.4378, 0.6439],
    [-1, 0.3777, 2.0149, 0.7423, 3.3932],
    [-1, 1.1498, -0.4067, 0.2469, 1.5866],
    [-1, 0.9325, 1.0950, 1.0359, 3.3591],
    [-1, 0.5060, 1.3317, 0.9222, 3.7174],
    [-1, 0.0497, -2.0656, 0.6124, -0.6585],
    [-1, 0.4004, 3.5369, 0.9766, 5.3532],
    [-1, -0.1874, 1.3343, 0.5374, 3.2189],
    [-1, 0.5060, 1.3317, 0.9222, 3.7174],
    [-1, 1.6375, -0.7911, 0.7537, 0.5515]
]

for test in tests:
    print("Testando inputs: ", test, end="\n")
    
    print("Válvula: ", "A" if output(weights, test) <= 0 else "B")

# Plotting the graph
plt.plot(range(1, epoch), eqms)
plt.xlabel("Epoch")
plt.ylabel("EQM")
plt.title("Epoch vs EQM")
plt.show()