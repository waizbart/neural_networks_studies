from numpy import dot
from random import random

def activation_fn(x: float) -> float:
    return 1.0 if x >= 0 else -1

def perceptron_output(weights, x) -> float:
    calculation = dot(weights, x)
    return activation_fn(calculation)

amostras = [
    [ -1, -0.6508, 0.1097, 4.0009, -1.0000 ],
    [ -1, -1.4492, 0.8896, 4.4005, -1.0000 ],
    [ -1, 2.0850, 0.6876, 12.0170, -1.0000 ],
    [ -1, 0.2626, 1.1476, 7.7985, 1.0000 ],
    [ -1, 0.6418, 1.0234, 7.0427, 1.0000 ],
    [ -1, 0.2569, 0.6730, 8.3265, -1.0000 ],
    [ -1, 1.1155, 0.6043, 7.4446, 1.0000 ],
    [ -1, 0.0914, 0.3399, 7.0677, -1.0000 ],
    [ -1, 0.0121, 0.5256, 4.6316, 1.0000 ],
    [ -1, -0.0429, 0.4660, 5.4323, 1.0000 ],
    [ -1, 0.4342, 0.6870, 8.2287, -1.0000 ],
    [ -1, 0.2735, 1.0287, 7.1934, 1.0000 ],
    [ -1, 0.4839, 0.4851, 7.4858, -1.0000 ],
    [ -1, 0.4089, 0.1267, 5.5019, -1.0000 ],
    [ -1, 1.4391, 0.1614, 8.5843, -1.0000 ],
    [ -1, -0.9115, 0.1973, 4.2648, -1.0000 ],
    [ -1, 0.3654, 1.0475, 7.1952, 1.0000 ],
    [ -1, 0.2144, 0.7515, 7.1699, 1.0000 ],
    [ -1, 0.2013, 1.0014, 6.5489, 1.0000 ],
    [ -1, 0.6483, 0.2183, 5.8991, 1.0000 ],
    [ -1, -0.1147, 0.2242, 7.2435, -1.0000 ],
    [ -1, -0.7970, 0.8795, 3.8762, 1.0000 ],
    [ -1, -1.0625, 0.6366, 2.4707, 1.0000 ],
    [ -1, 0.5307, 0.1285, 5.6883, 1.0000 ],
    [ -1, -1.2200, 0.7777, 1.7252, 1.0000 ],
    [ -1, 0.3957, 0.1076, 5.6623, -1.0000 ],
    [ -1, -0.1013, 0.5989, 7.1812, -1.0000 ],
    [ -1, 2.4482, 0.9455, 11.2095, 1.0000 ],
    [ -1, 2.0149, 0.6192, 10.9263, -1.0000 ],
    [ -1, 0.2012, 0.2611, 5.4631, 1.0000 ],
]

weights = [random() for _ in range(4)] # 3 inputs + 1 para o limiar

print("Initial weights: ", weights)

LEARNING_RATE = 0.01

epoch = 1
MAX_EPOCHS = 1000

while epoch <= MAX_EPOCHS:
    print("\nEpoch:", epoch)
    
    erro = False
    validation = {"right": 0, "wrong": 0}
    
    for amostra in amostras:
        out = perceptron_output(weights, x=amostra[:4])
        d = amostra[4]
        
        if out != d:
            for i in range(4):
                weights[i] += LEARNING_RATE * (d - out) * amostra[i]
            erro = True
            validation["wrong"] += 1
        else:
            validation["right"] += 1
            
    accuracy = (validation["right"] / (validation["wrong"] + validation["right"])) * 100
    print("Accuracy: ", accuracy, "%")
            
    epoch += 1
    
    if not erro:
        print("\nTraining completed after", epoch - 1, "epochs.")
        break

if epoch > MAX_EPOCHS:
    print("\nReached maximum epochs. Training might not be fully converged.")
    
print("Final weights: ", weights)

# Avaliar com pesos personalizados

print("\n\n---Testing the model---\n")

tests = [
    [-0.3665, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548]
]

for test in tests:
    print("Testando inputs: ", test, end="\n")
    
    print("Saída: ", perceptron_output(weights, [-1] + test))


