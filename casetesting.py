import numpy as np
import ModelFunctions

case = 1991
desired = 1
Weights = np.load('model_weights.npy')

Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error = ModelFunctions.forwardPass(Weights, desired, case)

print(Yz)