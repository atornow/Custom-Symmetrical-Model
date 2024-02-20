import numpy as np
import ModelFunctions

# Hyper parameters of the model
modelSize = 10
learningRate = 0.25
momentum = 0.9
epochs = 2000
genData = False
genValid = True
genWeights = True

# See how same validation set error after training decreases as chosen variable changes
ErrorSet, vErrorSet = ModelFunctions.runEpochs(epochs, momentum, learningRate, genWeights, genData, genValid, modelSize)

print(np.load('model_weights.npy'))
ModelFunctions.plot_error_set(vErrorSet)


