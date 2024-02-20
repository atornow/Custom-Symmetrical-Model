import numpy as np
import ModelFunctions

# Hyper parameters of the model
modelSize = 10
learningRate = 0.05
momentum = 0.8
epochs = 1000
fullRunCount = 5
genData = False
genValid = False
genWeights = False
train = True
vErrorSet = [0.0] * fullRunCount
Cases, Desired, validationCases, validationDesired = ModelFunctions.load_or_generate_data(100, genData, genValid)

if train:

    # See how same validation set error after training decreases as chosen variable changes
    for i in range(0, fullRunCount):
        learningRate += 0.05
        ErrorSetEpoch = ModelFunctions.runEpochs(epochs, Cases, Desired, momentum, learningRate, genWeights, modelSize)

        # Perform forward pass for each vCase using learned weights
        vCaseIndex = 0
        for vCase in validationCases:
            Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error = ModelFunctions.forwardPass(np.load('model_weights.npy'),
                                                                                    validationDesired[vCaseIndex], vCase
                                                                                    )

            # Sum error over the set and save as index in main validation error set
            vErrorSet[i] += np.square(error)

ModelFunctions.plot_error_set(vErrorSet)
print(np.load('model_weights.npy'))

