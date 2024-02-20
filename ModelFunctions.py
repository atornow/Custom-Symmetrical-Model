import numpy as np
import math
import os
import matplotlib.pyplot as plt
from case_generator import generate_cases

def load_or_generate_data(x, newData, newValid):
    if newData:
        valid = False
        symmetrical_numbers, binary_array = generate_cases(x, valid, save_to_file=True)
    elif os.path.exists('symmetrical_numbers.npy') and os.path.exists('binary_array.npy'):
        symmetrical_numbers = np.load('symmetrical_numbers.npy')
        binary_array = np.load('binary_array.npy')

    if newValid:
        valid = True
        validationSym, validationBinary = generate_cases(x, valid, save_to_file=True)
    elif os.path.exists('validationSym.npy') and os.path.exists('validationBinary.npy'):
        validationSym = np.load('validationSym.npy')
        validationBinary = np.load('validationBinary.npy')

    return symmetrical_numbers, binary_array, validationSym, validationBinary

def activate(x):
    return 1 / (1 + math.exp(-x))

def activateInverse(x):
    return math.exp(x) / np.square((math.exp(x) + 1))

def weightUpdater(index, EofX, XofW, Weights, newWeights, momentum, learningRate, savedChanges):
    newChange = -1 * learningRate * (EofX * XofW)
    newWeights[index] = Weights[index] + newChange + (momentum * savedChanges[index])
    savedChanges[index] = newChange

    return newWeights[index], newChange

def forwardPass(Weights, Desired, case):
    Xa, Xb, Xz = 0, 0, 0
    weightIndex = 0
    for digit in str(case):
        digitInt = int(digit)
        Xa += digitInt * Weights[weightIndex]
        Xb += digitInt * Weights[weightIndex + 1]
        weightIndex += 2

    Ya = activate(Xa)
    Yb = activate(Xb)

    Xz = (Ya * Weights[weightIndex]) + (Yb * Weights[weightIndex + 1])
    Yz = activate(Xz)
    weightIndex += 1

    error = np.square(Yz - Desired) / 2

    return Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error

def backwardsPass(case, caseIndex, Weights, Desired, ErrorSet, momentum, learningRate, savedChanges):
    newWeights = [0.0] * len(Weights)
    Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error = forwardPass(Weights, Desired[caseIndex], case)
    ErrorSet.append(error)

    derivativeEofXz = 2 * (Yz - Desired[caseIndex]) * activateInverse(Xz)
    derivativeEofXa = derivativeEofXz * Weights[8] * activateInverse(Xa)
    derivativeEofXb = derivativeEofXz * Weights[9] * activateInverse(Xb)

    currentDigit = len(str(case)) - 1

    newWeights[weightIndex], savedChanges[weightIndex] = weightUpdater(weightIndex, derivativeEofXz, Yb, Weights, newWeights, momentum, learningRate,
                                 savedChanges)
    weightIndex -= 1
    newWeights[weightIndex], savedChanges[weightIndex] = weightUpdater(weightIndex, derivativeEofXz, Ya, Weights, newWeights, momentum, learningRate,
                                 savedChanges)
    weightIndex -= 1

    while weightIndex >= 0:
        newWeights[weightIndex], savedChanges[weightIndex] = weightUpdater(weightIndex, derivativeEofXb, int(str(case)[currentDigit]), Weights,
                                     newWeights, momentum, learningRate, savedChanges)
        weightIndex -= 1
        newWeights[weightIndex], savedChanges[weightIndex] = weightUpdater(weightIndex, derivativeEofXa, int(str(case)[currentDigit]), Weights,
                                     newWeights, momentum, learningRate, savedChanges)
        weightIndex -= 1
        currentDigit -= 1

    return newWeights, ErrorSet, savedChanges

def generateWeights(modelSize):
    # Generates weight initializations
    Weights = np.random.uniform(-0.3, 0.3, modelSize)
    np.save('initial_weights.npy', Weights)


# Runs through Epochs number of epochs on a single Cases set
def runEpochs(epochs, Cases, Desired, momentum, learningRate, genWeights, modelSize):
    if genWeights:
        generateWeights(modelSize)
    Weights = np.load('initial_weights.npy')
    savedChanges = [0.0] * modelSize
    ErrorSetEpoch = []
    for epoch in range(1, epochs):
        caseIndex = 0
        ErrorSet = []

        # Train model using backwards propagation every on a per case basis
        for case in Cases:
            Weights, ErrorSet, savedChanges = backwardsPass(case, caseIndex, Weights, Desired, ErrorSet, momentum, learningRate, savedChanges)
            caseIndex += 1
        ErrorSetEpoch.append(ErrorSet[-1])

    np.save('model_weights.npy', Weights)

    return ErrorSetEpoch

def plot_error_set(ErrorSet):
    # Determine the size of ErrorSet
    size = len(ErrorSet)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(ErrorSet, label='Error per Case')

    # Scaling the x-axis based on the size of ErrorSet
    if size > 10000000:
        ax.set_xscale('log')
        ax.set_xlabel('Case (log scale)')
    else:
        ax.set_xlabel('Case')

    # Enhancements for readability
    ax.set_ylabel('Error')
    ax.set_title('Error per Case in a Single Training Epoch')
    ax.grid(True)
    ax.legend()

    # Show the plot
    plt.show()