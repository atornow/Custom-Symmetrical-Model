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


def weightUpdater(EofX, XofW):
    newChange = (EofX * XofW)

    return newChange


def forwardPass(Weights, desired, case):
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

    error = np.square(Yz - desired) / 2

    return Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error


def backwardsPass(case, Weights, desired, ErrorSet):
    savedChanges = np.zeros(len(Weights), dtype=float)
    Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error = forwardPass(Weights, desired, case)
    ErrorSet.append(error)

    derivativeEofXz = 2 * (Yz - desired) * activateInverse(Xz)
    derivativeEofXa = derivativeEofXz * Weights[8] * activateInverse(Xa)
    derivativeEofXb = derivativeEofXz * Weights[9] * activateInverse(Xb)

    currentDigit = len(str(case)) - 1

    savedChanges[weightIndex] = weightUpdater(derivativeEofXz, Yb)
    weightIndex -= 1
    savedChanges[weightIndex] = weightUpdater(derivativeEofXz, Ya)
    weightIndex -= 1

    while weightIndex >= 0:
        savedChanges[weightIndex] = weightUpdater(derivativeEofXb, int(str(case)[currentDigit])
                                                  )
        weightIndex -= 1
        savedChanges[weightIndex] = weightUpdater(derivativeEofXa, int(str(case)[currentDigit])
                                                  )
        weightIndex -= 1
        currentDigit -= 1

    return ErrorSet, savedChanges


def generateWeights(modelSize):
    # Generates weight initializations
    Weights = np.random.uniform(-0.4, 0.4, modelSize)
    np.save('initial_weights.npy', Weights)


# Runs through Epochs number of epochs on a single Cases set
def runEpochs(epochs, momentum, learningRate, genWeights, genData, genValid, modelSize):
    # If new intialization weights are desired, generate them
    if genWeights:
        generateWeights(modelSize)
    Weights = np.load('initial_weights.npy')

    Cases, Desired, validationCases, validationDesired = load_or_generate_data(200, genData, False)
    Cases, Desired, validationCases, validationDesired = load_or_generate_data(500, False, genValid)

    ErrorSetEpoch = []
    vErrorSetEpoch = [0.0] * epochs
    momentumSavedChanges = [0.0] * modelSize

    for epoch in range(0, epochs):
        caseIndex = 0
        ErrorSet = []
        savedChangesEpoch = np.zeros(modelSize, dtype=float)

        # Run backwards pass on each case and add EofW derivatives to savedChangesEpoch
        for case in Cases:
            ErrorSet, savedChanges = backwardsPass(case, Weights, Desired[caseIndex], ErrorSet)
            savedChangesEpoch += savedChanges
            caseIndex += 1
        ErrorSetEpoch.append(ErrorSet[-1])

        # Average derivatives over an epoch and apply them with momentum to Weights
        savedChangesEpoch = savedChangesEpoch / len(Cases)
        for weightIndex in range(0, modelSize):
            newChange = (learningRate * savedChangesEpoch[weightIndex]) + (momentum * momentumSavedChanges[weightIndex])
            momentumSavedChanges[weightIndex] = newChange
            Weights[weightIndex] -= newChange

        if genValid:
            for i in range(0, len(validationCases)):
                Xa, Ya, Xb, Yb, Xz, Yz, weightIndex, error = forwardPass(Weights, validationDesired[i], validationCases[i])
                vErrorSetEpoch[epoch] += error

            # vErrorSetEpoch[epoch] = vErrorSetEpoch[epoch] / len(validationCases)

    np.save('model_weights.npy', Weights)
    return ErrorSetEpoch, vErrorSetEpoch


def plot_error_set(ErrorSet):
    # Determine the size of ErrorSet
    size = len(ErrorSet)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(ErrorSet, label='Error')

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
