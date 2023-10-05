# modules
import numpy as np
import math
from keras.datasets import mnist
import activationAndCostLib as lib
import neuralNetwork

# load data

def loadData(trainSize,testSize):
    """
    :param trainSize: number of data used for training the NN
    :param testSize: number of data used for testing the NN
    :return: testingData: list containing data to train NN. First list element contains flattened 784 pixel greyscale
    image. Second list element records the corresponding digits for each image.
    trainingData: list containing data to test NN. First list element contains flattened 784 pixel greyscale
    image. Second list element records the corresponding digits for each image.
    """

    # load raw data
    (rawTrainX, rawTrainY), (rawTestX, rawTestY) = mnist.load_data()
    # initialise output lists
    trainingData = [None, None]
    testingData = [None, None]
    # format data
    trainingData[0] = [x.reshape([784, 1]) for x in rawTrainX[:trainSize]]
    trainingData[1] = [formatOutput(y) for y in rawTrainY[:trainSize]]
    testingData[0] = [x.reshape([784, 1]) for x in rawTestX[:testSize]]
    testingData[1] = rawTestY[:testSize]
    return trainingData, testingData

def formatOutput(num):
    listOutput = np.zeros([10, 1])
    listOutput[num] = 1
    return listOutput


# image classification example:

#assign arguments
neuralList = [28**2,30,10]
gradCost = lib.crossEntropy
activation = lib.sigmoid
diffActivation = lib.diffSigmoid
trainingData,testingData = loadData(60000,10000)
batchSize = 10
stepSize = 0.1
regStrength = 0.1
# intialise class object
this = neuralNetwork.Object(neuralList,gradCost,activation,diffActivation,trainingData,testingData,batchSize,stepSize,regStrength)
# train network
this.trainNetwork()
# validate network
this.validate()