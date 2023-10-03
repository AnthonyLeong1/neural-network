# import modules
import numpy as np
import copy

# network

class Object():

    def __init__(self,neuralList,gradCost,activation,diffActivation,trainingData,testingData,batchSize=10,stepSize=0.1,regStrength=0):
        """
        :param neuralList: list of number of neurons in each layer. First list element is input layer, last list element
        is the output layer. The other list elements correspond to hidden layers.
        :param gradCost: gradient of the cost function
        :param activation: activation function applied to output of each layer
        :param diffActivation: derivative of activation function
        :param trainingData: data to train network
        :param testingData: data to test network
        :param batchSize: integer value for the sample size used in stochastic gradient descent
        :param stepSize: float value for step size in gradient descent
        :param regStrength: float vlaue for regularisation parameter in gradient descent
        """

        ## properties

        # assigning inputs
        self.neuralList = neuralList
        self.L = len(neuralList)-1
        # weights and bias
        self.wList,self.bList = self.initWeightAndBiasRand(neuralList)
        # properties of loaded data
        self.xTrainList = trainingData[0]
        self.yTrainList = trainingData[1]
        self.xTestList = testingData[0]
        self.yTestList = testingData[1]
        # properties for traning network
        self.zList = [np.zeros([lenLayer,1]) for lenLayer in neuralList]  # output of layer (before activation)
        self.aList = copy.deepcopy(self.zList)   # output of layer (after activation)
        self.dList = [np.zeros(layerLen) for layerLen in neuralList[1:]] # errors
        self.currentOutput = None # current output
        self.gradCost = gradCost # gradient of cost function
        self.activation = activation # activation function
        self.diffActivation = diffActivation # derivative of activation function
        self.batchSize = batchSize # size of batches in stochastic gradient descent
        self.stepSize = stepSize # step size in gradient descent
        self.regStrength = regStrength # strength of regularisation

    # methods to intialise bias and weights

    @staticmethod
    def initWeightAndBiasRand(neuralList):
        biasList = []
        weightList = []
        L = len(neuralList)-1
        for l in range(L):
            biasList.append(np.random.randn(neuralList[l+1],1))
            weightList.append(np.random.randn(neuralList[l+1],neuralList[l]))
        return weightList,biasList

    @staticmethod
    def initWeightAndBiasZeros(neuralList):
        biasList = []
        weightList = []
        L = len(neuralList) - 1
        for l in range(L):
            biasList.append(np.zeros([neuralList[l + 1], 1]))
            weightList.append(np.zeros([neuralList[l + 1], neuralList[l]]))
        return weightList, biasList

    # methods to train network

    def trainNetwork(self):
        numBatches = int(len(self.xTrainList) / self.batchSize)
        for i in range(numBatches):
            self.gradientDescent(self.xTrainList[i * self.batchSize:(i + 1) * self.batchSize], self.yTrainList[i * self.batchSize:(i + 1) * self.batchSize])
        print("Network trained")

    def gradientDescent(self,inputBatch,outputBatch):
        """
        performs gradient descent for an input batch of size m and The corresponding outputs are given in output batch
        """
        m = len(inputBatch)
        wAddList,bAddList = self.initWeightAndBiasZeros(self.neuralList)  # change in weight and bias under gradient descent

        for currentInput,currentOutput in zip(inputBatch,outputBatch): # loop over members of batch
            #self.aList[0] = copy.deepcopy(currentInput)
            self.zList[0] = copy.deepcopy(currentInput)
            self.aList[0] = self.activation(self.zList[0])
            self.currentOutput = copy.deepcopy(currentOutput) # change output to CNN
            self.feedForward()
            self.backPropogation()
            for l in range(self.L):  # loop over layers of CNN
                wAddList[l] += np.outer(self.dList[l],self.aList[l])
                bAddList[l] += self.dList[l]
        for l in range(self.L):
            self.wList[l] = (1-self.regStrength*self.stepSize/m)*self.wList[l] - (self.stepSize/m)*wAddList[l] # change weight and bias
            self.bList[l] -= (self.stepSize/m)*bAddList[l]

    def feedForward(self):
        for l,[w,b] in enumerate(zip(self.wList,self.bList)):
            self.zList[l+1] =  np.matmul(w, self.aList[l]) + b
            self.aList[l+1] = self.activation(self.zList[l+1])

    def backPropogation(self):
        # compute output error
        self.dList[-1] = self.gradCost(self.aList[-1],self.currentOutput)*self.diffActivation(self.zList[-1])
        # backpropogate error
        for l in range(self.L-2,-1,-1):
            self.dList[l]  =  (np.matmul(np.transpose(self.wList[l+1]), self.dList[l+1])) * self.diffActivation(self.zList[l+1])

    # method to validate results

    def validate(self):
        L = len(self.xTestList)
        numCorrect = 0
        for testInput,testOutput in zip(self.xTestList,self.yTestList):
            #self.aList[0] = copy.deepcopy(testInput)
            self.zList[0] = copy.deepcopy(testInput)
            self.aList[0] = self.activation(self.zList[0])
            self.feedForward()
            if self.aList[-1].argmax() == testOutput:
                numCorrect+=1
        print("Success rate: " + str(numCorrect*100/L) + "%")
