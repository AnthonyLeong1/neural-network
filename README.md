# neural-network #

The directory comprises of 4 files:

## neuralNetwork.py ##

This file contains a python class with the fully connected layers of the neural network. The class must be initialised with a number of arguments:
* *neuralList*: list of integers. Each integer denotes the number of neurons in each layer. The first list element corresponds to the input layer and the last list element corresponds to the output layer.
* *gradCost*: function. This is the gradient of the cost function.
* *activation*: function. This is the activation function applied to the output of each layer.
* *diffActivation*: function. This is the derivative of the activation function.
* *trainingData*: list of two elements, the first element is a list of input data. Each piece of input data is a list of neurons with the same dimensions as the input layer. The second element is a list of the corresponding expected outputs. Each output is a list of neurons with the same dimensions as the output layer.
* *testingData*: similarly defined as the training data, except this data is used to validate the accuracy of the neural network.
* *batchSize*: integer. This is the sample size of test data used in stochastic gradient descent.
* *stepSize*: float. This is the step size used within gradient descent.
* *regStrength*: float. This determines the strength of L2 regularisation. Regularisation as the effect of encouraging the weights to the have a smaller magnitude. When regStrength=0, no regularisation is present.

2 class methods may be run. __trainNetwork__ trains the network, optimising weights and biases using stochastic gradient descent and backpropogation. __validate__ determines the accuracy of the neural network by comparing the output predicted by the neural network with the expected output.
## poolingAndConvolution.py ##

This file contains two functions, __convolutionLayer__ and __poolingLayer__. These functions enable the application of a CNN (convolution neural network) for image detection. The processes of convolution and pooling are applied before the data passesthrough the fully connected layers of the neural network.

__convolutionLayer__ applies the process of convolution, a filter is applied to the input image (given as a 2d numpy array) to extract salient features from the image.

__poolingLayer__ applies pooling - either max pooling or average pooling - to reduce the dimensions of the input data.

## activationAndCostLib.py ##

The class in __neuralNetwork.py__ incorporates cost functions and activation functions as input parameters. Examples of such functions can be stored within __activationAndCostLib.py__.
## classifyingDigitsExample.py ##

We apply the neural network to a simple image classification problem of identifying a single digit number given a greyscale image of the digit. The images are sourced from the MNIST data set which are available from the Keras module. An improved identification accuracy may be achieved with further experimentation of hyperparameters.
## Modules ##

Finally, to run the neural network a number of modules are required:
* numpy 
* math
* copy
* keras: this is only necessary for __classifyingDigitsExample.py__ to load the input data
