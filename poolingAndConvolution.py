# modules
import numpy as np
import math

# convolution layer
def convolutionLayer(mxInput, mxFilter):
    """
    :param mxInput: input as a 2d numpy array
    :param mxFilter: filter as a 2d numpy array
    :return: image map as a 2d numpy array
    """
    dimsInput = np.array(np.shape(mxInput))
    dimsFilter = np.array(np.shape(mxFilter))
    imageMap = np.zeros(dimsInput - dimsFilter + 1)
    for i in range(len(imageMap)):
        for j in range(len(imageMap[0])):
            imageMap[i, j] = np.sum(mxFilter * mxInput[i:i + dimsFilter[0], j:j + dimsFilter[1]])
    return imageMap

# pooling layer
def poolingLayer(mxInput, dims=[2, 2], stride=2, poolType='max'):
    """
    :param mxInput: input as a 2d numpy array
    :param dims: dimensions of pooling filter
    :param stride: distance pooling filter is shifted
    :param poolType: either 'max' or 'average' for maximum pooling or average pooling respectively
    :return: image map as a numpy array
    """
    outputDims = [math.floor(x/stride) for x in np.shape(mxInput)]
    mxOut = np.zeros(outputDims)
    if poolType == 'max':
        for i in range(0, outputDims[0]):
            for j in range(0, outputDims[1]):
                mxOut[i, j] = mxInput[i * stride:i * stride + dims[0], j * stride:j * stride + dims[1]].max()
    elif poolType == 'average':
        for i in range(0, outputDims[0]):
            for j in range(0, outputDims[1]):
                mxOut[i, j] = mxInput[i * stride:i * stride + dims[0], j * stride:j * stride + dims[1]].mean()
    else:
        return 'poolType must be either "max" or "average"'
    return mxOut
