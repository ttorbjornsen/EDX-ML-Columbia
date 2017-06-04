import sys
import os
import pandas as pd
import numpy as np

def parseInput(input):
    try:
        return int(input)
    except ValueError:
        return float(input)

inputLambda = parseInput(sys.argv[1])
inputSigma = parseInput(sys.argv[2])


scriptDir = os.path.dirname(__file__) #<-- absolute dir the script is in
rrTargetFileRelPath = 'wRR_' + str(inputLambda) + '.csv'
rrTargetFileAbsPath = os.path.join(scriptDir, rrTargetFileRelPath)

alTargetFileRelPath = 'active_' + str(inputLambda) + '_' + str(inputSigma) + '.csv'
alTargetFileAbsPath = os.path.join(scriptDir, alTargetFileRelPath)

scriptDir = os.path.dirname(__file__) #<-- absolute dir the script is in
xTrainPath = os.path.join(scriptDir, sys.argv[3])
yTrainPath = os.path.join(scriptDir, sys.argv[4])
xTestPath = os.path.join(scriptDir, sys.argv[5])

X = pd.read_csv(xTrainPath, header=None)
y = pd.read_csv(yTrainPath, header=None)
d = X.shape[1] #number of dimensions (feaatures)

# PART 1 - Ridge regression
identityMatrix = np.identity(d)
X_T = X.transpose()
leastSquares = (np.linalg.inv(X_T.dot(X))).dot(X_T).dot(y)
ridgeRegression = (np.linalg.inv(inputLambda*identityMatrix + X_T.dot(X))).dot(X_T).dot(y)
dfRidgeRegression = pd.DataFrame(ridgeRegression)
dfRidgeRegression.to_csv(rrTargetFileAbsPath, sep=';', header=None, index=False)


# PART 2 - Active learning
XTest = pd.read_csv(xTestPath, header=None)


def getHighestVariance(indexTraversalList, XTest):
    highestVariance = -1
    tempIndex = -1
    for xTestRow in range(0, XTest.shape[0]):
        x0 = XTest.iloc[[xTestRow]]
        x0_T = x0.transpose()
        sigma0 = inputSigma + x0.dot(covarianceSigmaMatrix).dot(x0_T).iloc[0][xTestRow]

        if (sigma0 > highestVariance) and (xTestRow+1 not in indexTraversalIndex):
            highestVariance = sigma0
            tempIndex = xTestRow + 1

    return tempIndex


indexTraversalIndex = []
for i in range(0,10): #find the 10 highest variances
    identityMatrix = np.identity(d)
    X_T = X.transpose()
    covarianceSigmaMatrix = np.linalg.inv(inputLambda * identityMatrix + ((X_T.dot(X))/inputSigma))
    highestVarianceIndex = getHighestVariance(indexTraversalIndex, XTest)
    indexTraversalIndex.append(highestVarianceIndex)
    X = X.append(XTest.iloc[[highestVarianceIndex-1]])

dfActiveLearning = pd.DataFrame(indexTraversalIndex)
dfActiveLearning.to_csv(alTargetFileAbsPath, sep=';', header=None, index=False)



