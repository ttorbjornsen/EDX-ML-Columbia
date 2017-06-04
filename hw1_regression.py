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
targetFileRelPath = 'wRR_' + str(inputLambda) + '.csv'
targetFileAbsPath = os.path.join(scriptDir, targetFileRelPath)

scriptDir = os.path.dirname(__file__) #<-- absolute dir the script is in
xTrainPath = os.path.join(scriptDir, sys.argv[3])
yTrainPath = os.path.join(scriptDir, sys.argv[4])
X = pd.read_csv(xTrainPath, header=None)
y = pd.read_csv(yTrainPath, header=None)
d = X.shape[1] #number of dimensions (feaatures)

identityMatrix = np.identity(d)
X_T = X.transpose()

leastSquares = (np.linalg.inv(X_T.dot(X))).dot(X_T).dot(y)
ridgeRegression = (np.linalg.inv(inputLambda*identityMatrix + X_T.dot(X))).dot(X_T).dot(y)

dfRidgeRegression = pd.DataFrame(ridgeRegression)
dfRidgeRegression.to_csv(targetFileAbsPath, sep=';', header=None, index=False)



