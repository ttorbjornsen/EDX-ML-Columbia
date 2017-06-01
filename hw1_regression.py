import sys
import os
import pandas as pd
import numpy as np

inputLambda = float(sys.argv[1])
inputSigma = float(sys.argv[2])

print 'lambda: ' + str(inputLambda)
print 'sigma: ' + str(inputSigma)


scriptDir = os.path.dirname(__file__) #<-- absolute dir the script is in
xTrainPath = os.path.join(scriptDir, sys.argv[3])
yTrainPath = os.path.join(scriptDir, sys.argv[4])
X = pd.read_csv(xTrainPath, header=None)
y = pd.read_csv(yTrainPath, header=None)
d = X.shape[0]

identityMatrix = np.identity(d)
temp = inputLambda*identityMatrix
X_T = X.transpose()

firstTerm = (X.dot(X_T) + inputLambda*identityMatrix)
result = np.linalg.inv(firstTerm).dot(X_T.dot(y))

# result.to_csv(targetFileAbsPath, sep=';',)



