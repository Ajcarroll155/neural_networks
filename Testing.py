from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)


hiddenVal = 0
penDataStorage = {}
carDataStorage = {}
while hiddenVal <= 40:
    counter = 0
    carVals = []
    penVals = []
    print("Running 5 Iterations with " , hiddenVal , " perceptrons in the hidden layer\n")
    print("STARTING CAR DATA ITERATIONS\n--------------------------------------\n")
    while counter < 5:
        (nNet, accVal) = testCarData([hiddenVal])
        carVals.append(accVal)
        counter = counter + 1
    print("CAR DATA COMPLETE\n")
    print("MAX ACCURACY: ", max(carVals), "\n")
    print("AVERAGE ACCURACY: ", average(carVals), "\n")
    print("STANDARD DEVIATION: ", stDeviation(carVals), "\n")
    carDataStorage[hiddenVal] = [max(carVals), average(carVals), stDeviation(carVals)]
    print("STARTING PEN DATA ITERATIONS\n--------------------------------------\n")
    counter = 0
    while counter < 5:
        (nNet, accVal) = testPenData([hiddenVal])
        penVals.append(accVal)
        counter = counter + 1
    print("PEN DATA COMPLETE\n")
    print("MAX ACCURACY: ", max(penVals), "\n")
    print("AVERAGE ACCURACY: ", average(penVals), "\n")
    print("STANDARD DEVIATION: ", stDeviation(penVals), "\n")
    penDataStorage[hiddenVal] = [max(penVals), average(penVals), stDeviation(penVals)]
    print("Increasing perceptrons in the hidden layer...")
    hiddenVal = hiddenVal + 5
print("Perceptron Limit Reached. Ending Test\n")
print("\n")

itr = 0
while itr <= 40:
    print("DATA FOR " , itr , " PERCEPTRONS:\n")
    penVals = penDataStorage[itr]
    carVals = carDataStorage[itr]
    print("PEN DATA:    |   MAX = ",max(penVals),"     |    AVERAGE = ",average(penVals),"     |    stDEV = ",stDeviation(penVals),"     |\n")
    print("CAR DATA:    |   MAX = ",max(carVals),"     |    AVERAGE = ",average(carVals),"     |    stDEV = ",stDeviation(carVals),"     |\n")
    print("-------------------------------------------------------------------------------------------------------------------------------\n")
    itr = itr + 5