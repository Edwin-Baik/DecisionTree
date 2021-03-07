import sys
import csv
import math
import matplotlib.pyplot as plt

def makeMatrix(trainInput):
    res = []
    attriVals = {}
    attriCount = 0
    for row in trainInput:
        rowVal = []
        for col in row:
            rowVal.append(col)
        res.append(rowVal)
    for attri in res[0]:
        attriVals[attri] = []
        attriCount += 1
    for row in range(1, len(res)):
        for col in range(len(res[row])):
            if res[row][col] not in attriVals[res[0][col]]:
                attriVals[res[0][col]].append(res[row][col])
    return (res, attriVals, attriCount)

def checkRow(row, split):
    for key in split:
        if str(row[key]) != str(split[key]):
            return False
    return True

def createMasterDict(trainMatrix, split):
    masterDict = {}   
    newMatrix = [trainMatrix[0]]   
    for i in range(len(trainMatrix[0]) - 1):
        if trainMatrix[0][i] not in masterDict:
            masterDict[trainMatrix[0][i]] = {}
    if len(split) == 0:
        for row in range(1, len(trainMatrix)):
                for attri in range(len(trainMatrix[row]) - 1):
                        value = trainMatrix[row][attri]
                        
                        if value not in masterDict[trainMatrix[0][attri]]:
                            masterDict[trainMatrix[0][attri]][value] = {}
                            
                        if trainMatrix[row][-1] not in masterDict[trainMatrix[0][attri]][value]:
                            masterDict[trainMatrix[0][attri]][value][trainMatrix[row][-1]] = 1
                        
                        else: 
                            masterDict[trainMatrix[0][attri]][value][trainMatrix[row][-1]] += 1
    else:
        for row in range(1, len(trainMatrix)):
            if checkRow(trainMatrix[row], split):
                newMatrix.append(trainMatrix[row])
        
        for row in range(1, len(newMatrix)):
            for attri in range(len(newMatrix[row]) - 1):
                    value = newMatrix[row][attri]
                    
                    if value not in masterDict[newMatrix[0][attri]]:
                        masterDict[newMatrix[0][attri]][value] = {}
                        
                    if newMatrix[row][-1] not in masterDict[newMatrix[0][attri]][value]:
                        masterDict[newMatrix[0][attri]][value][newMatrix[row][-1]] = 1
                    
                    else: 
                        masterDict[newMatrix[0][attri]][value][newMatrix[row][-1]] += 1
    return masterDict

def entropyCalc(trainMatrix):
    countDict = {}
    overall = 0
    for row in range(1,len(trainMatrix)):
        yVal = trainMatrix[row][-1]
        if yVal not in countDict:
            countDict[yVal] = 1
        else:
            countDict[yVal] += 1
        overall += 1
    entropyList = []
    for label in countDict:
        frac = countDict[label] / overall
        entropy = -1 * frac * math.log(frac, 2)
        entropyList.append(entropy)
    return sum(entropyList)

def condEntropy(masterDict, index, trainMatrix):
    attribute = trainMatrix[0][index]
    attriSubSet = {}
    overallSum = 0
    for key in masterDict:
        if key == attribute:
            attriSubSet = masterDict[key]
    overall = 0
    for key in attriSubSet:
        for val in attriSubSet[key]:
            overall += attriSubSet[key][val]
    for key in attriSubSet:
        innerDict = attriSubSet[key]
        conditionSum = 0
        for val in innerDict:
            partialSum = sum(innerDict.values())
            count = innerDict[val]
            frac = (count/partialSum)
            conditionSum += frac * math.log(frac, 2)
        overallSum += conditionSum * (-1) * (partialSum / overall)
    return overallSum                

def mutualInfo(trainMatrix, split):
    labelEntropy = entropyCalc(trainMatrix)
    masterDict = createMasterDict(trainMatrix, split)
    mutualDict = {}
    for index in range(len(trainMatrix[0])-1):
        mutualDict[index] = labelEntropy - condEntropy(masterDict, index, trainMatrix)
    return mutualDict 
    
def decisionStump(trainMatrix, split):
    labelVals = {}
    for row in range(1, len(trainMatrix)):
        if checkRow(trainMatrix[row], split):
            if trainMatrix[row][-1] not in labelVals:
                labelVals[trainMatrix[row][-1]] = 1
            else:
                labelVals[trainMatrix[row][-1]] += 1
    if len(labelVals) == 0:
        return 'NA'
    return max(sorted(labelVals, reverse=True),key = labelVals.get)
    

def decisionTree(trainMatrix, currDepth, maxDepth, DT, attriValues, split):
    mutualDict = mutualInfo(trainMatrix, split)
    if currDepth < maxDepth:
        if len(list(set(list(mutualDict.values())))) == 1:
            val = decisionStump(trainMatrix, split)
            return val
        else:
            bestAttri = max(mutualDict, key = mutualDict.get)
            attriName = trainMatrix[0][bestAttri]
            if attriName not in DT:
                innerDict = {}
                for value in attriValues[attriName]:
                    innerDict[value] = {}
                DT[attriName] = innerDict
            for value in DT[attriName]:
                split[bestAttri] = value
                DT[attriName][value] = decisionTree(trainMatrix, currDepth + 1, maxDepth, {}, attriValues, split)
                split.pop(bestAttri)
            return DT
    else:
        return decisionStump(trainMatrix, split)

def recursiveCheck(trainMatrix, DT, indexInitial, initialAttri, line):
    if isinstance(DT[initialAttri][line[indexInitial]], dict):
        DT = DT[initialAttri][line[indexInitial]]
        initialAttri = trainMatrix[0].index(list(DT.keys())[0])
        indexInitial = list(DT.keys())[0]
        return recursiveCheck(trainMatrix, DT, initialAttri, indexInitial, line)
    else: 
        return DT[initialAttri][line[indexInitial]]
        
    
def trainMetrics(DT, traininput, trainout, metricout):
    out_file = open(traininput)
    read_in = csv.reader(out_file, delimiter = '\t')
    next(read_in, None)
    errors = 0
    overall = 0
    out_file = open(trainout, 'w')
    metric_file = open(metricout, 'w')
    for line in read_in:
        if isinstance(DT, str):
            if DT != line[-1]:
                errors += 1
            out_file.write(str(DT) + '\n')
        else:
            indexInitial = trainMatrix[0].index(list(DT.keys())[0])
            initialAttri = list(DT.keys())[0]
            val = recursiveCheck(trainMatrix, DT, indexInitial, initialAttri, line)
            if val != line[-1]:
                errors += 1
            out_file.write(str(val) + '\n')
        overall += 1
    metric_file.write("error(train): " + str(errors/overall) + "\n")
    out_file.close()
    return errors/overall

def testMetrics(DT, testinput, testout, metricout):
    out_file = open(testinput)
    read_in = csv.reader(out_file, delimiter = '\t')
    next(read_in, None)
    out_file = open(testout, 'w')
    metric_file = open(metricout, 'a')
    errors = 0
    overall = 0
    for line in read_in:
        if isinstance(DT, str):
            if DT != line[-1]:
                errors += 1
            out_file.write(str(DT) + '\n')
        else:
            indexInitial = trainMatrix[0].index(list(DT.keys())[0])
            initialAttri = list(DT.keys())[0]
            val = recursiveCheck(trainMatrix, DT, indexInitial, initialAttri, line)
            if val != line[-1]:
                errors += 1
            out_file.write(str(val) + '\n')
        overall += 1
    metric_file.write("error(test): " + str(errors/overall) + "\n")
    out_file.close()
    return errors/overall

def getErrors(DT, traininput, testinput):
    train_file = open(traininput)
    train_in = csv.reader(train_file, delimiter = '\t')
    next(train_in, None)
    test_file = open(testinput)
    test_in = csv.reader(test_file, delimiter = '\t')
    next(test_in, None)
    errors = 0
    overall = 0
    for line in train_in:
        if isinstance(DT, str):
            if DT != line[-1]:
                errors += 1
        else:
            indexInitial = trainMatrix[0].index(list(DT.keys())[0])
            initialAttri = list(DT.keys())[0]
            val = recursiveCheck(trainMatrix, DT, indexInitial, initialAttri, line)
            if val != line[-1]:
                errors += 1
        overall += 1
    trainError = errors/overall
    errors = 0
    overall = 0
    for line in test_in:
        if isinstance(DT, str):
            if DT != line[-1]:
                errors += 1
        else:
            indexInitial = trainMatrix[0].index(list(DT.keys())[0])
            initialAttri = list(DT.keys())[0]
            val = recursiveCheck(trainMatrix, DT, indexInitial, initialAttri, line)
            if val != line[-1]:
                errors += 1
        overall += 1
    testError = errors/overall
    return (trainError, testError)


def makePlot(trainMatrix, attriValues, attriCount, traininput, testinput):
    x = []
    trainError = []
    testError = []
    for depth in range(attriCount + 1):
        x.append(depth)
        DT = decisionTree(trainMatrix, 0, depth, {}, attriValues, {})
        (trE, tE) = getErrors(DT, traininput, testinput)
        trainError.append(trE)
        testError.append(tE)
    plt.plot(x, trainError, color = 'blue', label = "Training Error") 
    plt.plot(x, testError, color = "red", label = 'Testing Error')
    plt.xlabel("Tree Depth")
    plt.ylabel("Errors")
    plt.legend()
    plt.show()

def createNumCount(trainMatrix, split):
    newMatrix = []
    countDict = {}
    for row in range(1, len(trainMatrix)):
        if checkRow(trainMatrix[row], split):
            newMatrix.append(trainMatrix[row])
        if trainMatrix[row][-1] not in countDict:
            countDict[trainMatrix[row][-1]] = 0
    numCount = []
    for row in newMatrix:
        countDict[row[-1]] += 1
    stringConvert = ''
    for key in sorted(countDict):
        stringConvert += str(countDict[key]) + " " + str(key) + '/'
    numCount.append(stringConvert[:len(stringConvert)-1])
    return numCount

def printTree(DT, trainMatrix, currDepth, maxDepth, split):
    if currDepth == 0:
        createNumCount(trainMatrix, split)
    for key in DT:
        if isinstance(DT, dict):
            for attri in DT[key]:
                mutualDict = mutualInfo(trainMatrix, split)
                bestAttri = max(mutualDict, key = mutualDict.get)
                split[bestAttri] = attri
                numCount = createNumCount(trainMatrix, split)
                print(((int(maxDepth) - currDepth + 1) * '|') + " " + str(key) + " = " + str(attri) + ": " + str(numCount))
                printTree(DT[key][attri], trainMatrix, currDepth - 1, maxDepth, split)
                split.pop(bestAttri)
        else:
            createNumCount(trainMatrix, split)
        
    
    
    

if __name__ == '__main__':
    traininput = sys.argv[1]
    testinput = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainout = sys.argv[4]  
    testout = sys.argv[5]
    metricout = sys.argv[6]
    in_file = open(traininput)
    read_in = csv.reader(in_file, delimiter = '\t')
    necessaryInfo = makeMatrix(read_in)
    trainMatrix = necessaryInfo[0]
    attriValues = necessaryInfo[1]
    attriCount = necessaryInfo[2] - 1
    DT = decisionTree(trainMatrix, 0, maxDepth, {}, attriValues, {})
    trainMetrics(DT, traininput, trainout, metricout)
    testMetrics(DT, testinput, testout, metricout)
    makePlot(trainMatrix, attriValues, attriCount, traininput, testinput)
    numCount = createNumCount(trainMatrix, {})
    print(numCount)
    printTree(DT, trainMatrix, maxDepth, maxDepth, {})



