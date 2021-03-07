import sys
import csv
import math

def entropyCalc(infile, outfile):
    in_file = open(infile)
    read_in = csv.reader(in_file, delimiter = '\t')
    next(read_in, None)
    countDict = {}
    overall = 0
    for val in read_in:
        if val[-1] not in countDict:
            countDict[val[-1]] = 1
        else:
            countDict[val[-1]] += 1
        overall += 1
    entropyList = []
    for label in countDict:
        frac = countDict[label] / overall
        entropy = -1 * frac * math.log(frac, 2)
        entropyList.append(entropy)
    minVal = countDict[min(countDict.keys(), key=(lambda k: countDict[k]))]
    out_file = open(outfile, 'w')
    out_file.write('entropy: ' + str(sum(entropyList)) + '\n')
    out_file.write('error: ' + str(minVal/overall) + '\n')
    out_file.close()


if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    entropyCalc(infile, outfile)