'''
Created on Apr 7, 2013

@author: tristan
'''

import matplotlib.pyplot as plt

def printCellCenteredMatrix(matrix, m, n, matrixName, index= -1):

    print "Cell centered values of " + matrixName + ":"
    if index != -1:
        for i in range(m):
            line = ''
            for j in range(n):
                line += "%.2f" % matrix[i][j][index] + "\t"
            print line
    else:
        for i in range(m):
            line = ''
            for j in range(n):
                line += str(matrix[i][j]) + "\t"
            print line

def print3DMatrix(matrix, m, n, index, matrixName=""):

    print str(index) + " value of " + matrixName + ":"
    for i in range(m):
        line = ''
        for j in range(n):
            line += "%.4f" % matrix[i][j][index] + "\t"
        print line


def print2DirectionInterfaceMatrix(matrix, m, n, direction, matrixName, index= -1):

    print getDirectionText(direction) + " value of " + matrixName + ":"
    if index != -1:
        for i in range(m):
            line = ''
            for j in range(n):
                if direction == 1:
                    line += str(matrix[i][j][0]) + "\t"
                elif direction == 3:
                    line += str(matrix[i][j][1]) + "\t"
                elif direction == 0:
                    line += str(matrix[i + 1][j][0][index]) + "\t"
                else:
                    line += str(matrix[i][j + 1][1][index]) + "\t"
            print line
    else:
        for i in range(m):
            line = ''
            for j in range(n):
                if direction == 1:
                    line += str(matrix[i][j][0]) + "\t"
                elif direction == 3:
                    line += str(matrix[i][j][1]) + "\t"
                elif direction == 0:
                    line += str(matrix[i + 1][j][0]) + "\t"
                else:
                    line += str(matrix[i][j + 1][1]) + "\t"
            print line

def print4DirectionCellMatrix(matrix, m, n, direction, matrixName, index= -1):

    print getDirectionText(direction) + " value of " + matrixName + ":"
    if index != -1:
        for i in range(m):
            line = ''
            for j in range(n):
                line += str(matrix[i][j][direction][index]) + "\t"
            print line
    else:
        for i in range(m):
            line = ''
            for j in range(n):
                line += str(matrix[i][j][direction]) + "\t"
            print line

def getDirectionText(direction):
    return {
        0 : 'North',
        1 : 'South',
        2 : 'East' ,
        3 : 'West'
        }.get(direction, 'Oops')

def plotTimeHistory(elementNum, fort63Loc):

    fort63 = open(fort63Loc, "r")

    fort63.readline()
    datLine = fort63.readline().split()
    numTS = int(datLine[0])
    numElements = int(datLine[1])
    fullDat = fort63.read().split()
    fort63.close()

    xVals = [i for i in range(numTS)]
    yVals = [float(fullDat[i * 2 * (numElements + 1) + 2 * elementNum + 1]) for i in range(numTS)]

    plt.plot(xVals, yVals)
    plt.show()































