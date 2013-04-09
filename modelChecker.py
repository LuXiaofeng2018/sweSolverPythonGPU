'''
Created on Apr 7, 2013

@author: tristan
'''


def printCellCenteredMatrix(matrix, m, n, matrixName):

    print "Cell centered values of " + matrixName + ":"
    for i in range(m):
        line = ''
        for j in range(n):
            line += str(matrix[i][j][0]) + "\t"
        print line

def print3DMatrix(matrix, m, n, index, matrixName=""):

    print str(index) + " value of " + matrixName + ":"
    for i in range(m):
        line = ''
        for j in range(n):
            line += str(matrix[i][j][index]) + "\t"
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
