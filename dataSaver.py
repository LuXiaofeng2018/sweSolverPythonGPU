'''
Created on Apr 15, 2013

@author: tristan
'''

import os

def writeCustomFort14(workingDir, meshCoordinates):

    fort14 = open(workingDir + '/fort.14g', 'w')
    fort14.write("Test fort.14 file for finite volume analysis\n")

    m = meshCoordinates.shape[0]
    n = meshCoordinates.shape[1]

    fort14.write(str(2 * (m - 2) * n - (m - 2) + 2 * n) + "\t" + str(m * n) + "\t" + str(m) + "\n")

    # Write nodes
    for i in range(m):
        for j in range(n):
            line = str(m * i + j + 1) + "\t" + str(meshCoordinates[i][j][0]) + "\t" + str(meshCoordinates[i][j][1]) + "\t" + str(meshCoordinates[i][j][2]) + "\n"
            fort14.write(line)

    # Write triangle strip list
    fort14.write("1\n")
    for i in range(m / 2):
        leftList = [j + 1 for j in range(2 * i * n, 2 * i * n + n)]
        rightList = [j + 1 for j in range(2 * i * n + n, 2 * i * n + 2 * n)]
        currList = [j for k in zip(leftList, rightList) for j in k]
        for j in range(1, len(currList)):
            fort14.write(str(currList[j]) + "\n")
        if (2 * (i + 1) < m):
            leftList = reversed(rightList)
            rightList = reversed([j + 1 for j in range(2 * i * n + 2 * n, 2 * i * n + 3 * n)])
            currList = [j for k in zip(leftList, rightList) for j in k]
            for j in range(1, len(currList)):
                fort14.write(str(currList[j]) + "\n")

    fort14.close()

def createFort63(workingDir, meshCoordinates):

    fort63 = open(workingDir + '/fort.63.temp', 'w')
    return fort63


def closeCustomFort63(workingDir, fort63temp, meshU, numTS):

    m = meshU.shape[0]
    n = meshU.shape[1]

    fort63temp.close()
    fort63temp = open(workingDir + '/fort.63.temp', 'r')

    fort63 = open(workingDir + '/fort.63g', 'w')
    fort63.write("Test fort.63 file for finite volume analysis\n")
    fort63.write(str(numTS) + "\t" + str(m * n) + "\t1.0\t1\n")
    fort63.write(fort63temp.read())

    fort63.close()
    fort63temp.close()
    os.remove(workingDir + '/fort.63.temp')

def writeCustomTimestep(fort63, meshU):

    m = meshU.shape[0]
    n = meshU.shape[1]

    fort63.write("1.0\t1\n")
    for i in range(m):
        for j in range(n):
            fort63.write(str(m * i + j + 1) + "\t" + str(meshU[i][j][0]) + "\n")

def writeTimestep(fort63, meshU, meshBottomCenters):

    m = meshU.shape[0]
    n = meshU.shape[1]

    fort63.write("1.0\t1\n")
    for i in range(m + 1):
        for j in range(n + 1):
            if (i < m and j < n):
                if (meshU[i][j][0] == meshBottomCenters[i][j]):
                    line = str((m + 1) * i + j + 1) + "\t" + "-99999.0\n"
                else:
                    line = str((m + 1) * i + j + 1) + "\t" + "%.5f" % meshU[i][j][0] + "\n"
            elif (i < m and j == n):
                if (meshU[i][j - 1][0] == meshBottomCenters[i][j - 1]):
                    line = str((m + 1) * i + j + 1) + "\t" + "-99999.0\n"
                else:
                    line = str((m + 1) * i + j + 1) + "\t" + "%.5f" % meshU[i][j - 1][0] + "\n"
            elif (i == m and j < n):
                if (meshU[i - 1][j][0] == meshBottomCenters[i - 1][j]):
                    line = str((m + 1) * i + j + 1) + "\t" + "-99999.0\n"
                else:
                    line = str((m + 1) * i + j + 1) + "\t" + "%.5f" % meshU[i - 1][j][0] + "\n"
            else:
                if (meshU[i - 1][j - 1][0] == meshBottomCenters[i - 1][j - 1]):
                    line = str((m + 1) * i + j + 1) + "\t" + "-99999.0\n"
                else:
                    line = str((m + 1) * i + j + 1) + "\t" + "%.5f" % meshU[i - 1][j - 1][0] + "\n"
            fort63.write(line)
