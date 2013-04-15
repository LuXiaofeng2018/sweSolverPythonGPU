'''
Created on Apr 15, 2013

@author: tristan
'''

import numpy as np


# Builds a square test mesh
def buildTestMesh(gridSize, gridPadding, cellWidth, cellHeight, floorElevation):

    m = gridSize + 2 * gridPadding
    n = gridSize + 2 * gridPadding

    # Mesh x,y,z data (bottom left corner of cell)
    # A 3-d array of the form [i, j, [x, y, z]] where i and j are
    # the element numbers along the x and y coordinates respectively
    # - In meshCoordinates, the point associated with the element i, j
    #   is the bottom left node of that element
    meshCoordinates = np.zeros((m + 1, n + 1, 3))
    for i in range(m + 1):
        for j in range(n + 1):
            meshCoordinates[i][j][0] = i * cellWidth
            meshCoordinates[i][j][1] = j * cellHeight
            meshCoordinates[i][j][2] = floorElevation

    # Calculate the bottom elevations at all integration points in the mesh
    # (See sheet for explanation of matrix)
    meshBottomIntegrationPoints = np.zeros((m + 1, n + 1, 2))
    for i in range(m):
        for j in range(n):

            # Each element stores only its own South [0] and West [1] values
            meshBottomIntegrationPoints[i][j][0] = (meshCoordinates[i][j][2] + meshCoordinates[i + 1][j][2]) / 2.0  # S
            meshBottomIntegrationPoints[i][j][1] = (meshCoordinates[i][j][2] + meshCoordinates[i][j + 1][2]) / 2.0  # W


    # Calculate the cell centered bottom slopes
    meshBottomSlopes = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            meshBottomSlopes[i][j][0] = (meshBottomIntegrationPoints[i + 1][j][1] - meshBottomIntegrationPoints[i][j][1]) / cellWidth  # dB/dx
            meshBottomSlopes[i][j][1] = (meshBottomIntegrationPoints[i][j + 1][0] - meshBottomIntegrationPoints[i][j][0]) / cellHeight  # dB/dy


    # Calculate the cell centered bottom elevations
    meshBottomCenters = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            meshBottomCenters[i][j] = (meshBottomIntegrationPoints[i + 1][j][1] - meshBottomIntegrationPoints[i][j][1]) / 2.0  # z


    return meshCoordinates, meshBottomIntegrationPoints, meshBottomSlopes, meshBottomCenters


def buildDoubleSlopingTestMesh(gridSize, gridPadding, cellWidth, cellHeight, lowestElevation, slope, middleNode):

    m = gridSize + 2 * gridPadding
    n = gridSize + 2 * gridPadding

    meshCoordinates = np.zeros((m + 1, n + 1, 3))
    for i in range(m + 1):
        for j in range(n + 1):
            meshCoordinates[i][j][0] = i * cellWidth
            meshCoordinates[i][j][1] = j * cellWidth
            meshCoordinates[i][j][2] = slope * abs(middleNode - i)

    meshBottomIntegrationPoints = np.zeros((m + 1, n + 1, 2))
    for i in range(m):
        for j in range(n):
            meshBottomIntegrationPoints[i][j][0] = (meshCoordinates[i][j][2] + meshCoordinates[i + 1][j][2]) / 2.0  # S
            meshBottomIntegrationPoints[i][j][1] = (meshCoordinates[i][j][2] + meshCoordinates[i][j + 1][2]) / 2.0  # W


    # Calculate the cell centered bottom slopes
    meshBottomSlopes = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            meshBottomSlopes[i][j][0] = (meshBottomIntegrationPoints[i + 1][j][1] - meshBottomIntegrationPoints[i][j][1]) / cellWidth  # dB/dx
            meshBottomSlopes[i][j][1] = (meshBottomIntegrationPoints[i][j + 1][0] - meshBottomIntegrationPoints[i][j][0]) / cellHeight  # dB/dy


    # Calculate the cell centered bottom elevations
    meshBottomCenters = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            meshBottomCenters[i][j] = (meshBottomIntegrationPoints[i + 1][j][1] - meshBottomIntegrationPoints[i][j][1]) / 2.0  # z


    return meshCoordinates, meshBottomIntegrationPoints, meshBottomSlopes, meshBottomCenters


# Build a square test mesh with sloping bottom
def buildSlopingTestMesh(gridSize, gridPadding, cellWidth, cellHeight, lowestElevation, highestElevation):

    m = gridSize + 2 * gridPadding
    n = gridSize + 2 * gridPadding

    meshCoordinates = np.zeros((m + 1, n + 1, 3))
    for i in range(m + 1):
        for j in range(n + 1):
            meshCoordinates[i][j][0] = i * cellWidth
            meshCoordinates[i][j][1] = j * cellWidth
            meshCoordinates[i][j][2] = highestElevation - i * (highestElevation - lowestElevation) / m


    meshBottomIntegrationPoints = np.zeros((m + 1, n + 1, 2))
    for i in range(m):
        for j in range(n):
            meshBottomIntegrationPoints[i][j][0] = (meshCoordinates[i][j][2] + meshCoordinates[i + 1][j][2]) / 2.0  # S
            meshBottomIntegrationPoints[i][j][1] = (meshCoordinates[i][j][2] + meshCoordinates[i][j + 1][2]) / 2.0  # W


    # Calculate the cell centered bottom slopes
    meshBottomSlopes = np.zeros((m, n, 2))
    for i in range(m):
        for j in range(n):
            meshBottomSlopes[i][j][0] = (meshBottomIntegrationPoints[i + 1][j][1] - meshBottomIntegrationPoints[i][j][1]) / cellWidth  # dB/dx
            meshBottomSlopes[i][j][1] = (meshBottomIntegrationPoints[i][j + 1][0] - meshBottomIntegrationPoints[i][j][0]) / cellHeight  # dB/dy


    # Calculate the cell centered bottom elevations
    meshBottomCenters = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            meshBottomCenters[i][j] = (meshBottomIntegrationPoints[i + 1][j][1] - meshBottomIntegrationPoints[i][j][1]) / 2.0  # z


    return meshCoordinates, meshBottomIntegrationPoints, meshBottomSlopes, meshBottomCenters


# Builds an initial U with the water free surface at a constant value. Zero initial velocities.
def buildConstantTestU(meshCoordinates, meshBottomCenters, freeSurfaceElevation):

    m = meshCoordinates.shape[0] - 1
    n = meshCoordinates.shape[1] - 1

    # Mesh conserved variables U = [w, h*u, h*v]
    # Initialized to constant depth and no velocities
    # A 3-d array of the form [i, j, U]
    # - In meshU, i, j refers to the cell centered average values
    meshU = np.zeros((m, n, 3))
    for i in range(m):
        for j in range(n):
            meshU[i][j][0] = freeSurfaceElevation
            meshU[i][j][1] = 0.0
            meshU[i][j][2] = 0.0

    return fixUndergroundWaterElevations(meshU, meshBottomCenters)



def buildPyramidTestU(meshCoordinates, meshBottomCenters, middleDepth, outerDepth, pyramidSize):

    m = meshCoordinates.shape[0] - 1
    n = meshCoordinates.shape[1] - 1

    pyramidSize = pyramidSize - pyramidSize % 2
    centerX = (m - m % 2) / 2 + 8
    centerY = (n - n % 2) / 2 + 8
    leftX = centerX - pyramidSize / 2
    bottomY = centerY - pyramidSize / 2
    rightX = leftX + pyramidSize
    topY = bottomY + pyramidSize

    meshU = np.zeros((m, n, 3))
    for i in range(m):
        for j in range(n):
            meshU[i][j][0] = outerDepth

    meshU = recursiveBuildPyramid(meshU, leftX, rightX, bottomY, topY, (middleDepth - outerDepth) / (pyramidSize / 2))

    return fixUndergroundWaterElevations(meshU, meshBottomCenters)







# # Helper Functions

def recursiveBuildPyramid(meshU, a, b, c, d, addHeight):

    for i in range(a, b):
        for j in range(c, d):
            meshU[i][j][0] += addHeight

    if (a + 1 < b - 1 and c + 1 < d - 1):
        return recursiveBuildPyramid(meshU, a + 1, b - 1, c + 1, d - 1, addHeight)
    else:
        return meshU


def fixUndergroundWaterElevations(meshU, meshBottomCenters):

    m = meshU.shape[0]
    n = meshU.shape[1]

    for i in range(m):
        for j in range(n):
            if meshU[i][j][0] < meshBottomCenters[i][j]:
                meshU[i][j][0] = meshBottomCenters[i][j]

    return meshU
