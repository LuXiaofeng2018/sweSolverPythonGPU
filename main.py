'''
Created on Apr 3, 2013

@author: tristan
'''

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from dataTransfer import *
from gpuHelper import *
from spacialDiscretization import *
from fluxCalculations import *
from sourceCalculations import *
from timestepper import *
from boundaryConditions import *
from modelChecker import *
from dataSaver import *
from meshBuilder import *

printGPUMemUsage()

workingDir = "/home/tristan/Desktop/"
saveOutput = True
test = True
time = 0.0
dt = 0.0
runTime = 10.0
simTime = 0.0
iterations = 0

# Build test mesh
freeSurface = 0.0
cellWidth = 1.0
cellHeight = 1.0
gridSize = 32
m = gridSize + 4
n = gridSize + 4

# Calculate GPU grid/block sizes
blockDim = 16
gridM = m / blockDim + (1 if (m % blockDim != 0) else 0)
gridN = n / blockDim + (1 if (n % blockDim != 0) else 0)

meshCoordinates, meshBottomIntPts, meshBottomSlopes, meshBottomCenters = buildDoubleSlopingTestMesh(gridSize, 2, cellWidth, cellHeight, 1.0, 0.1, 12)
meshU = buildPyramidTestU(meshCoordinates, meshBottomCenters, 5.0, 2.0, 5)

# print3DMatrix(meshCoordinates, m, n, 2, "meshCoordinates")
# printCellCenteredMatrix(meshBottomCenters, m, n, "meshbottomcenters")
# print3DMatrix(meshU, m, n, 0, "meshU")
# exit()

# meshCoordinates = np.zeros((m + 1, n + 1, 3))
# for i in range(m + 1):
#     for j in range(n + 1):
#         meshCoordinates[i][j][0] = float(j)
#         meshCoordinates[i][j][1] = float(i)
#         if j <= n / 2:
#             meshCoordinates[i][j][2] = j - 5.0
#         else:
#             meshCoordinates[i][j][2] = n - j - 5.0
#
if saveOutput:
    writeCustomFort14(workingDir, meshCoordinates)
    fort63 = createFort63(workingDir, meshCoordinates)
#
# meshBottomIntPts = np.zeros((m + 1, n + 1, 2))
# for i in range(m + 1):
#     for j in range(n + 1):
#         if j < n / 2:
#             if j == n / 2 - 1:
#                 meshBottomIntPts[i][j][0] = j - 5.0
#                 meshBottomIntPts[i][j][1] = j - 5.0
#             else:
#                 meshBottomIntPts[i][j][0] = j + 0.5 - 5.0
#                 meshBottomIntPts[i][j][1] = j - 5.0
#         else:
#             meshBottomIntPts[i][j][0] = n - j - 1 - 0.5 - 5.0
#             meshBottomIntPts[i][j][1] = n - j - 1 - 5.0
#
# meshU = np.zeros((m, n, 3))
# for i in range(m):
#     for j in range(n):
#         if meshBottomIntPts[i][j][0] <= freeSurface:
#             meshU[i][j][0] = freeSurface
#         else:
#             meshU[i][j][0] = meshBottomIntPts[i][j][0]


# Allocate memory on GPU
meshUGPU = sendToGPU(meshU)
meshUIntPtsGPU = gpuarray.zeros((m, n, 4, 3), np.float32)
meshBottomIntPtsGPU = sendToGPU(meshBottomIntPts)
meshHUVIntPtsGPU = gpuarray.zeros((m, n, 4, 3), np.float32)
meshPropSpeedsGPU = gpuarray.zeros((m, n, 4), np.float32)
meshFluxesGPU = gpuarray.zeros((m, n, 2, 3), np.float32)
meshSlopeSourceGPU = gpuarray.zeros((m, n, 2), np.float32)
meshShearSourceGPU = gpuarray.zeros((m, n), np.float32)
meshRValuesGPU = gpuarray.zeros((m, n, 3), np.float32)
meshUstarGPU = gpuarray.zeros_like(meshUGPU)

printGPUMemUsage()


# Start Timestepping
while time < runTime:

    if saveOutput:
        meshU = meshUGPU.get()
        writeCustomTimestep(fort63, meshU)

    # Reconstruct free surface
    if (test == False):
        freeSurfaceTime = reconstructFreeSurfaceTimed(meshUGPU, meshUIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
        positivityTime = preservePositivityTimed(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU, m, n, [blockDim, blockDim], [gridN, gridM])
        huvTime = calculateHUVTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
        updateUTime = updateUIntPtsTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
        propSpeedTime = calculatePropSpeedsTimed(meshPropSpeedsGPU, meshHUVIntPtsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
        fullTime = 0.0
    else:
        freeSurfaceTime = 0.0
        positivityTime = 0.0
        huvTime = 0.0
        updateUTime = 0.0
        propSpeedTime = 0.0
        fullTime = fullPropSpeedsTimed(meshUGPU, meshBottomIntPtsGPU, meshUIntPtsGPU, meshHUVIntPtsGPU, meshPropSpeedsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])

    # Calculate Fluxes and Source Terms
    fluxTime = fluxSolverTimed(meshFluxesGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, meshPropSpeedsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
    slopeSourceTime = solveBedSlopeTimed(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
    shearSourceTime = solveBedShearTimed(meshShearSourceGPU, meshUGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])

    buildRTime = buildRValuesTimed(meshRValuesGPU, meshFluxesGPU, meshSlopeSourceGPU, m, n, [blockDim, blockDim], [gridN, gridM])

    # Calculate Timestep
    dt = calculateTimestep(meshPropSpeedsGPU, cellWidth)

    # Build U* and apply boundary conditions
    uStarTime = buildUstarTimed(meshUstarGPU, meshUGPU, meshRValuesGPU, meshShearSourceGPU, dt, m, n, [blockDim, blockDim], [gridN, gridM])
    # uStar = meshUstarGPU.get()
    # print "Before"
    # print3DMatrix(uStar, m, n, 2, "uStar")
    bcTimeStar = applyWallBoundariesTimed(meshUstarGPU, m, n, [blockDim, blockDim], [gridN, gridM])
    # uStar = meshUstarGPU.get()
    # print "After"
    # print3DMatrix(uStar, m, n, 2, "uStar")

    # Reconstruct free surface
    if (test == False):
        freeSurfaceTimeStar = reconstructFreeSurfaceTimed(meshUstarGPU, meshUIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
        positivityTimeStar = preservePositivityTimed(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUstarGPU, m, n, [blockDim, blockDim], [gridN, gridM])
        huvTimeStar = calculateHUVTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
        updateUstarTime = updateUIntPtsTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
        propSpeedStarTime = calculatePropSpeedsTimed(meshPropSpeedsGPU, meshHUVIntPtsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
        fullTimeStar = 0.0
    else:
        freeSurfaceTimeStar = 0.0
        positivityTimeStar = 0.0
        huvTimeStar = 0.0
        updateUstarTime = 0.0
        propSpeedStarTime = 0.0
        fullTimeStar = fullPropSpeedsTimed(meshUstarGPU, meshBottomIntPtsGPU, meshUIntPtsGPU, meshHUVIntPtsGPU, meshPropSpeedsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])

    # Calculate Fluxes and Source Terms
    fluxStarTime = fluxSolverTimed(meshFluxesGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, meshPropSpeedsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
    slopeSourceStarTime = solveBedSlopeTimed(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
    shearSourceStarTime = solveBedShearTimed(meshShearSourceGPU, meshUstarGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
    buildRStarTime = buildRValuesTimed(meshRValuesGPU, meshFluxesGPU, meshSlopeSourceGPU, m, n, [blockDim, blockDim], [gridN, gridM])

    # Build Unext and apply boundary conditions
    buildUnextTime = buildUnextTimed(meshUGPU, meshUGPU, meshUstarGPU, meshRValuesGPU, meshShearSourceGPU, dt, m, n, [blockDim, blockDim], [gridN, gridM])
    # uStar = meshUGPU.get()
    # print "Before"
    # print3DMatrix(uStar, m, n, 2, "U")
    bcTime = applyWallBoundariesTimed(meshUGPU, m, n, [blockDim, blockDim], [gridN, gridM])
    # uStar = meshUGPU.get()
    # print "After"
    # print3DMatrix(uStar, m, n, 2, "U")

    timestepTime = (freeSurfaceTime + positivityTime + huvTime + updateUTime + propSpeedTime + fluxTime + slopeSourceTime + shearSourceTime + buildRTime +
                   uStarTime + bcTimeStar + freeSurfaceTimeStar + positivityTimeStar + huvTimeStar + updateUstarTime + propSpeedStarTime + fluxStarTime + slopeSourceStarTime +
                   shearSourceStarTime + buildRStarTime + buildUnextTime + bcTime + fullTime + fullTimeStar)

    simTime += timestepTime

    # print "Total timestep calculation time: " + str(timestepTime) + " sec"
    time += dt
    iterations += 1

if saveOutput:
    closeCustomFort63(workingDir, fort63, meshU, iterations)
    print "Finished. " + str(iterations) + " timesteps saved."
print "Total simulation time: " + str(simTime) + " seconds"

# freeSurfaceTime = reconstructFreeSurfaceTimed(meshUGPU, meshUIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
# positivityTime = preservePositivityTimed(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU, m, n, [blockDim, blockDim], [gridN, gridM])
# huvTime = calculateHUVTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
# updateUTime = updateUIntPtsTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
# propSpeedTime = calculatePropSpeedsTimed(meshPropSpeedsGPU, meshHUVIntPtsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
# fluxTime = fluxSolverTimed(meshFluxesGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, meshPropSpeedsGPU, m, n, [blockDim, blockDim], [gridN, gridM])
# slopeSourceTime = solveBedSlopeTimed(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
# shearSourceTime = solveBedShearTimed(meshShearSourceGPU, meshUGPU, meshBottomIntPtsGPU, m, n, cellWidth, cellHeight, [blockDim, blockDim], [gridN, gridM])
# buildRTime = buildRValuesTimed(meshRValuesGPU, meshFluxesGPU, meshSlopeSourceGPU, m, n, [blockDim, blockDim], [gridN, gridM])
# timestep = calculateTimestep(meshPropSpeedsGPU, cellWidth)
# print "Timestep: " + str(timestep)
# uStarTime = buildUstarTimed(meshUstarGPU, meshUGPU, meshRValuesGPU, meshShearSourceGPU, timestep, m, n, [blockDim, blockDim], [gridN, gridM])

# meshUIntPts = meshUIntPtsGPU.get()
# huvIntPts = meshHUVIntPtsGPU.get()
# propSpeeds = meshPropSpeedsGPU.get()
# fluxes = meshFluxesGPU.get()
# slopeSource = meshSlopeSourceGPU.get()
# shearSource = meshShearSourceGPU.get()
# RValues = meshRValuesGPU.get()


# print "Time to reconstruct free-surface:\t" + str(freeSurfaceTime) + " sec"
# print "Time to preserve positivity:\t\t" + str(positivityTime) + " sec"
# print "Time to calculate huv:\t\t\t" + str(huvTime) + " sec"
# print "Time to update U at integration points:\t" + str(updateUTime) + " sec"
# print "Time to calculate propagation speeds:\t" + str(propSpeedTime) + " sec"
# print "Time to calculate fluxes:\t\t" + str(fluxTime) + " sec"
# print "Time to calculate slope source:\t\t" + str(slopeSourceTime) + " sec"
# print "Time to calculate shear source:\t\t" + str(shearSourceTime) + " sec"
# print "Time to build R-values:\t\t\t" + str(buildRTime) + " sec"
# print "Time to build Unext:\t\t\t" + str(uStarTime) + " sec"
# print "\nTotal time:\t" + str(freeSurfaceTime + positivityTime + huvTime + updateUTime + propSpeedTime + fluxTime + slopeSourceTime + shearSourceTime + buildRTime + uStarTime)

# direction = 2
# printCellCenteredMatrix(meshU, m, n, 'meshU')
# print2DirectionInterfaceMatrix(meshBottomIntPts, m, n, direction, 'meshBottomIntPts')
# print4DirectionCellMatrix(meshUIntPts, m, n, direction, 'meshUIntPts', 0)
# print4DirectionCellMatrix(huvIntPts, m, n, direction, "huvIntPts", 0)
# print4DirectionCellMatrix(propSpeeds, m, n, 2, "propSpeeds")
# print4DirectionCellMatrix(fluxes, m, n, 1, "fluxes", 1)
# print3DMatrix(slopeSource, m, n, 0, "slopeSource")
# printCellCenteredMatrix(shearSource, m, n, "shearSource")
# printCellCenteredMatrix(RValues, m, n, "RValues", 1)

