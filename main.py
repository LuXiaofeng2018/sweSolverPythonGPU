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
from modelChecker import *

printGPUMemUsage()

# Build test mesh
m = 64  # number of rows
n = 64  # number of columns
freeSurface = 1.5
cellWidth = 1.0
cellHeight = 1.0

# Calculate GPU grid/block sizes
blockDim = 16
gridM = m / blockDim + (1 if (m % blockDim != 0) else 0)
gridN = n / blockDim + (1 if (n % blockDim != 0) else 0)

meshBottomIntPts = np.zeros((m + 1, n + 1, 2))
for i in range(m + 1):
    for j in range(n + 1):
        if j < n / 2:
            if j == n / 2 - 1:
                meshBottomIntPts[i][j][0] = j - 5.0
                meshBottomIntPts[i][j][1] = j - 5.0
            else:
                meshBottomIntPts[i][j][0] = j + 0.5 - 5.0
                meshBottomIntPts[i][j][1] = j - 5.0
        else:
            meshBottomIntPts[i][j][0] = n - j - 1 - 0.5 - 5.0
            meshBottomIntPts[i][j][1] = n - j - 1 - 5.0

meshU = np.zeros((m, n, 3))
for i in range(m):
    for j in range(n):
        if meshBottomIntPts[i][j][0] <= 0.0:
            meshU[i][j][0] = 0.0
        else:
            meshU[i][j][0] = meshBottomIntPts[i][j][0]


meshUGPU = sendToGPU(meshU)
meshUIntPtsGPU = gpuarray.zeros((m, n, 4, 3), np.float32)  # Empty U for in-cell integration points
meshBottomIntPtsGPU = sendToGPU(meshBottomIntPts)  # gpuarray.zeros((m + 1, n + 1, 2), np.float32)  # Flat bottom with zero elevation
meshHUVIntPtsGPU = gpuarray.zeros((m, n, 4, 3), np.float32)
meshPropSpeedsGPU = gpuarray.zeros((m, n, 4), np.float32)
meshFluxesGPU = gpuarray.zeros((m, n, 2, 3), np.float32)
meshSlopeSourceGPU = gpuarray.zeros((m, n, 2), np.float32)

printGPUMemUsage()

# Get function handles from GPU module
reconstructFreeSurface = spacialModule.get_function("reconstructFreeSurface")
preservePositivity = spacialModule.get_function("preservePositivity")
calculateHUV = spacialModule.get_function("calculateHUV")
updateUIntPts = spacialModule.get_function("updateUIntPts")
calculatePropagationSpeeds = spacialModule.get_function("calculatePropagationSpeeds")
fluxSolver = fluxModule.get_function("fluxSolver")
bedSlopeSourceSolver = sourceModule.get_function("bedSlopeSourceSolver")

freeSurfaceTime = reconstructFreeSurface(meshUGPU, meshUIntPtsGPU, np.int32(m), np.int32(n), np.float32(cellWidth), np.float32(cellHeight), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)
positivityTime = preservePositivity(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU, np.int32(m), np.int32(n), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)
huvTime = calculateHUV(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, np.int32(m), np.int32(n), np.float32(cellWidth), np.float32(cellHeight), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)
updateUTime = updateUIntPts(meshHUVIntPtsGPU, meshUIntPtsGPU, np.int32(m), np.int32(n), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)
propSpeedTime = calculatePropagationSpeeds(meshPropSpeedsGPU, meshHUVIntPtsGPU, np.int32(m), np.int32(n), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)
fluxTime = fluxSolver(meshFluxesGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, meshPropSpeedsGPU, np.int32(m), np.int32(n), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)
slopeSourceTime = bedSlopeSourceSolver(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, np.int32(m), np.int32(n), np.float32(cellWidth), np.float32(cellHeight), block=(blockDim, blockDim, 1), grid=(gridN, gridM), time_kernel=True)

# meshUIntPts = meshUIntPtsGPU.get()
# huvIntPts = meshHUVIntPtsGPU.get()
# propSpeeds = meshPropSpeedsGPU.get()
# fluxes = meshFluxesGPU.get()
slopeSource = meshSlopeSourceGPU.get()

print "Time to reconstruct free-surface:\t" + str(freeSurfaceTime) + " sec"
print "Time to preserve positivity:\t\t" + str(positivityTime) + " sec"
print "Time to calculate huv:\t\t\t" + str(huvTime) + " sec"
print "Time to update U at integration points:\t" + str(updateUTime) + " sec"
print "Time to calculate propagation speeds:\t" + str(propSpeedTime) + " sec"
print "Time to calculate fluxes:\t\t" + str(fluxTime) + " sec"
print "Time to calculate slope source:\t\t" + str(slopeSourceTime) + " sec"
print "\nTotal time:\t" + str(freeSurfaceTime + positivityTime + huvTime + updateUTime + propSpeedTime + fluxTime + slopeSourceTime)

direction = 2
# printCellCenteredMatrix(meshU, m, n, 'meshU')
# print2DirectionInterfaceMatrix(meshBottomIntPts, m, n, direction, 'meshBottomIntPts')
# print4DirectionCellMatrix(meshUIntPts, m, n, direction, 'meshUIntPts', 0)
# print4DirectionCellMatrix(huvIntPts, m, n, direction, "huvIntPts", 0)
# print4DirectionCellMatrix(propSpeeds, m, n, 2, "propSpeeds")
# print4DirectionCellMatrix(fluxes, m, n, 1, "fluxes", 1)
# print3DMatrix(slopeSource, m, n, 0, "slopeSource")

