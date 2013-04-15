'''
Created on Apr 15, 2013

@author: tristan
'''
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

boundaryConditionsModule = SourceModule("""

    __global__ void applyWallBoundaries(float *meshU, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < 2)
        {
            meshU[row*n*3 + col*3] = meshU[(3-row)*n*3 + col*3];
            meshU[row*n*3 + col*3 + 2] = meshU[(3-row)*n*3 + col*3 +2];
        }
        else if (row > m-3 && row < m)
        {
            meshU[row*n*3 + col*3] = meshU[(2*m-5-row)*n*3 + col*3];
            meshU[row*n*3 + col*3 + 2] = meshU[(2*m-5-row)*n*3 + col*3 + 2];
        }
        
        if (col < 2)
        {
            meshU[row*n*3 + col*3] = meshU[row*n*3 + (3-col)*3];
            meshU[row*n*3 + col*3 + 1] = meshU[row*n*3 + (3-col)*3 + 1];
        }
        else if (col > n-3 && col < n)
        {
            meshU[row*n*3 + col*3] = meshU[row*n*3 + (2*n-5-col)*3];
            meshU[row*n*3 + col*3 + 1] = meshU[row*n*3 + (2*n-5-col)*3 + 1];
        }
    }

""")

wallBoundaries = boundaryConditionsModule.get_function("applyWallBoundaries")

def applyWallBoundaries(meshUGPU, m, n, blockDims, gridDims):

    wallBoundaries(meshUGPU,
                   np.int32(m), np.int32(n),
                   block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def applyWallBoundariesTimed(meshUGPU, m, n, blockDims, gridDims):

    return wallBoundaries(meshUGPU,
                          np.int32(m), np.int32(n),
                          block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)
