'''
Created on Apr 9, 2013

@author: tristan
'''
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.cumath as cumath
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# This function assumes a square cell and destructively modifies
# the meshPropSpeedsGPU matrix
def calculateTimestep(meshPropSpeedsGPU, cellDim):


    maxPropSpeed = gpuarray.max(cumath.fabs(meshPropSpeedsGPU)).get()
    return cellDim / (4.0 * maxPropSpeed)

timeModule = SourceModule("""

    __global__ void buildUstar(float *meshUstar, float *meshU, float *meshR, float *meshShearSource, float dt, int m, int n)
    {
    
        int row = blockIdx.y * blockDim.y + threadIdx.y + 2;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 2;
        
        int uIndex = row*n*3 + col*3;
        
        if (row < m-2 && col < n-2)
        {
            meshUstar[uIndex] = meshU[uIndex] + dt*meshR[uIndex];
            meshUstar[uIndex + 1] = (meshU[uIndex + 1] + dt*meshR[uIndex + 1]) / (1.0f + dt*meshShearSource[row*n + col]);
            meshUstar[uIndex + 2] = (meshU[uIndex + 2] + dt*meshR[uIndex + 2]) / (1.0f + dt*meshShearSource[row*n + col]);
        }
        
    }
    
    __global__ void buildUnext(float *meshUnext, float *meshU, float *meshUstar, float *meshRstar, float *meshShearSourceStar, float dt, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 2;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 2;
        
        int uIndex = row*n*3 + col*3;
        
        if (row < m-2 && col < n-2)
        {
            meshUnext[uIndex] = 0.5f * meshU[uIndex] + 0.5f * (meshUstar[uIndex] + dt*meshRstar[uIndex]);
            meshUnext[uIndex+1] = (0.5f * meshU[uIndex+1] + 0.5f * (meshUstar[uIndex+1] + dt*meshRstar[uIndex+1])) / (1.0f + 0.5f * dt * meshShearSourceStar[row*n+col]);
            meshUnext[uIndex+2] = (0.5f * meshU[uIndex+2] + 0.5f * (meshUstar[uIndex+2] + dt*meshRstar[uIndex+2])) / (1.0f + 0.5f * dt * meshShearSourceStar[row*n+col]);
        } 
    }

""")


