'''
Created on Apr 9, 2013

@author: tristan
'''
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

sourceModule = SourceModule("""

    // Bed slope matrix is only going to store 2 values because the third value is always 0
    __global__ void bedSlopeSourceSolver(float *meshBedSlope, float *meshUIntPts, float *meshBottomIntPts, int m, int n, float dx, float dy)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        
        int slopeCellIndex = row*n*2 + col*2;
        int uCellIndex = row*n*4*3 + col*4*3;
        int bottomCellIndex = row*(n+1)*2 + col*2;
        
        if (col < n-1 && row < m-1)
        {
            float hX = ((meshUIntPts[uCellIndex + 2*3] - meshBottomIntPts[bottomCellIndex+3]) + (meshUIntPts[uCellIndex + 3*3] - meshBottomIntPts[bottomCellIndex+1])) / 2.0f;    // h at cell center
            float hY = ((meshUIntPts[uCellIndex] - meshBottomIntPts[bottomCellIndex + (n+1)*2]) + (meshUIntPts[uCellIndex + 1*3] - meshBottomIntPts[bottomCellIndex])) / 2.0f;    // h at cell center
            float slopeX = (meshBottomIntPts[bottomCellIndex+3] - meshBottomIntPts[bottomCellIndex+1]) / dx;        // slope in x-dir
            float slopeY = (meshBottomIntPts[bottomCellIndex + (n+1)*2] - meshBottomIntPts[bottomCellIndex]) / dy;    // slope in y-dir
            meshBedSlope[slopeCellIndex] =  -9.81f * slopeX * hX; // slope source in x-dir
            meshBedSlope[slopeCellIndex+1] = -9.81f * slopeY * hY; // slope source in y-dir
        }
    }
    
    // Bed shear is a 2-D array of values, one value for each cell because the 0-index is always 0.0, and the 1- and 2-index are equivalent
    __global__ void bedShearSourceSolver(float *meshBedShear, float *meshU, float *meshBottomIntPts, int m, int n, float dx, float dy)
    {
        float manningsN = 0.03;
        float sqrt2 = sqrtf(2.0f);
        float Kappa = 0.01f * fmaxf(1.0f, fminf(dx, dy));
        
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        
        if (col < n-1 && row < m-1)
        {
            float h = meshU[row*n*3 + col*3] - (meshBottomIntPts[row*(n+1)*2 + (col+1)*2 + 1] + meshBottomIntPts[row*(n+1)*2 + col*2 + 1]) / 2.0f;
            if (h > 0.0f)
            {
                float denom = sqrtf(powf(h, 2.0f) + fmaxf(powf(h, 4.0f), Kappa));
                float u = (sqrt2 * h * meshU[row*n*3 + col*3 + 1]) / denom;
                float v = (sqrt2 * h * meshU[row*n*3 + col*3 + 2]) / denom;
                float Cz = powf(h, 1.0f/6.0f) / manningsN;
                float solution = (-9.81f * sqrtf(powf(u, 2.0f) + powf(v, 2.0f))) / (h * powf(Cz, 2.0f));
                meshBedShear[row*n + col] = solution;
            } else {
                meshBedShear[row*n + col] = 0.0f;
            }
        }
        
    }
    
    // Wind shear matrix will only store 2 values because the third value is always 0
    __global__ void windShearSourceSolver(float *meshWindShear, float *meshWindSpeeds, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        
        float Wc = 5.6f;
        
        if (col < n-1 && row < m-1)
        {
            float windSpeed = meshWindSpeeds[row*n*2 + col*2];    // Magnitude of wind speed
            float theta = meshWindSpeeds[row*n*2 + col*2 + 1];    // Angle (in radians) from positive x-dir
            
            float k = 0.0f;
            if (windSpeed <= Wc)
            {
                k = 0.0000012f;
            } else {
                k = 0.0000012f + 0.00000225f * powf(1.0f - (Wc / windSpeed), 2.0f)
            }
            
            meshWindShear[row*n*2 + col*2] = k * powf(windSpeed, 2.0f) * cosf(theta);
            meshWindShear[row*n*2 + col*2 + 1] = k * powf(windSpeed, 2.0f) * sinf(theta);
        }
    }

""")

bedSlopeSourceSolver = sourceModule.get_function("bedSlopeSourceSolver")
bedShearSourceSolver = sourceModule.get_function("bedShearSourceSolver")

def solveBedSlope(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    bedSlopeSourceSolver(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU,
                         np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                         block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def solveBedSlopeTimed(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    return bedSlopeSourceSolver(meshSlopeSourceGPU, meshUIntPtsGPU, meshBottomIntPtsGPU,
                                np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                                block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

def solveBedShear(meshBedShearGPU, meshUGPU, meshBottomIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    bedShearSourceSolver(meshBedShearGPU, meshUGPU, meshBottomIntPtsGPU,
                         np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                         block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def solveBedShearTimed(meshBedShearGPU, meshUGPU, meshBottomIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    return bedShearSourceSolver(meshBedShearGPU, meshUGPU, meshBottomIntPtsGPU,
                                np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                                block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

