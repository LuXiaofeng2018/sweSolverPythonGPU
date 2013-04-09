'''
Created on Apr 9, 2013

@author: tristan
'''
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
            float slopeY = (meshBottomIntPts[bottomCellIndex + (n+1)*2] - meshBottomIntPts[bottomCellIndex]) / dy;    // slope in y-dire
            meshBedSlope[slopeCellIndex] =  -9.81f * slopeX * hX; // slope source in x-dir
            meshBedSlope[slopeCellIndex+1] = -9.81f * slopeY * hY; // slope source in y-dir
        }
    } 

""")
