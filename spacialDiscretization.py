'''
Created on Apr 3, 2013

@author: tristan
'''

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


spacialModule = SourceModule("""

    __device__ float minmod(float a, float b, float c)
    {
        float ab = fminf(fabsf(a), fabsf(b)) * (copysignf(1.0f, a) + copysignf(1.0f, b)) * 0.5f;
        return fminf(fabsf(ab), fabsf(c)) * (copysignf(1.0f, ab) + copysignf(1.0f, c)) * 0.5f;
    }

    __global__ void reconstructFreeSurface(float *meshU, float *meshUIntPts, int m, int n, float cellWidth, float cellHeight)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int currIndexU = row*n*3 + col*3;
        int upIndexU = (row+1)*n*3 + col*3;
        int downIndexU = (row-1)*n*3 + col*3;
        int rightIndexU = row*n*3 + (col+1)*3;
        int leftIndexU = row*n*3 + (col-1)*3;
        
        int NUIndex = row*n*4*3 + col*4*3 + 0*3 + 0;
        int SUIndex = row*n*4*3 + col*4*3 + 1*3 + 0;
        int EUIndex = row*n*4*3 + col*4*3 + 2*3 + 0;
        int WUIndex = row*n*4*3 + col*4*3 + 3*3 + 0;
        
        if (col < n-1 && row < m-1)
        {
            
            float forward, central, backward, slope;
        
            for (int i=0; i<3; i++)
            {
                // North and South
                forward = (meshU[upIndexU + i] - meshU[currIndexU + i])/cellHeight;
                central = (meshU[upIndexU + i] - meshU[downIndexU + i])/(2*cellHeight);
                backward = (meshU[currIndexU + i] - meshU[downIndexU + i])/cellHeight;
                slope = minmod(1.3f*forward, central, 1.3f*backward);
                
                meshUIntPts[NUIndex + i] = meshU[currIndexU + i] + (cellHeight/2.0f)*slope;
                meshUIntPts[SUIndex + i] = meshU[currIndexU + i] - (cellHeight/2.0f)*slope;
            
                // East and West
                forward = (meshU[rightIndexU + i] - meshU[currIndexU + i])/cellWidth;
                central = (meshU[rightIndexU + i] - meshU[leftIndexU + i])/(2*cellWidth);
                backward = (meshU[currIndexU + i] - meshU[leftIndexU + i])/cellWidth;
                slope = minmod(1.3f*forward, central, 1.3f*backward);
                
                meshUIntPts[EUIndex + i] = meshU[currIndexU + i] + (cellWidth/2.0f)*slope;
                meshUIntPts[WUIndex + i] = meshU[currIndexU + i] - (cellWidth/2.0f)*slope;
            }
        } 
    }
    
    __global__ void preservePositivity(float *meshUIntPts, float *meshBottomIntPts, float *meshU, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int cellIndex = row*(n+1)*2 + col*2;
        int northIndex = row*n*4*3 + col*4*3 + 0*3;
        int southIndex = row*n*4*3 + col*4*3 + 1*3;
        int eastIndex = row*n*4*3 + col*4*3 + 2*3;
        int westIndex = row*n*4*3 + col*4*3 + 3*3;
        
        if (col < n-1 && row < m-1)
        {
            if (meshUIntPts[northIndex] < meshBottomIntPts[cellIndex + (n+1)*2])
            {
                meshUIntPts[northIndex] = meshBottomIntPts[cellIndex + (n+1)*2];
                meshUIntPts[southIndex] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[cellIndex + (n+1)*2];
            }
            else if (meshUIntPts[southIndex] < meshBottomIntPts[cellIndex])
            {
                meshUIntPts[southIndex] = meshBottomIntPts[cellIndex];
                meshUIntPts[northIndex] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[cellIndex];
            }
            
            if (meshUIntPts[eastIndex] < meshBottomIntPts[cellIndex+3])
            {
                meshUIntPts[eastIndex] = meshBottomIntPts[cellIndex+3];
                meshUIntPts[westIndex] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[cellIndex+3];
            }
            else if (meshUIntPts[westIndex] < meshBottomIntPts[cellIndex+1])
            {
                meshUIntPts[westIndex] = meshBottomIntPts[cellIndex+1];
                meshUIntPts[eastIndex] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[cellIndex+1];
            }
        }
    }
    
    
    __global__ void calculateHUV(float *huvIntPts, float *meshUIntPts, float *meshBottomIntPts, int m, int n, float dx, float dy)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int cellIndex = row*(n+1)*2 + col*2;
        int northIndex = row*n*4*3 + col*4*3 + 0*3;
        int southIndex = row*n*4*3 + col*4*3 + 1*3;
        int eastIndex = row*n*4*3 + col*4*3 + 2*3;
        int westIndex = row*n*4*3 + col*4*3 + 3*3;
        
        float Kappa = 0.01f * fmaxf(1.0f, fminf(dx, dy));
        float sqrt2 = sqrtf(2.0f);
        
        if (col < n-1 && row < m-1)
        {
            // Calculate h at the four integration points
            huvIntPts[northIndex] = meshUIntPts[northIndex] - meshBottomIntPts[cellIndex + (n+1)*2];
            huvIntPts[southIndex] = meshUIntPts[southIndex] - meshBottomIntPts[cellIndex];
            huvIntPts[eastIndex] = meshUIntPts[eastIndex] - meshBottomIntPts[cellIndex+3];
            huvIntPts[westIndex] = meshUIntPts[westIndex] - meshBottomIntPts[cellIndex+1];
            
            // Calculate u at the four integration points
            huvIntPts[northIndex + 1] = (sqrt2 * huvIntPts[northIndex] * meshUIntPts[northIndex + 1]) / sqrtf(powf(huvIntPts[northIndex], 4.0f) + fmaxf(powf(huvIntPts[northIndex], 4.0f), Kappa));
            huvIntPts[southIndex + 1] = (sqrt2 * huvIntPts[southIndex] * meshUIntPts[southIndex + 1]) / sqrtf(powf(huvIntPts[southIndex], 4.0f) + fmaxf(powf(huvIntPts[southIndex], 4.0f), Kappa));
            huvIntPts[eastIndex + 1] = (sqrt2 * huvIntPts[eastIndex] * meshUIntPts[eastIndex + 1]) / sqrtf(powf(huvIntPts[eastIndex], 4.0f) + fmaxf(powf(huvIntPts[eastIndex], 4.0f), Kappa));
            huvIntPts[westIndex + 1] = (sqrt2 * huvIntPts[westIndex] * meshUIntPts[westIndex + 1]) / sqrtf(powf(huvIntPts[westIndex], 4.0f) + fmaxf(powf(huvIntPts[westIndex], 4.0f), Kappa));
            
            // Calculate v at the four integration points
            huvIntPts[northIndex + 2] = (sqrt2 * huvIntPts[northIndex] * meshUIntPts[northIndex + 2]) / sqrtf(powf(huvIntPts[northIndex], 4.0f) + fmaxf(powf(huvIntPts[northIndex], 4.0f), Kappa));
            huvIntPts[southIndex + 2] = (sqrt2 * huvIntPts[southIndex] * meshUIntPts[southIndex + 2]) / sqrtf(powf(huvIntPts[southIndex], 4.0f) + fmaxf(powf(huvIntPts[southIndex], 4.0f), Kappa));
            huvIntPts[eastIndex + 2] = (sqrt2 * huvIntPts[eastIndex] * meshUIntPts[eastIndex + 2]) / sqrtf(powf(huvIntPts[eastIndex], 4.0f) + fmaxf(powf(huvIntPts[eastIndex], 4.0f), Kappa));
            huvIntPts[westIndex + 2] = (sqrt2 * huvIntPts[westIndex] * meshUIntPts[westIndex + 2]) / sqrtf(powf(huvIntPts[westIndex], 4.0f) + fmaxf(powf(huvIntPts[westIndex], 4.0f), Kappa));
        }
    }
    
    
    
    
    
    __global__ void rFSshared(float *meshU, float *meshUIntPts, int m, int n, float cellWidth, float cellHeight)
    {
        extern __shared__ float blockU[];
        
        int blockRow = blockIdx.y*blockDim.y;
        int blockCol = blockIdx.x*blockDim.x;
        int row = threadIdx.y;
        int col = threadIdx.x;
        
        blockU[blockDim.x*row*3 + col*3 + 0] = meshU[(blockRow+row)*n*3 + (blockCol+col)*3 + 0];
        blockU[blockDim.x*row*3 + col*3 + 1] = meshU[(blockRow+row)*n*3 + (blockCol+col)*3 + 1];
        blockU[blockDim.x*row*3 + col*3 + 2] = meshU[(blockRow+row)*n*3 + (blockCol+col)*3 + 2];
        
        __syncthreads();
        
        int currIndexU = row*blockDim.x*3 + col*3;
        int upIndexU = (row+1)*blockDim.x*3 + col*3;
        int downIndexU = (row-1)*blockDim.x*3 + col*3;
        int rightIndexU = row*blockDim.x*3 + (col+1)*3;
        int leftIndexU = row*blockDim.x*3 + (col-1)*3;
        
        int NUIndex = (blockRow+row)*n*4*3 + (blockCol+col)*4*3 + 0*3 + 0;
        int SUIndex = (blockRow+row)*n*4*3 + (blockCol+col)*4*3 + 1*3 + 0;
        int EUIndex = (blockRow+row)*n*4*3 + (blockCol+col)*4*3 + 2*3 + 0;
        int WUIndex = (blockRow+row)*n*4*3 + (blockCol+col)*4*3 + 3*3 + 0;
        
        if (row > 0 && row < blockDim.y-1 && col > 0 && col < blockDim.x-1)
        {
            for (int i=0; i<3; i++)
            {
                // North and South
                float forward = (blockU[upIndexU + i] - blockU[currIndexU + i])/cellHeight;
                float central = (blockU[upIndexU + i] - blockU[downIndexU + i])/(2*cellHeight);
                float backward = (blockU[currIndexU + i] - blockU[downIndexU + i])/cellHeight;
                float slope = minmod(1.3f*forward, central, 1.3f*backward);
                
                meshUIntPts[NUIndex + i] = blockU[currIndexU + i] + (cellHeight/2.0f)*slope;
                meshUIntPts[SUIndex + i] = blockU[currIndexU + i] - (cellHeight/2.0f)*slope;
            
                // East and West
                forward = (blockU[rightIndexU + i] - blockU[currIndexU + i])/cellWidth;
                central = (blockU[rightIndexU + i] - blockU[leftIndexU + i])/(2*cellWidth);
                backward = (blockU[currIndexU + i] - blockU[leftIndexU + i])/cellWidth;
                slope = minmod(1.3f*forward, central, 1.3f*backward);
                
                meshUIntPts[EUIndex + i] = blockU[currIndexU + i] + (cellWidth/2.0f)*slope;
                meshUIntPts[WUIndex + i] = blockU[currIndexU + i] - (cellWidth/2.0f)*slope;
            }
        }
    }

""")



