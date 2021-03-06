'''
Created on Apr 3, 2013

@author: tristan
'''

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


spacialModule = SourceModule("""

    __device__ float minmod(float a, float b, float c)
    {
        float ab = fminf(fabsf(a), fabsf(b)) * (copysignf(1.0f, a) + copysignf(1.0f, b)) * 0.5f;
        return fminf(fabsf(ab), fabsf(c)) * (copysignf(1.0f, ab) + copysignf(1.0f, c)) * 0.5f;
    }
    
    __global__ void FullPropSpeeds(float *meshU, float *meshBottomIntPts, float *meshUIntPts, float *huvIntPts, float *propSpeeds, int m, int n, float dx, float dy)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        
        float N[3], S[3], E[3], W[3];    // These are the [w, hu, hv] vectors
        float n_vec[3], s_vec[3], e_vec[3], w_vec[3];    // These are the [h, u, v] vectors
        
        if (col < n-1 && row < m-1)
        {
            float forward, central, backward, slope;
            float Kappa = 0.01f * fmaxf(1.0f, fminf(dx, dy));
            float sqrt2 = sqrtf(2.0f);
            float g = 9.81f;
            
            for (int i=0; i<3; i++)
            {
                // North and South
                forward = (meshU[(row+1)*n*3 + col*3 + i] - meshU[row*n*3 + col*3 + i])/dy;
                central = (meshU[(row+1)*n*3 + col*3 + i] - meshU[(row-1)*n*3 + col*3 + i])/(2*dy);
                backward = (meshU[row*n*3 + col*3 + i] - meshU[(row-1)*n*3 + col*3 + i])/dy;
                slope = minmod(1.3f*forward, central, 1.3f*backward);
                
                N[i] = meshU[row*n*3 + col*3 + i] + (dy/2.0f)*slope;
                S[i] = meshU[row*n*3 + col*3 + i] - (dy/2.0f)*slope;
            
                // East and West
                forward = (meshU[row*n*3 + (col+1)*3 + i] - meshU[row*n*3 + col*3 + i])/dx;
                central = (meshU[row*n*3 + (col+1)*3 + i] - meshU[row*n*3 + (col-1)*3 + i])/(2*dx);
                backward = (meshU[row*n*3 + col*3 + i] - meshU[row*n*3 + (col-1)*3 + i])/dx;
                slope = minmod(1.3f*forward, central, 1.3f*backward);
                
                E[i] = meshU[row*n*3 + col*3 + i] + (dx/2.0f)*slope;
                W[i] = meshU[row*n*3 + col*3 + i] - (dx/2.0f)*slope;
            }
            
            if (N[0] < meshBottomIntPts[row*(n+1)*2 + col*2 + (n+1)*2])
            {
                N[0] = meshBottomIntPts[row*(n+1)*2 + col*2 + (n+1)*2];
                S[0] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[row*(n+1)*2 + col*2 + (n+1)*2];
            }
            else if (S[0] < meshBottomIntPts[row*(n+1)*2 + col*2])
            {
                S[0] = meshBottomIntPts[row*(n+1)*2 + col*2];
                N[0] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[row*(n+1)*2 + col*2];
            }
            
            if (E[0] < meshBottomIntPts[row*(n+1)*2 + col*2+3])
            {
                E[0] = meshBottomIntPts[row*(n+1)*2 + col*2+3];
                W[0] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[row*(n+1)*2 + col*2+3];
            }
            else if (W[0] < meshBottomIntPts[row*(n+1)*2 + col*2+1])
            {
                W[0] = meshBottomIntPts[row*(n+1)*2 + col*2+1];
                E[0] = 2*meshU[row*n*3 + col*3] - meshBottomIntPts[row*(n+1)*2 + col*2+1];
            }
            
            // Calculate h at the four integration points
            n_vec[0] = N[0] - meshBottomIntPts[row*(n+1)*2 + col*2 + (n+1)*2];
            s_vec[0] = S[0] - meshBottomIntPts[row*(n+1)*2 + col*2];
            e_vec[0] = E[0] - meshBottomIntPts[row*(n+1)*2 + col*2+3];
            w_vec[0] = W[0] - meshBottomIntPts[row*(n+1)*2 + col*2+1];
            
            // Calculate u at the four integration points
            n_vec[1] = (sqrt2 * n_vec[0] * N[1]) / sqrtf(powf(n_vec[0], 4.0f) + fmaxf(powf(n_vec[0], 4.0f), Kappa));
            s_vec[1] = (sqrt2 * s_vec[0] * S[1]) / sqrtf(powf(s_vec[0], 4.0f) + fmaxf(powf(s_vec[0], 4.0f), Kappa));
            e_vec[1] = (sqrt2 * e_vec[0] * E[1]) / sqrtf(powf(e_vec[0], 4.0f) + fmaxf(powf(e_vec[0], 4.0f), Kappa));
            w_vec[1] = (sqrt2 * w_vec[0] * W[1]) / sqrtf(powf(w_vec[0], 4.0f) + fmaxf(powf(w_vec[0], 4.0f), Kappa));
            
            // Calculate v at the four integration points
            n_vec[2] = (sqrt2 * n_vec[0] * N[2]) / sqrtf(powf(n_vec[0], 4.0f) + fmaxf(powf(n_vec[0], 4.0f), Kappa));
            s_vec[2] = (sqrt2 * s_vec[0] * S[2]) / sqrtf(powf(s_vec[0], 4.0f) + fmaxf(powf(s_vec[0], 4.0f), Kappa));
            e_vec[2] = (sqrt2 * e_vec[0] * E[2]) / sqrtf(powf(e_vec[0], 4.0f) + fmaxf(powf(e_vec[0], 4.0f), Kappa));
            w_vec[2] = (sqrt2 * w_vec[0] * W[2]) / sqrtf(powf(w_vec[0], 4.0f) + fmaxf(powf(w_vec[0], 4.0f), Kappa));
            
            // Update hu and hv North
            N[1] = n_vec[0] * n_vec[1];
            N[2] = n_vec[0] * n_vec[2];
            
            // Update hu and hv South
            S[1] = s_vec[0] * s_vec[1];
            S[2] = s_vec[0] * s_vec[2];
            
            // Update hu and hv East
            E[1] = e_vec[0] * e_vec[1];
            E[2] = e_vec[0] * e_vec[2];
            
            // Update hu and hv West
            W[1] = w_vec[0] * w_vec[1];
            W[2] = w_vec[0] * w_vec[2];
            
            
            // Make sure all threads reach this point before continuing
            __syncthreads();
            
            
            // Push meshUIntPts and huv to global memory
            meshUIntPts[row*n*4*3 + col*4*3 + 0*3] = N[0];
            meshUIntPts[row*n*4*3 + col*4*3 + 0*3 + 1] = N[1];
            meshUIntPts[row*n*4*3 + col*4*3 + 0*3 + 2] = N[2];
            
            meshUIntPts[row*n*4*3 + col*4*3 + 1*3] = S[0];
            meshUIntPts[row*n*4*3 + col*4*3 + 1*3 + 1] = S[1];
            meshUIntPts[row*n*4*3 + col*4*3 + 1*3 + 2] = S[2];
            
            meshUIntPts[row*n*4*3 + col*4*3 + 2*3] = E[0];
            meshUIntPts[row*n*4*3 + col*4*3 + 2*3 + 1] = E[1];
            meshUIntPts[row*n*4*3 + col*4*3 + 2*3 + 2] = E[2];
            
            meshUIntPts[row*n*4*3 + col*4*3 + 3*3] = W[0];
            meshUIntPts[row*n*4*3 + col*4*3 + 3*3 + 1] = W[1];
            meshUIntPts[row*n*4*3 + col*4*3 + 3*3 + 2] = W[2];
            
            huvIntPts[row*n*4*3 + col*4*3 + 0*3] = n_vec[0];
            huvIntPts[row*n*4*3 + col*4*3 + 0*3 + 1] = n_vec[1];
            huvIntPts[row*n*4*3 + col*4*3 + 0*3 + 2] = n_vec[2];
            
            huvIntPts[row*n*4*3 + col*4*3 + 1*3] = s_vec[0];
            huvIntPts[row*n*4*3 + col*4*3 + 1*3 + 1] = s_vec[1];
            huvIntPts[row*n*4*3 + col*4*3 + 1*3 + 2] = s_vec[2];
            
            huvIntPts[row*n*4*3 + col*4*3 + 2*3] = e_vec[0];
            huvIntPts[row*n*4*3 + col*4*3 + 2*3 + 1] = e_vec[1];
            huvIntPts[row*n*4*3 + col*4*3 + 2*3 + 2] = e_vec[2];
            
            huvIntPts[row*n*4*3 + col*4*3 + 3*3] = w_vec[0];
            huvIntPts[row*n*4*3 + col*4*3 + 3*3 + 1] = w_vec[1];
            huvIntPts[row*n*4*3 + col*4*3 + 3*3 + 2] = w_vec[2];
            
            
            // Shouldnt need this, but just be sure that all global memory writes have finished
            __syncthreads();
            
            
            // North propagation speed of this cell
            propSpeeds[row*n*4 + col*4 + 0] = fminf(fminf(huvIntPts[row*n*4*3 + col*4*3 + 0*3+2] - sqrtf(g * huvIntPts[row*n*4*3 + col*4*3 + 0*3]), huvIntPts[(row+1)*n*4*3 + col*4*3 + 1*3+2] - sqrtf(g * huvIntPts[(row+1)*n*4*3 + col*4*3 + 1*3])), 0.0f);
            
            // South propagation speed of the cell above this one
            propSpeeds[(row+1)*n*4 + col*4 + 1] = fmaxf(fmaxf(huvIntPts[row*n*4*3 + col*4*3 + 0*3+2] + sqrtf(g * huvIntPts[row*n*4*3 + col*4*3 + 0*3]), huvIntPts[(row+1)*n*4*3 + col*4*3 + 1*3+2] + sqrtf(g * huvIntPts[(row+1)*n*4*3 + col*4*3 + 1*3])), 0.0f);
            
            // East propagation speed of this cell
            propSpeeds[row*n*4 + col*4 + 2] = fminf(fminf(huvIntPts[row*n*4*3 + col*4*3 + 2*3+1] - sqrtf(g * huvIntPts[row*n*4*3 + col*4*3 + 2*3]), huvIntPts[row*n*4*3 + (col+1)*4*3 + 3*3+1] - sqrtf(g * huvIntPts[row*n*4*3 + (col+1)*4*3 + 3*3])), 0.0f);
            
            // West propagation speed of the cell to the right of this one
            propSpeeds[row*n*4 + (col+1)*4 + 3] = fmaxf(fmaxf(huvIntPts[row*n*4*3 + col*4*3 + 2*3+1] + sqrtf(g * huvIntPts[row*n*4*3 + col*4*3 + 2*3]), huvIntPts[row*n*4*3 + (col+1)*4*3 + 3*3+1] + sqrtf(g * huvIntPts[row*n*4*3 + (col+1)*4*3 + 3*3])), 0.0f);
            
        }
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
    
    // Shared memory would be really easy in this kernel
    __global__ void updateUIntPts(float *huvIntPts, float *meshUIntPts, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int northIndex = row*n*4*3 + col*4*3 + 0*3;
        int southIndex = row*n*4*3 + col*4*3 + 1*3;
        int eastIndex = row*n*4*3 + col*4*3 + 2*3;
        int westIndex = row*n*4*3 + col*4*3 + 3*3;
        
        if (col < n-1 && row < m-1)
        {
            // Update hu and hv North
            meshUIntPts[northIndex+1] = huvIntPts[northIndex] * huvIntPts[northIndex+1];
            meshUIntPts[northIndex+2] = huvIntPts[northIndex] * huvIntPts[northIndex+2];
            
            // Update hu and hv South
            meshUIntPts[southIndex+1] = huvIntPts[southIndex] * huvIntPts[southIndex+1];
            meshUIntPts[southIndex+2] = huvIntPts[southIndex] * huvIntPts[southIndex+2];
            
            // Update hu and hv East
            meshUIntPts[eastIndex+1] = huvIntPts[eastIndex] * huvIntPts[eastIndex+1];
            meshUIntPts[eastIndex+2] = huvIntPts[eastIndex] * huvIntPts[eastIndex+2];
            
            // Update hu and hv West
            meshUIntPts[westIndex+1] = huvIntPts[westIndex] * huvIntPts[westIndex+1];
            meshUIntPts[westIndex+2] = huvIntPts[westIndex] * huvIntPts[westIndex+2];
        }
         
    }
    
    __global__ void calculatePropagationSpeeds(float *propSpeeds, float *huvIntPts, int m, int n)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
        int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
        int northIndex = row*n*4*3 + col*4*3 + 0*3;
        int southIndex = (row+1)*n*4*3 + col*4*3 + 1*3;
        int eastIndex = row*n*4*3 + col*4*3 + 2*3;
        int westIndex = row*n*4*3 + (col+1)*4*3 + 3*3;
        
        float g = 9.81f;
        
        if (col < n-1 && row < m-1)
        {
            // North propagation speed of this cell
            propSpeeds[row*n*4 + col*4 + 0] = fminf(fminf(huvIntPts[northIndex+2] - sqrtf(g * huvIntPts[northIndex]), huvIntPts[southIndex+2] - sqrtf(g * huvIntPts[southIndex])), 0.0f);
            
            // South propagation speed of the cell above this one
            propSpeeds[(row+1)*n*4 + col*4 + 1] = fmaxf(fmaxf(huvIntPts[northIndex+2] + sqrtf(g * huvIntPts[northIndex]), huvIntPts[southIndex+2] + sqrtf(g * huvIntPts[southIndex])), 0.0f);
            
            // East propagation speed of this cell
            propSpeeds[row*n*4 + col*4 + 2] = fminf(fminf(huvIntPts[eastIndex+1] - sqrtf(g * huvIntPts[eastIndex]), huvIntPts[westIndex+1] - sqrtf(g * huvIntPts[westIndex])), 0.0f);
            
            // West propagation speed of the cell to the right of this one
            propSpeeds[row*n*4 + (col+1)*4 + 3] = fmaxf(fmaxf(huvIntPts[eastIndex+1] + sqrtf(g * huvIntPts[eastIndex]), huvIntPts[westIndex+1] + sqrtf(g * huvIntPts[westIndex])), 0.0f);
        }
    }
    
    

    
    // This kernel doesn't work yet, don't use it
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


reconstructFreeSurfaceGPU = spacialModule.get_function("reconstructFreeSurface")
preservePositivityGPU = spacialModule.get_function("preservePositivity")
calculateHUVGPU = spacialModule.get_function("calculateHUV")
updateUIntPtsGPU = spacialModule.get_function("updateUIntPts")
calculatePropagationSpeedsGPU = spacialModule.get_function("calculatePropagationSpeeds")
FullPropSpeeds = spacialModule.get_function("FullPropSpeeds")

def fullPropSpeeds(meshUGPU, meshBottomIntPtsGPU, meshUIntPtsGPU, huvIntPtsGPU, propSpeeds, m, n, dx, dy, blockDims, gridDims):

    FullPropSpeeds(meshUGPU, meshBottomIntPtsGPU, meshUIntPtsGPU, huvIntPtsGPU, propSpeeds,
                   np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                   block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]),)

def fullPropSpeedsTimed(meshUGPU, meshBottomIntPtsGPU, meshUIntPtsGPU, huvIntPtsGPU, propSpeeds, m, n, dx, dy, blockDims, gridDims):

    return FullPropSpeeds(meshUGPU, meshBottomIntPtsGPU, meshUIntPtsGPU, huvIntPtsGPU, propSpeeds,
                          np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                              block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

def reconstructFreeSurface(meshUGPU, meshUIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    reconstructFreeSurfaceGPU(meshUGPU, meshUIntPtsGPU,
                              np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                              block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def reconstructFreeSurfaceTimed(meshUGPU, meshUIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    return reconstructFreeSurfaceGPU(meshUGPU, meshUIntPtsGPU,
                                     np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                                     block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

def preservePositivity(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU, m, n, blockDims, gridDims):

    preservePositivityGPU(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU,
                          np.int32(m), np.int32(n),
                          block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def preservePositivityTimed(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU, m, n, blockDims, gridDims):

    return preservePositivityGPU(meshUIntPtsGPU, meshBottomIntPtsGPU, meshUGPU,
                          np.int32(m), np.int32(n),
                          block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

def calculateHUV(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    calculateHUVGPU(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU,
                    np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                    block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def calculateHUVTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU, m, n, dx, dy, blockDims, gridDims):

    return calculateHUVGPU(meshHUVIntPtsGPU, meshUIntPtsGPU, meshBottomIntPtsGPU,
                    np.int32(m), np.int32(n), np.float32(dx), np.float32(dy),
                    block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

def updateUIntPts(meshHUVIntPtsGPU, meshUIntPtsGPU, m, n, blockDims, gridDims):

    updateUIntPtsGPU(meshHUVIntPtsGPU, meshUIntPtsGPU,
                    np.int32(m), np.int32(n),
                    block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def updateUIntPtsTimed(meshHUVIntPtsGPU, meshUIntPtsGPU, m, n, blockDims, gridDims):

    return updateUIntPtsGPU(meshHUVIntPtsGPU, meshUIntPtsGPU,
                    np.int32(m), np.int32(n),
                    block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

def calculatePropSpeeds(meshPropSpeedsGPU, meshHUVIntPtsGPU, m, n, blockDims, gridDims):

    calculatePropagationSpeedsGPU(meshPropSpeedsGPU, meshHUVIntPtsGPU,
                                  np.int32(m), np.int32(n),
                                  block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]))

def calculatePropSpeedsTimed(meshPropSpeedsGPU, meshHUVIntPtsGPU, m, n, blockDims, gridDims):

    return calculatePropagationSpeedsGPU(meshPropSpeedsGPU, meshHUVIntPtsGPU,
                                  np.int32(m), np.int32(n),
                                  block=(blockDims[0], blockDims[1], 1), grid=(gridDims[0], gridDims[1]), time_kernel=True)

