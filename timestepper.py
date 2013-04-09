'''
Created on Apr 9, 2013

@author: tristan
'''
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

timeModule = SourceModule("""

    // The propagation speeds matrix is not used past this point in a timestep, so we can do a destructive reduction on it
    // to determine the maximum propagation speed
    // Call this kernel in a 1-D block of size (1, mxn)
    __global__ void calculateTimestep(float *propSpeeds, int m, int n, float dx, float dy)
    {
    
        int cellIndex = threadIdx.x
        int stride;
        
        for (int i=1; i < 4*m*n; i++)
        {
            
        }
        
        
        
    }

""")
