'''
Created on Apr 3, 2013

@author: tristan
'''

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


def sendToGPU(numpyArray):

    numpyArray = numpyArray.astype(np.float32)
    arrayGPU = gpuarray.to_gpu(numpyArray)
    return arrayGPU

