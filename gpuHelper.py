'''
Created on Apr 5, 2013

@author: tristan
'''
import pycuda
import pycuda.driver as cuda

def printGPUMemUsage():
    memory = cuda.mem_get_info()
    print "Memory Usage: " + str(100 * (1 - float(memory[0]) / float(memory[1]))) + "%\t" + str(memory[0] / 1048576.0) + " MB Free"


def calculateGridBlockDims():

    threadsPerBlock = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
    gridDimX = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_GRID_DIM_X)
    gridDimY = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)
    multiprocessors = cuda.Device(0).get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    threadsPerProcessor = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)
    maxThreads = threadsPerProcessor * multiprocessors
    maxBlocks = maxThreads / threadsPerBlock
    print maxBlocks
