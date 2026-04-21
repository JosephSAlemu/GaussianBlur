#include "CudaFramework.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//define method signature above main for NVCC to recognize it and allow you to move method below main method.
cudaError_t guassianBlur();

__global__ void ImageKernel()
{
    //Implement
}

int main()
{
    Cuda::Trace::out("Testing");
    guassianBlur();
    return 0;
}

// Helper function for Gaussian Blur
cudaError_t guassianBlur()
{
    // Implement
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    
    int threads = 256;
    int x = -1;
    int y = -1;
    dim3 Grid(x, y ,1);
    dim3 Block(threads, 1, 1);
    // Launch a kernel on the GPU with one thread for each element.
    //ImageKernel<<<Grid, Block>>>();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

Error:
    
    return cudaStatus;
}
