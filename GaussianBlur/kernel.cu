#include "kernel.cuh"
#include "common.h"
#include "CudaFramework.h"
#include "device_launch_parameters.h"
//Try to load 
__global__ void imageKernel()
{
    //Implement
}

__global__ void tiledImageKernel()
{
  //Implement
}


// Helper function for Gaussian Blur
cudaError_t deviceBlur(unsigned char* img_buffer, unsigned h, unsigned w)
{
    // Implement
      cudaError_t cudaStatus;
   // unsigned char* img_buffer_d;
   // unsigned h_d;
   // unsigned w_d;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    
    int threads = 32;
    int x = -1;
    int y = -1;
    dim3 Block(threads, threads, 1);
    dim3 Grid(x, y ,1);
    // Launch a kernel on the GPU with one thread for each element.
    imageKernel<<<Grid, Block>>>();

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
