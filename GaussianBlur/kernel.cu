#include "kernel.cuh"
#include "CudaFramework.h"
#include "device_launch_parameters.h"
//Try to load 
__global__ void imageBlurKernel(RGBPixel* pixels, int** kernel, int divisor, int w, int h, int halo)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w - halo && row < h - halo &&
        col >= halo && row >= halo)
    {
        int i = row * w + col;
        unsigned rRes = 0;
        unsigned gRes = 0;
        unsigned bRes = 0;

        for (int c = -halo; c < halo; c++)
        {
            int kRow = halo + c;
            for (int r = -halo; r < halo; r++)
            {
                int kCol = halo + r;
                int val = kernel[kRow][kCol];
                int index = i + (w * c) + r;
                RGBPixel pixel = pixels[index];
                rRes += pixel.r * val;
                gRes += pixel.g * val;
                bRes += pixel.b * val;
            }
        }

        RGBPixel pixel = pixels[i];
        pixel.r = rRes / divisor;
        pixel.g = gRes / divisor;
        pixel.b = bRes / divisor;
    }
}

__global__ void tiledImageBlurKernel()
{
  //Implement
}

void getVersion()
{
    // Maybe will make this method return specific things depending on your GPU.
    int version;
    cudaRuntimeGetVersion(&version);
    int major = version / 1000;
    int minor = version / 10;
}

// Helper function for Gaussian Blur
cudaError_t deviceBlur(GaussianKernel& gaussian, ImgData& data, int passes)
{


    // Implement
    // unsigned char* img_buffer_d;
    // unsigned h_d;
    // unsigned w_d;
    
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    
    // Calculated using Nvisidia NSight Compute
    int threads = 8;
    int x = 4;
    int y = 4;
    dim3 Block(threads, threads, 1);
    dim3 Grid(x, y ,1);

    // Launch a kernel on the GPU with one thread for each element.
    //imageBlurKernel<<<Grid, Block>>>();

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
