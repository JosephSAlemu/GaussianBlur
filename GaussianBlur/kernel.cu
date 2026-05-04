#include "kernel.cuh"
#include "CudaFramework.h"
#include "device_launch_parameters.h"

// As much as I hate it, probably better for CUDA to use constants as opposed to a double pointer.
__global__ void image5x5BlurKernel(RGBPixel* pixels, RGBPixel* out, int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w - 2 && row < h - 2 &&
        col >= 2 && row >= 2)
    {
        int i = row * w + col;
        unsigned rRes = 0;
        unsigned gRes = 0;
        unsigned bRes = 0;
         
        int twoZero = i - 2;

        // using constants to speedup CUDA rather than using a for loop
        // cause a 5x5 Gaussian kernel is fixed size and calling a 2D int array for gaussian blur
        // kernel might be slow with global memory accesses.
        int zeroZero = twoZero + (w * -2);
        int oneZero = twoZero + (w * -1);
        int threeZero = twoZero + (w * 1);
        int fourZero = twoZero + (w * 2);
        
        rRes += pixels[zeroZero].r;
        gRes += pixels[zeroZero].g;
        bRes += pixels[zeroZero].b;

        rRes += pixels[zeroZero + 1].r * 4;
        gRes += pixels[zeroZero + 1].g * 4;
        bRes += pixels[zeroZero + 1].b * 4;

        rRes += pixels[zeroZero + 2].r * 7;
        gRes += pixels[zeroZero + 2].g * 7;
        bRes += pixels[zeroZero + 2].b * 7;

        rRes += pixels[zeroZero + 3].r * 4;
        gRes += pixels[zeroZero + 3].g * 4;
        bRes += pixels[zeroZero + 3].b * 4;

        rRes += pixels[zeroZero + 4].r;
        gRes += pixels[zeroZero + 4].g;
        bRes += pixels[zeroZero + 4].b;


        rRes += pixels[oneZero].r * 4;
        gRes += pixels[oneZero].g * 4;
        bRes += pixels[oneZero].b * 4;

        rRes += pixels[oneZero + 1].r * 16;
        gRes += pixels[oneZero + 1].g * 16;
        bRes += pixels[oneZero + 1].b * 16;

        rRes += pixels[oneZero + 2].r * 26;
        gRes += pixels[oneZero + 2].g * 26;
        bRes += pixels[oneZero + 2].b * 26;

        rRes += pixels[oneZero + 3].r * 16;
        gRes += pixels[oneZero + 3].g * 16;
        bRes += pixels[oneZero + 3].b * 16;

        rRes += pixels[oneZero + 4].r * 4;
        gRes += pixels[oneZero + 4].g * 4;
        bRes += pixels[oneZero + 4].b * 4;


        rRes += pixels[twoZero].r * 7;
        gRes += pixels[twoZero].g * 7;
        bRes += pixels[twoZero].b * 7;

        rRes += pixels[twoZero + 1].r * 26;
        gRes += pixels[twoZero + 1].g * 26;
        bRes += pixels[twoZero + 1].b * 26;

        rRes += pixels[twoZero + 2].r * 41;
        gRes += pixels[twoZero + 2].g * 41;
        bRes += pixels[twoZero + 2].b * 41;

        rRes += pixels[twoZero + 3].r * 26;
        gRes += pixels[twoZero + 3].g * 26;
        bRes += pixels[twoZero + 3].b * 26;

        rRes += pixels[twoZero + 4].r * 7;
        gRes += pixels[twoZero + 4].g * 7;
        bRes += pixels[twoZero + 4].b * 7;


        rRes += pixels[threeZero].r * 4;
        gRes += pixels[threeZero].g * 4;
        bRes += pixels[threeZero].b * 4;

        rRes += pixels[threeZero + 1].r * 16;
        gRes += pixels[threeZero + 1].g * 16;
        bRes += pixels[threeZero + 1].b * 16;

        rRes += pixels[threeZero + 2].r * 26;
        gRes += pixels[threeZero + 2].g * 26;
        bRes += pixels[threeZero + 2].b * 26;

        rRes += pixels[threeZero + 3].r * 16;
        gRes += pixels[threeZero + 3].g * 16;
        bRes += pixels[threeZero + 3].b * 16;

        rRes += pixels[threeZero + 4].r * 4;
        gRes += pixels[threeZero + 4].g * 4;
        bRes += pixels[threeZero + 4].b * 4;


        rRes += pixels[fourZero].r;
        gRes += pixels[fourZero].g;
        bRes += pixels[fourZero].b;

        rRes += pixels[fourZero + 1].r * 4;
        gRes += pixels[fourZero + 1].g * 4;
        bRes += pixels[fourZero + 1].b * 4;

        rRes += pixels[fourZero + 2].r * 7;
        gRes += pixels[fourZero + 2].g * 7;
        bRes += pixels[fourZero + 2].b * 7;

        rRes += pixels[fourZero + 3].r * 4;
        gRes += pixels[fourZero + 3].g * 4;
        bRes += pixels[fourZero + 3].b * 4;

        rRes += pixels[fourZero + 4].r;
        gRes += pixels[fourZero + 4].g;
        bRes += pixels[fourZero + 4].b;

        out[i].r = rRes / 273;
        out[i].g = gRes / 273;
        out[i].b = bRes / 273;
    }
}

__global__ void tiledImage5x5BlurKernel()
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
cudaError_t deviceBlur(ImgData& out, int passes)
{
    // Implement
    RGBPixel* pixels_d;
    RGBPixel* out_d;
    int w_d = out.w;
    int h_d = out.h;
    
    cudaError_t cudaStatus;
    size_t len = sizeof(RGBPixel) * out.len;

    // Allocate GPU buffers for three vectors (two input, one output)
    // Dividing image dimensions by warp size. 1024 threads (max threads on 7.5 architecture) does give 100% warp occupancy.
    // If you don't understand what I said above, just pass through an image that's width and height are divisible by 32.
    int threads = 32;
    int blocksx = out.w / 32;
    int blocksy = out.h / 32;
    dim3 Block(threads, threads, 1);
    dim3 Grid(blocksx, blocksy, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Remember you only cudaMalloc for pointers. Instrinsic types can just be passed.
    cudaStatus = cudaMalloc((void**) &pixels_d, len);
    if (cudaStatus != cudaSuccess) 
    {
      fprintf(stderr, "CUDA malloc failed for pixels_d");
      goto Error;
    }

    cudaStatus = cudaMalloc((void**) &out_d, len);
    if (cudaStatus != cudaSuccess) 
    {
      fprintf(stderr, "CUDA malloc failed for out_d");
      goto Error;
    }

    for (int i = 0; i < passes; i++)
    {
      cudaStatus = cudaMemcpy(pixels_d, out.pixels, len, cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess)
      {
        fprintf(stderr, "CUDA memcpy failed for pixels_d");
        goto Error;
      }

      cudaStatus = cudaMemcpy(out_d, out.pixels, len, cudaMemcpyHostToDevice);
      if (cudaStatus != cudaSuccess)
      {
        fprintf(stderr, "CUDA memcpy failed for out_d");
        goto Error;
      }

      // Launch a kernel on the GPU with one thread for each element.
      image5x5BlurKernel << <Grid, Block >> > (pixels_d, out_d, w_d, h_d);

      // Check for any errors launching the kernel
      cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess)
      {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
      }

      // cudaDeviceSynchronize waits for the kernel to finish, and returns
      // any errors encountered during the launch.
      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess)
      {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
      }

      cudaStatus = cudaMemcpy(out.pixels, out_d, len, cudaMemcpyDeviceToHost);
      if (cudaStatus != cudaSuccess)
      {
        fprintf(stderr, "CUDA memcpy failed for data.pixels");
        goto Error;
      }
    }
    

    cudaFree(pixels_d);
    cudaFree(out_d);

    return cudaStatus;

Error:
    cudaFree(pixels_d);
    cudaFree(out_d);

    return cudaStatus;
}