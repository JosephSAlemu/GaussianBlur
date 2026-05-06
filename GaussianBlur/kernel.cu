#include "kernel.cuh"
#include "CudaFramework.h"
#include <device_launch_parameters.h>

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

        // Comment above is wrong. Look at #pragma unroll for effectively use loops in CUDA.
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

// Naive approach where I use hardcoded constants on ScareCrow.png dimensions
// Block_Width and Block_Length = 32
// Halo is 2 in all directions
// 36 x 36 blocks
#define TILE_WIDTH 36
__global__ void tiledImage5x5BlurKernel(RGBPixel* pixels, RGBPixel* out, int w, int h)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  // share declaration
  // in our shared cache we should have 49 pixels in total.
  // Our shared space will be blockDim.x + 2 and blockDim.y + 2
  // shared memory is unique for each for each separate block
  __shared__ unsigned char redTiles_ds   [TILE_WIDTH][TILE_WIDTH];
  __shared__ unsigned char greenTiles_ds [TILE_WIDTH][TILE_WIDTH];
  __shared__ unsigned char blueTiles_ds  [TILE_WIDTH][TILE_WIDTH];

  int i = row * w + col;

  // Code below is for tile offset
  int tCol = threadIdx.x + 2;
  int tRow = threadIdx.y + 2;

  // share load. Everyone is responsible for setting their own pixel unless you're on an edge
  redTiles_ds  [tRow][tCol] = pixels[i].r;
  greenTiles_ds[tRow][tCol] = pixels[i].g;
  blueTiles_ds [tRow][tCol] = pixels[i].b;
  

  // get two left pixels
  if (threadIdx.x == 0 && blockIdx.x != 0)
  {

#pragma unroll
    for (int k = 1; k <= 2; k++)
    {
      redTiles_ds  [tRow][tCol-k] = pixels[i-k].r;
      greenTiles_ds[tRow][tCol-k] = pixels[i-k].g;
      blueTiles_ds [tRow][tCol-k] = pixels[i-k].b;
    }

    // get two top pixels and four top left corner pixels
    if (threadIdx.y == 0 && blockIdx.y != 0)
    {

#pragma unroll
      for (int j = 1; j <= 2; j++)
      {
#pragma unroll
        for (int k = 0; k <= 2; k++)
        {
          redTiles_ds  [tRow-j][tCol-k] = pixels[i-(j*w)-k].r;
          greenTiles_ds[tRow-j][tCol-k] = pixels[i-(j*w)-k].g;
          blueTiles_ds [tRow-j][tCol-k] = pixels[i-(j*w)-k].b;
        }
      }

    }
    // get two bottom pixels and four bottom left corner pixels
    else if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1)
    {

#pragma unroll
      for (int j = 1; j <= 2; j++)
      {
#pragma unroll
        for (int k = 0; k <= 2; k++)
        {
          redTiles_ds  [tRow+j][tCol-k] = pixels[i+(j*w)-k].r;
          greenTiles_ds[tRow+j][tCol-k] = pixels[i+(j*w)-k].g;
          blueTiles_ds [tRow+j][tCol-k] = pixels[i+(j*w)-k].b;
        }
      }

    }

  }
  // get two right pixels
  else if (threadIdx.x == blockDim.x - 1 && blockIdx.x != gridDim.x - 1)
  {

#pragma unroll
    for (int k = 1; k <= 2; k++)
    {
      redTiles_ds  [tRow][tCol+k] = pixels[i+k].r;
      greenTiles_ds[tRow][tCol+k] = pixels[i+k].g;
      blueTiles_ds [tRow][tCol+k] = pixels[i+k].b;
    }

    // get two top pixels and four top right corner pixels
    if (threadIdx.y == 0 && blockIdx.y != 0)
    {

#pragma unroll
      for (int j = 1; j <= 2; j++)
      {
#pragma unroll
        for (int k = 0; k <= 2; k++)
        {
          redTiles_ds  [tRow-j][tCol+k] = pixels[i-(j*w)+k].r;
          greenTiles_ds[tRow-j][tCol+k] = pixels[i-(j*w)+k].g;
          blueTiles_ds [tRow-j][tCol+k] = pixels[i-(j*w)+k].b;
        }
      }

    }
    
    // get two bottom pixels and four bottom right pixels.
    else if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1)
    {
#pragma unroll
      for (int j = 1; j <= 2; j++)
      {
#pragma unroll
        for (int k = 0; k <= 2; k++)
        {
          redTiles_ds  [tRow+j][tCol+k] = pixels[i+(j*w)+k].r;
          greenTiles_ds[tRow+j][tCol+k] = pixels[i+(j*w)+k].g;
          blueTiles_ds [tRow+j][tCol+k] = pixels[i+(j*w)+k].b;
        }
      }

    }
  }
  
  // get two top pixels
  else if (threadIdx.y == 0 && blockIdx.y != 0)
  {

#pragma unroll
    for (int j = 1; j <= 2; j++)
    {
      redTiles_ds  [tRow-j][tCol] = pixels[i-(j*w)].r;
      greenTiles_ds[tRow-j][tCol] = pixels[i-(j*w)].g;
      blueTiles_ds [tRow-j][tCol] = pixels[i-(j*w)].b;
    }
  }
  
  // get two bottom pixels
  else if (threadIdx.y == blockDim.y - 1 && blockIdx.y != gridDim.y - 1)
  {
#pragma unroll
    for (int j = 1; j <= 2; j++)
    {
      redTiles_ds  [tRow+j][tCol] = pixels[i+(j*w)].r;
      greenTiles_ds[tRow+j][tCol] = pixels[i+(j*w)].g;
      blueTiles_ds [tRow+j][tCol] = pixels[i+(j*w)].b;
    }
  }


  //sync
  __syncthreads();

  
  if (col < w - 2 && row < h - 2 &&
    col >= 2 && row >= 2)
  {
    unsigned rRes = 0;
    unsigned gRes = 0;
    unsigned bRes = 0;


    // cause a 5x5 Gaussian kernel is fixed size and calling a 2D int array for gaussian blur
    // kernel might be slow with global memory accesses.
    // 
    // remember that if the col and row are proper, then tCol and tRow which already have the offsets are at i.
    int rowZero = tRow - 2;   
    int colZero = tCol - 2; 

    int rowOne   = rowZero + 1;
    int rowTwo   = rowZero + 2;
    int rowThree = rowZero + 3;
    int rowFour  = rowZero + 4;

    int colOne   = colZero + 1;
    int colTwo   = colZero + 2;
    int colThree = colZero + 3;
    int colFour  = colZero + 4;


    rRes += redTiles_ds  [rowZero][colZero];
    gRes += greenTiles_ds[rowZero][colZero];
    bRes += blueTiles_ds [rowZero][colZero];
                                 
    rRes += redTiles_ds  [rowZero][colOne]   * 4;
    gRes += greenTiles_ds[rowZero][colOne]   * 4;
    bRes += blueTiles_ds [rowZero][colOne]   * 4;
                                 
    rRes += redTiles_ds  [rowZero][colTwo]   * 7;
    gRes += greenTiles_ds[rowZero][colTwo]   * 7;
    bRes += blueTiles_ds [rowZero][colTwo]   * 7;
                                 
    rRes += redTiles_ds  [rowZero][colThree] * 4;
    gRes += greenTiles_ds[rowZero][colThree] * 4;
    bRes += blueTiles_ds [rowZero][colThree] * 4;
                                 
    rRes += redTiles_ds  [rowZero][colFour];
    gRes += greenTiles_ds[rowZero][colFour];
    bRes += blueTiles_ds [rowZero][colFour];



    rRes += redTiles_ds  [rowOne][colZero]  * 4;
    gRes += greenTiles_ds[rowOne][colZero]  * 4;
    bRes += blueTiles_ds [rowOne][colZero]  * 4;
                          
    rRes += redTiles_ds  [rowOne][colOne]   * 16;
    gRes += greenTiles_ds[rowOne][colOne]   * 16;
    bRes += blueTiles_ds [rowOne][colOne]   * 16;
                          
    rRes += redTiles_ds  [rowOne][colTwo]   * 26;
    gRes += greenTiles_ds[rowOne][colTwo]   * 26;
    bRes += blueTiles_ds [rowOne][colTwo]   * 26;
                          
    rRes += redTiles_ds  [rowOne][colThree] * 16;
    gRes += greenTiles_ds[rowOne][colThree] * 16;
    bRes += blueTiles_ds [rowOne][colThree] * 16;
                             
    rRes += redTiles_ds  [rowOne][colFour]  * 4;
    gRes += greenTiles_ds[rowOne][colFour]  * 4;
    bRes += blueTiles_ds [rowOne][colFour]  * 4;



    rRes += redTiles_ds  [rowTwo][colZero]  * 7;
    gRes += greenTiles_ds[rowTwo][colZero]  * 7;
    bRes += blueTiles_ds [rowTwo][colZero]  * 7;
                           
    rRes += redTiles_ds  [rowTwo][colOne]   * 26;
    gRes += greenTiles_ds[rowTwo][colOne]   * 26;
    bRes += blueTiles_ds [rowTwo][colOne]   * 26;
                           
    rRes += redTiles_ds  [rowTwo][colTwo]   * 41;
    gRes += greenTiles_ds[rowTwo][colTwo]   * 41;
    bRes += blueTiles_ds [rowTwo][colTwo]   * 41;
                           
    rRes += redTiles_ds  [rowTwo][colThree] * 26;
    gRes += greenTiles_ds[rowTwo][colThree] * 26;
    bRes += blueTiles_ds [rowTwo][colThree] * 26;
                           
    rRes += redTiles_ds  [rowTwo][colFour]  * 7;
    gRes += greenTiles_ds[rowTwo][colFour]  * 7;
    bRes += blueTiles_ds [rowTwo][colFour]  * 7;



    rRes += redTiles_ds  [rowThree][colZero]  * 4;
    gRes += greenTiles_ds[rowThree][colZero]  * 4;
    bRes += blueTiles_ds [rowThree][colZero]  * 4;
                          
    rRes += redTiles_ds  [rowThree][colOne]   * 16;
    gRes += greenTiles_ds[rowThree][colOne]   * 16;
    bRes += blueTiles_ds [rowThree][colOne]   * 16;
                          
    rRes += redTiles_ds  [rowThree][colTwo]   * 26;
    gRes += greenTiles_ds[rowThree][colTwo]   * 26;
    bRes += blueTiles_ds [rowThree][colTwo]   * 26;
                          
    rRes += redTiles_ds  [rowThree][colThree] * 16;
    gRes += greenTiles_ds[rowThree][colThree] * 16;
    bRes += blueTiles_ds [rowThree][colThree] * 16;
                          
    rRes += redTiles_ds  [rowThree][colFour]  * 4;
    gRes += greenTiles_ds[rowThree][colFour]  * 4;
    bRes += blueTiles_ds [rowThree][colFour]  * 4;


    rRes += redTiles_ds  [rowFour][colZero];
    gRes += greenTiles_ds[rowFour][colZero];
    bRes += blueTiles_ds [rowFour][colZero];
                          
    rRes += redTiles_ds  [rowFour][colOne]   * 4;
    gRes += greenTiles_ds[rowFour][colOne]   * 4;
    bRes += blueTiles_ds [rowFour][colOne]   * 4;
                          
    rRes += redTiles_ds  [rowFour][colTwo]   * 7;
    gRes += greenTiles_ds[rowFour][colTwo]   * 7;
    bRes += blueTiles_ds [rowFour][colTwo]   * 7;
                          
    rRes += redTiles_ds  [rowFour][colThree] * 4;
    gRes += greenTiles_ds[rowFour][colThree] * 4;
    bRes += blueTiles_ds [rowFour][colThree] * 4;
                          
    rRes += redTiles_ds  [rowFour][colFour];
    gRes += greenTiles_ds[rowFour][colFour];
    bRes += blueTiles_ds [rowFour][colFour];

    out[i].r = rRes / 273;
    out[i].g = gRes / 273;
    out[i].b = bRes / 273;
  }
}

cudaError_t deviceBlur(ImgData& out, CudaType cuda, int passes)
{
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
    int blocksX = out.w / threads;
    int blocksY = out.h / threads;
    dim3 Block(threads, threads, 1);
    dim3 Grid(blocksX, blocksY, 1);

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
      if (cuda == CudaType::TILED)
      {
        tiledImage5x5BlurKernel << <Grid, Block >> > (pixels_d, out_d, w_d, h_d);
      }
      else
      {
        image5x5BlurKernel << <Grid, Block >> > (pixels_d, out_d, w_d, h_d);
      }

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching imageBlur!\n", cudaStatus);
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