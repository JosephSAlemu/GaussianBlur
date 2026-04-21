//----------------------------------------------------------------------------
// Copyright 2026, Ed Keenan, all rights reserved.
//----------------------------------------------------------------------------

#ifndef _CudaFramework_h
#define _CudaFramework_h

#ifdef __INTELLISENSE__
	// This flags cuda_runtime.h to expose everything to Intellisense
	//#define __CUDACC__
	// Used in place of the cuda brackets to fix Visual Studio C++ Intellisense
	// Will be replaced with <<< ... >>> when built
	// Usage:
	//	function CudaKernel(blocksPerGrid, threadsPerBlock) (a, b, c, d);
#define CudaKernel(...)
// Allows CUDA style array initialization without a Visual Studio warning
// Usage:
//	__shared__ uint8_t array[128] InitZero;
#define InitZero = {0}
#else
#define CudaKernel(...) <<< __VA_ARGS__ >>>
#define InitZero
#endif

// The main CUDA runtime header must be included after defining __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Cuda error handling
#include <stdio.h>
#include <stdlib.h>
//I had to add this manually to fix my errors.
#include "Framework.h"

namespace Cuda
{

	namespace Trace
	{
		static void out(const char *const fmt, ...) noexcept
		{
			va_list args;

			static std::mutex mut;
			std::lock_guard<std::mutex> _guard(mut);

#pragma warning( push )
#pragma warning( disable : 26492 )
#pragma warning( disable : 26481 )
			va_start(args, fmt);
#pragma warning( pop )
			static char buffer[512];
			vsprintf_s(&buffer[0], 512, fmt, args);
			OutputDebugStringA(&buffer[0]);

			// va_end(args); - original.. below to new code
			args = static_cast<va_list> (nullptr);
		}
	}

}

#pragma warning( push )
#pragma warning( disable : 4505 )
static void CudaAssert(cudaError_t err, const char *file, int line) noexcept
{
	if(err != cudaSuccess)
	{
		Trace::out("%s in \n%s(%d) : <double-click>\n\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}
#pragma warning( pop )

// If the cuda function fails, prints the error and exits the process
#define CudaTry( err ) (CudaAssert( err, __FILE__, __LINE__ ))

#include <cstdarg>
#include <mutex>

#ifndef UNICODE
#define UNICODE
#endif
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>


namespace Gpu
{
	template <typename ...Params>
	__device__ static void out(const char *const fmt, Params&& ...args) noexcept
	{
		printf(fmt, std::forward<Params>(args)...);
	}
}


void printDevProp(cudaDeviceProp devProp)
{
	Cuda::Trace::out("%s\n", devProp.name);
	Cuda::Trace::out("Major revision number:         %d\n", devProp.major);
	Cuda::Trace::out("Minor revision number:         %d\n", devProp.minor);
	Cuda::Trace::out("Total global memory:           %u bytes\n", devProp.totalGlobalMem);
	Cuda::Trace::out("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	Cuda::Trace::out("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	Cuda::Trace::out("Total registers per block:     %d\n", devProp.regsPerBlock);
	Cuda::Trace::out("Warp size:                     %d\n", devProp.warpSize);
	Cuda::Trace::out("Maximum memory pitch:          %u\n", devProp.memPitch);
	Cuda::Trace::out("Total constant memory:         %u\n", devProp.totalConstMem);
	return;
}


int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch(devProp.major)
	{
		case 2: // Fermi
			if(devProp.minor == 1) cores = mp * 48;
			else cores = mp * 32;
			break;
		case 3: // Kepler
			cores = mp * 192;
			break;
		case 5: // Maxwell
			cores = mp * 128;
			break;
		case 6: // Pascal
			if((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
			else if(devProp.minor == 0) cores = mp * 64;
			else Cuda::Trace::out("Unknown device type\n");
			break;
		case 7: // Volta and Turing
			if((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
			else Cuda::Trace::out("Unknown device type\n");
			break;
		case 8: // Ampere
			if(devProp.minor == 0) cores = mp * 64;
			else if(devProp.minor == 6) cores = mp * 128;
			else if(devProp.minor == 9) cores = mp * 128; // ada lovelace
			else Cuda::Trace::out("Unknown device type\n");
			break;
		case 9: // Hopper
			if(devProp.minor == 0) cores = mp * 128;
			else Cuda::Trace::out("Unknown device type\n");
			break;
		case 10: // Blackwell
			if(devProp.minor == 0) cores = mp * 128;
			else Cuda::Trace::out("Unknown device type\n");
			break;
		case 12: // Blackwell
			if(devProp.minor == 0) cores = mp * 128;
			else Cuda::Trace::out("Unknown device type\n");
			break;
		default:
			Cuda::Trace::out("Unknown device type\n");
			break;
	}
	return cores;
}



class Headers
{
public:
	Headers()
	{
		Cuda::Trace::out("\n");
		Cuda::Trace::out("----------------------------------\n");
		Cuda::Trace::out("\n");
	}
	~Headers()
	{
		Cuda::Trace::out("\n");
		Cuda::Trace::out("----------------------------------\n");
		Cuda::Trace::out("\n");
	}
};


#endif

// --- End of File ----
