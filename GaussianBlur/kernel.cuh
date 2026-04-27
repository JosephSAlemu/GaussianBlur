#include "CudaFramework.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t deviceBlur(char* img_buffer, unsigned h, unsigned w);