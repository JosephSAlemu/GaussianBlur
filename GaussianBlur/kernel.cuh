#include "cuda_runtime.h"
#include <stdio.h>
#include "common.h"

cudaError_t deviceBlur(GaussianKernel& gaussian, ImgData& data, int passes = 1);