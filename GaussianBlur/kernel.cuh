#pragma once
#include <cuda_runtime.h>
#include "common.h"

cudaError_t deviceBlur(ImgData& out, CudaType cuda, int passes = 1);
