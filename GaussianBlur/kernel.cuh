#pragma once
#include "cuda_runtime.h"
#include "common.h"

cudaError_t deviceBlur(ImgData& out, int passes = 1);
