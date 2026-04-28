
#include "cuda_runtime.h"
#include <stdio.h>

cudaError_t deviceBlur(unsigned char* img_buffer, unsigned h, unsigned w);