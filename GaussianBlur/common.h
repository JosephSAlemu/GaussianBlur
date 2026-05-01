#pragma once

#define KERNEL_SIZE_5X5 5
#define KERNEL_DENOMINATOR_5X5 273
#include "lodepng.h"


struct RGBPixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	RGBPixel() = default;
};

struct ImgData
{
	RGBPixel* pixels;
	unsigned w;
	unsigned h;
	size_t len;
	lodepng::State state;
	ImgData() = default;
	~ImgData()
	{
		delete[] pixels;
	}
};

struct GaussianKernel
{
	int** kernel;
	int halo;
	int divisor;

	void set5x5ImageKernel()
	{
		this->kernel = new int* [KERNEL_SIZE_5X5];
		this->kernel[0] = new int[KERNEL_SIZE_5X5] {1, 4, 7, 4, 1};
		this->kernel[1] = new int[KERNEL_SIZE_5X5] {4, 16, 25, 16, 4};
		this->kernel[2] = new int[KERNEL_SIZE_5X5] {7, 25, 41, 25, 7};
		this->kernel[3] = new int[KERNEL_SIZE_5X5] {4, 16, 25, 16, 4};
		this->kernel[4] = new int[KERNEL_SIZE_5X5] {1, 4, 7, 4, 1};

		this->halo = KERNEL_SIZE_5X5 / 2;
		this->divisor = KERNEL_DENOMINATOR_5X5;
	}
	GaussianKernel() = default;
	~GaussianKernel()
	{
		if (this->halo == 2 && kernel != nullptr)
		{
			for (int i = 0; i < KERNEL_SIZE_5X5; i++)
			{
				delete[] this->kernel[i];
			}
			delete[] this->kernel;
		}
	}
};


enum class ImageError : uint8_t
{
	SUCCESS,
	FAILURE
};


