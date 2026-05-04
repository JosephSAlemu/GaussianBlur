#pragma once

#define KERNEL_SIZE_5X5 5
#define KERNEL_DENOMINATOR_5X5 273
#include "lodepng.h"
#include "Framework.h"

struct RGBPixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	RGBPixel() noexcept(true) = default;
	RGBPixel(const RGBPixel& rhs) noexcept(true)
		:r(rhs.r),
		 g(rhs.g),
		 b(rhs.b)
	{};
	RGBPixel& operator=(const RGBPixel& rhs) noexcept(true)
	{
		if (this != &rhs)
		{
			this->r = rhs.r;
			this->g = rhs.g;
			this->b = rhs.b;
		}
		return *this;
	}
	bool operator==(const RGBPixel& rhs)
	{
		return (this->r == rhs.r && 
						this->g == rhs.g &&
						this->b == rhs.b);
	}
	~RGBPixel() noexcept(true) = default;
};

struct ImgData
{
	RGBPixel* pixels;
	unsigned w;
	unsigned h;
	size_t len;
	lodepng::State* state;
	ImgData() noexcept(true) = default;
	ImgData(const ImgData& rhs)
		: w(rhs.w),
			h(rhs.w),
		  len(rhs.len)
	{
		this->state = new lodepng::State(*(rhs.state));
		this->pixels = new RGBPixel[rhs.len]();
		for (size_t i = 0; i < rhs.len; i++)
		{
			this->pixels[i] = rhs.pixels[i];
		}
	}
	ImgData& operator= (const ImgData& rhs)
	{
		if (this != &rhs)
		{
			delete this->state;
			delete[] this->pixels;

			this->state = new lodepng::State(*(rhs.state));
			this->pixels = new RGBPixel[rhs.len];
			for (size_t i = 0; i < len; i++)
			{
				this->pixels[i] = rhs.pixels[i];
			}
		}
		return *this;
	}
	~ImgData()
	{
		delete this->state;
		delete[] this->pixels;
	}
};

struct GaussianKernel5x5
{
	GaussianKernel5x5() noexcept(true)
	{
		this->kernel = new int* [KERNEL_SIZE_5X5];
		this->kernel[0] = new int[KERNEL_SIZE_5X5] {1, 4, 7, 4, 1};
		this->kernel[1] = new int[KERNEL_SIZE_5X5] {4, 16, 26, 16, 4};
		this->kernel[2] = new int[KERNEL_SIZE_5X5] {7, 26, 41, 26, 7};
		this->kernel[3] = new int[KERNEL_SIZE_5X5] {4, 16, 26, 16, 4};
		this->kernel[4] = new int[KERNEL_SIZE_5X5] {1, 4, 7, 4, 1};

		this->halo = 2;
		this->divisor = KERNEL_DENOMINATOR_5X5;
	}

	~GaussianKernel5x5() noexcept(true)
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

	int** kernel;
	int halo;
	int divisor;
};


enum class ImageError : uint8_t
{
	SUCCESS,
	FAILURE
};
