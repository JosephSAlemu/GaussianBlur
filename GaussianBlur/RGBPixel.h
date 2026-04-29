#pragma once

struct RGBPixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	RGBPixel() : r(0), g(0), b(0) {}
	RGBPixel(unsigned char red, unsigned char green, unsigned char blue) : r(red), g(green), b(blue) {}
};