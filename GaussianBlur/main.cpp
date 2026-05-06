//----------------------------------------------------------------------------
// Copyright 2026, Ed Keenan, all rights reserved.
//----------------------------------------------------------------------------
#include "kernel.cuh"
/*
Display general info about the PNG.
*/
void displayPNGInfo(const LodePNGInfo &info)
{
	const LodePNGColorMode &color = info.color;

	Trace::out("\n");
	Trace::out("Compression method: %d \n", info.compression_method);
	Trace::out("     Filter method: %d \n", info.filter_method);
	Trace::out("  Interlace method: %d \n", info.interlace_method);
	Trace::out("        Color type: %d (LCT_RGB = 2) \n", color.colortype);
	Trace::out("         Bit depth: %d \n", color.bitdepth);
	Trace::out("    Bits per pixel: %d \n", lodepng_get_bpp(&color));
	Trace::out("Channels per pixel: %d \n", lodepng_get_channels(&color));
	Trace::out(" Is greyscale type: %d \n", lodepng_is_greyscale_type(&color));
	Trace::out("    Can have alpha: %d \n", lodepng_can_have_alpha(&color));
	Trace::out("      Palette size: %d \n", color.palettesize);
	Trace::out("     Has color key: %d \n", color.key_defined);

	if(color.key_defined)
	{
		Trace::out("Color key r: %d \n", color.key_r);
		Trace::out("Color key g: %d \n", color.key_g);
		Trace::out("Color key b: %d \n", color.key_b);
	}
	Trace::out("Texts: %d \n", info.text_num);
	for(size_t i = 0; i < info.text_num; i++)
	{
		Trace::out("Text: %d : %s \n", info.text_keys[i], info.text_strings[i]);
	}
	Trace::out("International texts: %d \n", info.itext_num);
	for(size_t i = 0; i < info.itext_num; i++)
	{
		std::cout << "Text: "
			<< info.itext_keys[i] << ", "
			<< info.itext_langtags[i] << ", "
			<< info.itext_transkeys[i] << ": "
			<< info.itext_strings[i] << std::endl << std::endl;
	}
	Trace::out("Time defined: %d \n", info.time_defined);
	if(info.time_defined)
	{
		const LodePNGTime &time = info.time;
		Trace::out("  year: %d\n", time.year);
		Trace::out(" month: %d\n", time.month);
		Trace::out("   day: %d\n", time.day);
		Trace::out("  hour: %d\n", time.hour);
		Trace::out("minute: %d\n", time.minute);
		Trace::out("second: %d\n", time.second);
	}
	Trace::out("Physical pixel dimensions and aspect ratio defined: %d\n", info.phys_defined);
	if(info.phys_defined)
	{
		Trace::out("physics X: %d \n", info.phys_x);
		Trace::out("physics Y: %d \n", info.phys_y);
		Trace::out("physics unit: %d (1 = meter) \n", info.phys_unit);
	}
}

/*
Display the names and sizes of all chunks in the PNG file.
*/
void displayChunkNames(const std::vector<unsigned char> &buffer)
{
	// Listing chunks is based on the original file, not the decoded png info.
	const unsigned char *chunk, *end;
	end = &buffer.back() + 1;
	chunk = &buffer.front() + 8;

	std::cout << std::endl << "Chunks:" << std::endl;
	std::cout << " type: length(s)";
	std::string last_type;
	while(chunk < end && end - chunk >= 8)
	{
		char type[5];
		lodepng_chunk_type(type, chunk);
		if(std::string(type).size() != 4)
		{
			std::cout << "this is probably not a PNG" << std::endl;
			return;
		}

		if(last_type != type)
		{
			std::cout << std::endl;
			std::cout << " " << type << ": ";
		}
		last_type = type;

		std::cout << lodepng_chunk_length(chunk) << ", ";

		chunk = lodepng_chunk_next_const(chunk, end);
	}
	std::cout << std::endl;
}

/*
Show the filtertypes of each scanline in this PNG image.
*/
void displayFilterTypes(const std::vector<unsigned char> &buffer, bool ignore_checksums)
{
	//Get color type and interlace type
	lodepng::State state;
	if(ignore_checksums)
	{
		state.decoder.ignore_crc = 1;
		state.decoder.zlibsettings.ignore_adler32 = 1;
	}
	unsigned w, h;
	unsigned error;
	error = lodepng_inspect(&w, &h, &state, &buffer[0], buffer.size());

	if(error)
	{
		std::cout << "inspect error " << error << ": " << lodepng_error_text(error) << std::endl;
		return;
	}

	if(state.info_png.interlace_method == 1)
	{
		std::cout << "showing filtertypes for interlaced PNG not supported by this example" << std::endl;
		return;
	}

	//Read literal data from all IDAT chunks
	const unsigned char *chunk, *begin, *end;
	end = &buffer.back() + 1;
	begin = chunk = &buffer.front() + 8;

	std::vector<unsigned char> zdata;

	while(chunk < end && end - chunk >= 8)
	{
		char type[5];
		lodepng_chunk_type(type, chunk);
		if(std::string(type).size() != 4)
		{
			std::cout << "this is probably not a PNG" << std::endl;
			return;
		}

		if(std::string(type) == "IDAT")
		{
			const unsigned char *cdata = lodepng_chunk_data_const(chunk);
			unsigned clength = lodepng_chunk_length(chunk);
			if(chunk + clength + 12 > end || clength > buffer.size() || chunk + clength + 12 < begin)
			{
				std::cout << "invalid chunk length" << std::endl;
				return;
			}

			for(unsigned i = 0; i < clength; i++)
			{
				zdata.push_back(cdata[i]);
			}
		}

		chunk = lodepng_chunk_next_const(chunk, end);
	}

	//Decompress all IDAT data
	std::vector<unsigned char> data;
	error = lodepng::decompress(data, &zdata[0], zdata.size());

	if(error)
	{
		std::cout << "decompress error " << error << ": " << lodepng_error_text(error) << std::endl;
		return;
	}

	//A line is 1 filter byte + all pixels
	size_t linebytes = 1 + lodepng_get_raw_size(w, 1, &state.info_png.color);

	if(linebytes == 0)
	{
		std::cout << "error: linebytes is 0" << std::endl;
		return;
	}

	std::cout << "Filter types: ";
	for(size_t i = 0; i < data.size(); i += linebytes)
	{
		std::cout << (int)(data[i]) << " ";
	}
	std::cout << std::endl;

}

ImageError _NODISCARD decodeImage(const char* src_path, ImgData& data) noexcept(true)
{
	std::vector<unsigned char> image;
	std::vector<unsigned char> buffer;
	unsigned error;

	data.state = new lodepng::State();

	data.state->decoder.color_convert = 0;
	data.state->decoder.remember_unknown_chunks = 1; //make it reproduce even unknown chunks in the saved image

	lodepng::load_file(buffer, src_path);
	
	// Reminder so I don't fucking forget but image buffer size is w*h*3 not w*h. Tripped me off until I realized it stores all 3 channels.
	error = lodepng::decode(image, data.w, data.h, *(data.state), buffer);
	if (error)
	{
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		return ImageError::FAILURE;
	}
	if (data.w * data.h * 3 != image.size())
	{
		Trace::out("Image has less or more than 3 channels. Please check the image");
		return ImageError::FAILURE;
	}
	// Convert from a single buffer to RGB. Remember image goes from RGB
	data.len = image.size() / 3;
	data.pixels = new RGBPixel[data.len];
	for (size_t i = 0; i < image.size(); i+=3) {
		size_t p = i / 3;
		data.pixels[p].r = image[i];
		data.pixels[p].g = image[i + 1];
		data.pixels[p].b = image[i+2];
	}

	bool ignore_checksums = false;

	Trace::out("  Filesize: %d (%d K)\n", buffer.size(), buffer.size() / 1024);
	Trace::out("     Width: %d \n", data.w);
	Trace::out("    Height: %d \n", data.h);
	Trace::out("Num pixels: %d \n", data.w * data.h);


	displayPNGInfo(data.state->info_png);
	std::cout << std::endl;
	displayChunkNames(buffer);
	std::cout << std::endl;
	displayFilterTypes(buffer, ignore_checksums);
	std::cout << std::endl;
	
	return ImageError::SUCCESS;
}

ImageError _NODISCARD encodeImage(const char* dest_path, ImgData& data) noexcept(true)
{
	// Convert from RGB to a single buffer again
	size_t len = data.len * 3;

	std::vector<unsigned char> buffer;
	std::vector<unsigned char> image(len);

	for (size_t i = 0; i < len; i+=3)
	{
		size_t p = i / 3;
		image[i]		 = data.pixels[p].r;
		image[i + 1] = data.pixels[p].g;
		image[i + 2] = data.pixels[p].b;
	}
	data.state->encoder.text_compression = 1;
	
	unsigned error = lodepng::encode(buffer, image, data.w, data.h, *(data.state));
	if (error)
	{
		std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		return ImageError::FAILURE;
	}

	lodepng::save_file(buffer, dest_path);
}


void hostBlur(ImgData& out, GaussianKernel5x5& gaussian, int passes = 1)
{
	// Create an Apron (Check that you are start at pixel row > 2 and row < rowLength - 2
	// Also check height > 2 and height < heightLength - 2
	// data stored R,G,B - one byte each pixel
	ImgData buffer(out);
	size_t len = out.len * sizeof(RGBPixel);
	for (int i = 0; i < passes; i++)
	{
		memcpy_s(buffer.pixels, len, out.pixels, len);
		int rhalo = gaussian.halo;
		int chalo = out.w * gaussian.halo;

		int leftCols = rhalo;
		int rightCols = out.w - leftCols;

		int topRows = chalo;
		int bottomRows = (out.w * out.h) - topRows;

		for (int i = 0; i < out.len; i++)
		{
			int col = i % out.w;
			int row = i;
			if (row >= topRows && row < bottomRows &&
				col >= leftCols && col < rightCols)
			{
				unsigned rRes = 0;
				unsigned gRes = 0;
				unsigned bRes = 0;

				int width = out.w;
				for (int c = -chalo; c <= chalo; c += out.w)
				{
					int kRow = c / width + rhalo;
					for (int r = -rhalo; r <= rhalo; r++)
					{
						int kCol = rhalo + r;
						int index = i + c + r;
						unsigned val = gaussian.kernel[kRow][kCol];

						rRes += buffer.pixels[index].r * val;
						gRes += buffer.pixels[index].g * val;
						bRes += buffer.pixels[index].b * val;
					}
				}
				out.pixels[i].r = rRes / gaussian.divisor;
				out.pixels[i].g = gRes / gaussian.divisor;
				out.pixels[i].b = bRes / gaussian.divisor;
			}
		}
	}
}

bool isSameImage(ImgData& lhs, ImgData& rhs)
{
	if (lhs.len != rhs.len) { return false; }
	for (int i = 0; i < rhs.len; i++)
	{
		if ( !(lhs.pixels[i] == rhs.pixels[i]) )
		{
			return false;
		}
	}
	return true;
}

void benchmark(ImgData& out_h, ImgData& out_d, ImgData& out_dt, GaussianKernel5x5& gaussian, int passes = 1)
{
	PerformanceTimer host;
	PerformanceTimer device;
	PerformanceTimer devicet;


	host.Tic();
	hostBlur(out_h, gaussian, passes);
	host.Toc();
	Trace::out("hostTime: %f\n", host.TimeInSeconds());

	device.Tic();
	deviceBlur(out_d, CudaType::NOTTILED, passes);
	device.Toc();
	Trace::out("No Tiles deviceTime: %f\n", device.TimeInSeconds());

	devicet.Tic();
	deviceBlur(out_dt, CudaType::TILED, passes);
	devicet.Toc();
	Trace::out("Tiled deviceTime: %f\n", devicet.TimeInSeconds());
}

int main()
{
	const char* src_path = "scarecrow.png";
	int passes = 1;

	ImgData data;
	GaussianKernel5x5 gaussian;

	ImageError error;
	error = decodeImage(src_path, data);
	if (error == ImageError::FAILURE)
	{
		return 1;
	}
	ImgData out_h(data);
	ImgData out_d(data);
	ImgData out_dt(data);

#ifdef BENCHMARK
	benchmark(out_h, out_d, out_dt, gaussian, passes);

#else

	deviceBlur(out_dt, CudaType::TILED, passes);
	deviceBlur(out_d, CudaType::NOTTILED, passes);
	hostBlur(out_h, gaussian, passes);

	if (!isSameImage(out_dt, out_d) || !isSameImage(out_h, out_d))
	{
		Trace::out("ERROR IN BLURRING CODE!");
		return 1;
	}

	Trace::out("Blur x%d times", passes);

	encodeImage("test_host.png", out_h);
	encodeImage("test_device.png", out_d);
	encodeImage("test_device_tiled.png", out_dt);


#endif


}

// ---  End of File ---