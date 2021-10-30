//
//  kernel.cu
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include "ppm.h"

using namespace std;

/*********** Gray Scale Filter  *********/

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__ // function executed on device, only callable from the host
void cuda_grayscale(int width, int height, BYTE *image, BYTE *image_out)
{
	// Implement grayscale filter kernel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int offset_out = row * width;      // 1 color per pixel
	int offset     = offset_out * 3; // 3 colors per pixel

	if(row < height && col < width) {
		BYTE *pixel = &image[offset + col * 3];
		image_out[offset_out + col] =
			pixel[0] * 0.2126f + // R
			pixel[1] * 0.7152f + // G
			pixel[2] * 0.0722f;  // B
	}
}


// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)

// Define the cGaussian array on the constant memory
__constant__ float cGaussian[64]; // saved in constant memory, within scope of grid and its lifetime within application

// __host__ function by default
void cuda_updateGaussian(int r, double sd)
{
    float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	// Copy computed fGaussian to the cGaussian on device memory
	cudaMemcpyToSymbol(cGaussian, fGaussian, (sizeof(float) * 64), 0, cudaMemcpyHostToDevice);
}

// Implement cuda_gaussian() kernel
__device__ // called by the cuda_bilateral_filter Kernel in device
inline double cuda_gaussian(float x, double sigma) {
    return expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

/*********** Bilateral Filter  *********/
// Parallel (GPU) Bilateral filter kernel
__global__ // function executed on device, only callable from the host
void cuda_bilateral_filter(BYTE* input, BYTE* output,
	int width, int height,
	int r, double sI, double sS)
{
    // Implement bilateral filter kernel
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < height && col < width) { // check condition
        double iFiltered = 0;
        double wP = 0;
        // Get the centre pixel value
        unsigned char centrePx = input[row * width + col];
        // Iterate through filter size from centre pixel
        for (int dy = -r; dy <= r; ++dy) {
            int neighborY = row + dy;
            if (neighborY < 0) neighborY = 0;
            else if (neighborY >= height) neighborY = height - 1;
            for (int dx = -r; dx <= r; ++dx) {
                int neighborX = col + dx;
                if (neighborX < 0) neighborX = 0;
                else if (neighborX >= width) neighborX = width - 1;
                // Get the current pixel; value
                unsigned char currPx = input[neighborY * width + neighborX];
                // Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
                double w = (cGaussian[dy + r] * cGaussian[dx + r]) * cuda_gaussian(centrePx - currPx, sI); // cGaussian stored in constant memory
                iFiltered += w * currPx;
                wP += w;
            }
        }
        output[row * width + col] = iFiltered / wP;
    }
}

// __host__ function by default
void gpu_pipeline(const Image & input, Image & output, int r, double sI, double sS)
{
	// Events to calculate gpu run time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GPU related variables
	BYTE *d_input = NULL;
	BYTE *d_image_out[2] = {0}; //temporary output buffers on gpu device
	int image_size = input.cols*input.rows;
	int suggested_blockSize;   // The launch configurator returned block size
	int suggested_minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch

	// ******* Grayscale kernel launch *************

	// Create block size for grayscaling kernel

	// syntax: Returns grid and block size that achieves maximum potential occupancy for a device function
	/*
	__host__ cudaError_t cudaOccupancyMaxPotentialBlockSize
	( int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int  blockSizeLimit = 0 )
	*/

	cudaOccupancyMaxPotentialBlockSize(&suggested_minGridSize, &suggested_blockSize, cuda_grayscale, 0, 0);

	int block_dim_x, block_dim_y;
	block_dim_x = (int) sqrt(suggested_blockSize); // suggested_blockSize: quadratic size of the block
	block_dim_y = block_dim_x;

	dim3 gray_block(block_dim_x, block_dim_y); // block dimensions determined

	 // Round up grids in X- and Y-directions according to rows and columns size of input(image)
	 // block_dim_x and block_dim_y adjusted with suggested_blockSize
	 int gridSizeX =ceil((input.cols + block_dim_x - 1) / block_dim_x );
	 int gridSizeY = ceil((input.rows + block_dim_y - 1) / block_dim_y);

    // Calculate grid size to cover the whole image
    dim3 gray_grid(gridSizeX, gridSizeY); // grid dimensions determined

	// Allocate the intermediate image buffers for each step
	Image img_out(input.cols, input.rows, 1, "P5");

	// Image size in bytes to set
	size_t image_byte_size = sizeof(BYTE) * image_size;

	for (int i = 0; i < 2; i++)
	{
		// Allocate memory on the device
		cudaMalloc((void**) &d_image_out[i], image_byte_size); // 8bpp image
		// Intialize allocated memory on device to zero
		cudaMemset((void*) &d_image_out[i], 0, image_byte_size);
	}

	// Copy input image to device
	// Allocate memory on device for input image
	cudaMalloc((void**) &d_input, image_byte_size * 3); //image_byte_size * 3 because of 24bpp image
	// Copy input image into the device memory
	cudaMemcpy(d_input, input.pixels, image_byte_size * 3, cudaMemcpyHostToDevice);


	cudaEventRecord(start, 0); // start timer
	// Convert input image to grayscale
	// Launch cuda_grayscale()
	// cuda_grayscale() Kernel
	cuda_grayscale <<< gray_grid, gray_block >>> (input.cols, input.rows, d_input, d_image_out[0]);

	cudaEventRecord(stop, 0); // stop timer
	cudaEventSynchronize(stop);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	cout << "GPU Grayscaling time: " << time << " (ms)\n";
	cout << "Launched blocks of size " << gray_block.x * gray_block.y << endl;

	// Transfer image from device to the main memory for saving onto the disk
	cudaMemcpy(img_out.pixels, d_image_out[0], image_byte_size, cudaMemcpyDeviceToHost);
	savePPM(img_out, "image_gpu_gray.ppm");

	// ******* Bilateral filter kernel launch *************

	// Creating the block size for grayscaling kernel

	cudaOccupancyMaxPotentialBlockSize(&suggested_minGridSize, &suggested_blockSize, cuda_bilateral_filter, 0, 0);

	block_dim_x = (int) sqrt(suggested_blockSize); // suggested_blockSize: quadratic size of the block
	block_dim_y = block_dim_x;

	dim3 bilateral_block(block_dim_x, block_dim_y); // block dimensions determined

	// Calculate grid size to cover the whole image

    // Round up grids in X- and Y-directions according to rows and columns size of input(image)
    // block_dim_x and block_dim_y adjusted with suggested_blockSize
	int gridSizeX_bilateral =ceil((input.cols + block_dim_x - 1) / block_dim_x );
	int gridSizeY_bilateral = ceil((input.rows + block_dim_y - 1) / block_dim_y);

	dim3 bilateral_grid(gridSizeX_bilateral, gridSizeY_bilateral); // grid dimensions determined


	// Create gaussain 1d array
	cuda_updateGaussian(r,sS);

	cudaEventRecord(start, 0); // start timer

	// Launch cuda_bilateral_filter()
	// Call bilateral filter for the grayscale image (d_image_out[0])
	cuda_bilateral_filter <<< bilateral_grid, bilateral_block >>> (d_image_out[0], d_image_out[1], input.cols, input.rows, r, sI, sS);

	cudaEventRecord(stop, 0); // stop timer
	cudaEventSynchronize(stop);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	cout << "GPU Bilateral Filter time: " << time << " (ms)\n";
	cout << "Launched blocks of size " << bilateral_block.x * bilateral_block.y << endl;

	// Copy output from device to host
	// Transfer image from device to the main memory for saving onto the disk
	cudaMemcpy(output.pixels, d_image_out[1], image_byte_size, cudaMemcpyDeviceToHost);
	//

	// ************** Finalization, cleaning up ************

	// Free GPU variables
	// Free device allocated memory
	cudaFree(d_input);
	cudaFree(d_image_out[0]);
	cudaFree(d_image_out[1]);
}
