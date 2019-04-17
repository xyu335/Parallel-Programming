#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

__global__ void kernel_float2uchar(float * original_image, unsigned char * output_gray_image, unsigned char * output_uchar_image, int size)
{
  // 
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = tx + bx * BLOCK_SIZE;
  if (index >= size) return;
  
  float r = original_image[3 * index];
  float g = original_image[3 * index + 1];
  float b = original_image[3 * index + 2];
  unsigned char ri = (unsigned char) (r * 255.0);
  unsigned char gi = (unsigned char) (g * 255.0);
  unsigned char bi = (unsigned char) (b * 255.0);
  
  output_uchar_image[3 * index] = ri;
  output_uchar_image[3 * index + 1] = gi;
  output_uchar_image[3 * index + 2] = bi;
  unsigned char tmp = (unsigned char) (0.21 * ri + 0.71 * gi + 0.07 * bi);
  output_gray_image[index] = tmp;
}

__global__ void kernel_atomic(unsigned char * input_gray_image, unsigned int * output_bucket, int size)
{
  // calculate val's frequencies accross the image
  __shared__ unsigned int privateBucket[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = tx + bx * BLOCK_SIZE;
  int stride = BLOCK_SIZE;
  int idx = index;
  privateBucket[tx] = 0;
  __syncthreads();
  // index => 0, 255
  while (idx < size)
  {
    atomicAdd(&privateBucket[input_gray_image[idx]], 1);
    if (input_gray_image[idx] == 0) printf("gray image val == 0! th: %d \n", idx);
    idx+=stride;
  }
  __syncthreads();
  
  output_bucket[tx] = privateBucket[tx];
}

// kernel_correct<<<GridDim, BlockDim>>>(device_histogram_cdf, device_final_image, outputImageSize);
__global__ void kernel_correct(unsigned char * input_image, float * corrected_cdf, float * outputImage, int size)
{
  // 1 thread handle 3 correction
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  unsigned int index = tx + bx * BLOCK_SIZE;
  if (index < size)
  {
    unsigned int startIdx = index * 3;
    for (int i = 0; i < 3; ++i)
    {
      outputImage[startIdx] = corrected_cdf[input_image[startIdx]];
      ++startIdx;
    }
  }
  __syncthreads();
}

/* tool functions */
void scan(unsigned int * input, float * histogram_cdf, unsigned int outputImageSize){
  // do scan prefix sum here
  unsigned int cum = 0;
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
  {
    cum += input[i];
    histogram_cdf[i] = (float) cum / outputImageSize;
    printf("%dth val: %f \n", i, histogram_cdf[i]);
  }
  return;
}

void correct(float min, float max, float * histogram_cdf)
{
  float range = max - min; 
  printf("final mapping for the image... \n");
  for (int i = 0; i < HISTOGRAM_LENGTH; ++i)
  {
    histogram_cdf[i] = ((histogram_cdf[i] - min) / range);
    printf("%dth val: %f \n",i, histogram_cdf[i]);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  // added lines for transform the data from wbImage_t to hostImageData
  hostInputImageData = wbImage_getData(inputImage); 
  hostOutputImageData = wbImage_getData(outputImage); 
  wbTime_stop(Generic, "Importing data and creating memory on host");

  /* device mem allocation & copy */
  float * deviceInputImage;
  unsigned char * deviceOutputImage_uchar; // 3 channel
  unsigned char * deviceOutputImage_gray;
  unsigned int * deviceOutputBucket;
  unsigned int * histogram;
  float * histogram_cdf;
  float * device_histogram_cdf;
  float * device_final_image;
  
  
  unsigned int inputImageSize = imageWidth * imageHeight * imageChannels;
  unsigned int outputImageSize = imageWidth * imageHeight;
  cudaMalloc((void **) &deviceInputImage, sizeof(float) * inputImageSize);
  cudaMalloc((void **) &deviceOutputImage_uchar, sizeof(unsigned char) * inputImageSize);
  cudaMalloc((void **) &deviceOutputImage_gray, sizeof(unsigned char) * outputImageSize);
  cudaMalloc((void **) &deviceOutputBucket, sizeof(unsigned int) * HISTOGRAM_LENGTH);
  histogram = (unsigned int *) malloc(sizeof(unsigned int) * HISTOGRAM_LENGTH);
  histogram_cdf = (float *) malloc(sizeof(float) * HISTOGRAM_LENGTH);
  cudaMalloc((void **) &device_histogram_cdf, sizeof(float) * HISTOGRAM_LENGTH); 
  cudaMalloc((void **) &device_final_image, sizeof(float) * inputImageSize); // final image should be 3 channels
  printf("Image import finished: \
         \n height * width * channel: %d, %d, %d \
         \n inputImageSize and output: %d, %d \n" \
         , imageHeight, imageWidth, imageChannels, inputImageSize, outputImageSize);   
  /* transform image into unsigned char value data 
      RGB => grayscale */ 
	/* device memory == pixels * unsigned char */
  cudaMemcpy(deviceInputImage, hostInputImageData, sizeof(float) * inputImageSize, cudaMemcpyHostToDevice);
  int blockNumber = ceil((double) outputImageSize / BLOCK_SIZE);
  
  dim3 GridDim(blockNumber, 1, 1);
  dim3 BlockDim(BLOCK_SIZE, 1, 1);
  kernel_float2uchar<<<GridDim, BlockDim>>>(deviceInputImage, deviceOutputImage_gray, deviceOutputImage_uchar, outputImageSize); 
  cudaDeviceSynchronize();
  
  /* compute histogram of the image */ /* bucket calculation */ 
  // input: luminous array  => output:  bucket[255] 
  dim3 GridDim_1(1, 1, 1);
  kernel_atomic<<<GridDim_1, BlockDim>>>(deviceOutputImage_gray, deviceOutputBucket, outputImageSize);
  cudaDeviceSynchronize();
  cudaMemcpy(histogram, deviceOutputBucket, sizeof(unsigned int) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);
  
  /* scan histogram */
  // input: bucket[255] => output: accumulated histogram acc_bucket[255]
  scan(histogram, histogram_cdf, outputImageSize);
  float cdf_min = histogram_cdf[0];
  float cdf_max = 1.0;
  correct(cdf_min, cdf_max, histogram_cdf);
  printf("histogram cdf min&max: %f, %f\n", cdf_min, cdf_max);
  
  
  /* transform input image with histogram equalization function  */
  // input: corrected 
  cudaMemcpy(device_histogram_cdf, histogram_cdf, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyHostToDevice);
  kernel_correct<<<GridDim, BlockDim>>>(deviceOutputImage_uchar, device_histogram_cdf, device_final_image, outputImageSize);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, device_final_image, sizeof(float) * inputImageSize, cudaMemcpyDeviceToHost);
  
  wbImage_setData(outputImage, hostOutputImageData);
  // free cuda
  cudaFree(deviceInputImage);
  cudaFree(deviceOutputImage_uchar);
  cudaFree(deviceOutputImage_gray);
  cudaFree(deviceOutputBucket);
  cudaFree(device_histogram_cdf);
  cudaFree(device_final_image);
  
  // modify the outputImage
  wbSolution(args, outputImage);

  // free memory
  free(histogram);
  free(histogram_cdf);

  return 0;
}
