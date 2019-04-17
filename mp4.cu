#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 3
#define RADIUS 1
#define INPUT_WIDTH 5

//@@ Define constant memory for device kernel here
// the shared cache
__constant__ float M[TILE_WIDTH * TILE_WIDTH * TILE_WIDTH];


// input should be the same size
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  
  // load data, using strategy 3, load the core, judge the boundary and read halo direct from the global memory
  __shared__ float N_sds[INPUT_WIDTH][INPUT_WIDTH][INPUT_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int bx = blockIdx.x * TILE_WIDTH;
  int by = blockIdx.y * TILE_WIDTH;
  int bz = blockIdx.z * TILE_WIDTH;
  
  int kt = tz * TILE_WIDTH * TILE_WIDTH + ty * TILE_WIDTH + tx;
  
  if (kt < INPUT_WIDTH * INPUT_WIDTH){
    // calculate the tile index
    int tile_x = kt % (INPUT_WIDTH);
    int tile_y = (kt / INPUT_WIDTH) % INPUT_WIDTH;
    
    for (int i =0; i < INPUT_WIDTH; ++i){
      int z_start = bz - 1 + i;
      if (bx + tile_x - 1 < x_size && bx + tile_x - 1 >= 0 && by + tile_y - 1 < y_size && by + tile_y - 1 >= 0 && z_start < z_size && z_start >= 0){
        N_ds[bx + tile_x - 1][by + tile_y - 1][i] = input[z_start * (y_size * x_size) + (by + tile_y -1) * x_size + bx+tile_x-1];
      }else {
        N_ds[bx + tile_x - 1][by + tile_y - 1][i] = 0;
      }
    }
    
  }
  __syncthreads();
  
  if (bx + tx >= 0 && bx + tx < x_size && by + ty >= 0 && by + ty < y_size && bz + tz >= 0 && bz + tz < z_size)
  {
    float P = 0;
    for (int x = 0; x < TILE_WIDTH; x++){
      for (int y = 0; y < TILE_WIDTH; y++){
        for (int z = 0; z < TILE_WIDTH; z++){
          P += N_ds[tx + x ][ty + y][tz + z] * M[z * (TILE_WIDTH * TILE_WIDTH) + y * (TILE_WIDTH) + x];
        }
      }
    }
    output[(tz + bz) * (y_size * x_size) + (ty + by) * (x_size) + (tx+bx)] = P;
  }
  
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);
  // set the TILE_LENGTH == 3 * 3 * 3
  
  int matrix_len = inputLength - 3;
  int matrix_size = matrix_len * sizeof(float);
  
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, matrix_size);
  cudaMalloc((void **)&deviceOutput, matrix_size);
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpy(deviceInput, hostInput + 3, matrix_size, cudaMemcpyHostToDevice);
  // constant mem declared elsewhere
  cudaMemcpyToSymbol(M, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 GridDim(ceil((1.0 * x_size)/TILE_WIDTH), ceil((1.0 * y_size)/TILE_WIDTH), ceil((1.0 * z_size)/TILE_WIDTH));
  dim3 BlockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, matrix_size, cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

