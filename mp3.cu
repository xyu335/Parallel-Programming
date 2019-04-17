#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH (4)
// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];
  
  
  // in block index, tx col, ty row
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int th_row, th_col, tile_upperbound; 
  
  // the upper bound of the tile index
  tile_upperbound = ceil((numAColumns * 1.0) / TILE_WIDTH);
  th_col = blockIdx.x * TILE_WIDTH + tx;
  th_row = blockIdx.y * TILE_WIDTH + ty;
  
  // boundary check
  //if (th_row >= numCRows || th_col >= numCColumns) return;
  
  int i,j;
  float tmp = 0;
  // int boundary;
  
  //for (i = 0; i < tile_upperbound; i++)
    for (i = 0; i < (numAColumns - 1) / TILE_WIDTH + 1; ++i)
  {
    
    // memcpy to the shared memory
    if (th_row < numARows && (i * TILE_WIDTH + tx) < numAColumns)
    {
       shared_A[ty][tx] = A[numAColumns * th_row + i * TILE_WIDTH + tx];
    }
    else shared_A[ty][tx] = 0;
   
    if (th_col < numBColumns && (i * TILE_WIDTH + ty) < numBRows)
    {
      shared_B[ty][tx] = B[numBColumns * (TILE_WIDTH * i + ty) + th_col];  
    }
    else shared_B[ty][tx] = 0;
    
    
    // all threads has to load its corresponding memory
    __syncthreads();
    
    // boundary check 
    for (j = 0; j < TILE_WIDTH; j++)
    {
      tmp += shared_A[ty][j] * shared_B[j][tx];
    }
    __syncthreads();
    
  }
  // C element updated inside the for loop of tile index
  if (th_row < numCRows && th_col < numCColumns)
    {
      C[th_row * numCColumns + th_col]  = tmp;  
    }
  printf("block %d %d,thread %d %d, value %f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tmp);  
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  
  // document memory space needed for three device
  // supppose no overflow of the size, int => size_t
  int capA = numARows * numAColumns * sizeof(float);
  int capB = numBRows * numBColumns * sizeof(float);
  int capC = numCRows * numCColumns * sizeof(float);
  
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA, capA);
  cudaMalloc((void **)&deviceB, capB);
  cudaMalloc((void **)&deviceC, capC);
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, capA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, capB, cudaMemcpyHostToDevice);
  
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((numCColumns*1.0)/TILE_WIDTH), ceil(( numCRows* 1.0)/TILE_WIDTH), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                                     numARows, numAColumns,
                                     numBRows, numBColumns,
                                     numCRows, numCColumns);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, capC, cudaMemcpyDeviceToHost);
  

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

