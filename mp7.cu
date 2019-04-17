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

#define BLOCK_SIZE (256)

/* 
  NNZ: Number of None-Zero elements
  maxRowNNZ: Max NNZ for a row (NNZ for the most dense row)
  ndata: Total NNZ
  dim: The size of vec and width of the original matrix. The original matrix is square
  float matCols[ndata]: Column coordinate in the original matrix. Indexed column-major
  float matData[ndata]: Data value in the original matrix. Indexed column-major
  int matColStart[maxRowNNZ]: The index of the first element in matCols and matData for each column, 
  
  int matRowPerm[dim]: Original row number in CSR for every row in JDS
  float vec[dim]: the dense multiplier vector
  int matRows[dim]: NNZ/size for every row
*/

// appropriate elements of the JDS data array, the JDS col index array, JDS row index array, 
// and the JDS transposed col ptr array to generate one Y element.
__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for jds format
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = bx * BLOCK_SIZE  + tx;
  int maxNNZ = * matRows; // the first one
  
  int current_row_size = 0;
  float sum = 0;
  if (index < dim) current_row_size = matRows[index];
  for (int i = 0; i < current_row_size; ++i)
  {
    // cumulate the multiplication
    int idx = matColStart[i] + index; // idx in the matData matrix
    int col = matCols[idx]; // original 
    // matCols and matData shares the idx, column based == colstart + original row number 
    // the index can be applied here since the size of row is sorted in descendantly. 
    // idx is the allocation of the point in the JDS data layout(column majored)
    // lookup the original column number and the value by using the index
    sum += vec[col] * matData[idx];
    //this lib is intended for coalesc
  }
  
  __syncthreads();
  if (index < dim) out[matRowPerm[index]] = sum;
 
  return;
}

// spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
//           deviceJDSData, deviceVector, dim);
// same arguments type and name as the other signature 
static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for jds format
  dim3 blockDim(BLOCK_SIZE, 1,1);
  dim3 gridDim(ceil( (double)dim / BLOCK_SIZE) ); // 
  spmvJDSKernel<<<gridDim, blockDim>>>(out, matColStart, matCols, matRowPerm, matRows, matData, vec, dim);
  return;  
}

int main(int argc, char **argv) {
  wbArg_t args;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector; 
  float *hostOutput;
  int *deviceJDSColStart; //  row pointer**
  int *deviceJDSCols; // col_index marker
  int *deviceJDSRowPerm; // permutation indices
  int *deviceJDSRows; // ? ?
  float *deviceJDSData; // data
  float *deviceVector; // vector for multiple 
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 3), &dim, "Real");
  // output for Y
  hostOutput = (float *)malloc(sizeof(float) * dim);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  
  /* conversion to JDS format */
  CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
           &hostJDSColStart, &hostJDSCols, &hostJDSData);
  maxRowNNZ = hostJDSRows[0]; // Number of None-Zero elements? so the col_start == 0, 13, the first two element make sense, and rest of them are garbages in the mem
  // test the converted value
  printf("dim: %d, maxRowNNZ: %d \n", dim, maxRowNNZ); // configuration params
  printf("\ncol printer for CSR, size %d\n", ncols);
  
  /*
  for (int i = 0; i < ncols; ++i)
  {
    printf("%dth -  col: %d \t row: %d \t permrow: %d \t col_start: %d \t data: %f\n", i,  * (hostJDSCols + i), *(hostJDSRows + i), *(hostJDSRowPerm + i), *(hostJDSColStart + 1), *(hostJDSData + i));
  }
  */
  
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
  cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
  cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
  cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);

  cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  // sparse matrix multiple vector function call
  spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
          deviceJDSData, deviceVector, dim);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceVector);
  cudaFree(deviceOutput);
  cudaFree(deviceJDSColStart);
  cudaFree(deviceJDSCols);
  cudaFree(deviceJDSRowPerm);
  cudaFree(deviceJDSRows);
  cudaFree(deviceJDSData);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  free(hostJDSColStart);
  free(hostJDSCols);
  free(hostJDSRowPerm);
  free(hostJDSRows);
  free(hostJDSData);

  return 0;
}

