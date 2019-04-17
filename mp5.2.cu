// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

/*
The purpose of this lab is to implement one or more kernels and their associated host code to perform parallel scan on a 1D list. 
The scan operator used will be addition. You should implement the work- efficient kernel discussed in lecture.
Your kernel should be able to handle input lists of arbitrary length. However, for simplicity, you can assume that the input list will be at most 2,048 * 2,048 elements.

0 allocate mem
1 implement the work-efficient scan kernel to generate per-block scan array and store the block sums into an auxiliary block sum array.
2 use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory
3 reuse the kernel to perform scan on the auxiliary block sum array to translate the elements into accumulative block sums. Note that this kernel will be launched with only one block.
4 implement the kernel that adds the accumulative block sums to the appropriate elements of the per-block scan array to complete the scan for all the elements.
*/

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // printf("KERNEL SCAN ENTERED...\n");
  __shared__ float buffer[BLOCK_SIZE * 2];
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = bx * BLOCK_SIZE + tx;
  int dataIndex = bx * BLOCK_SIZE * 2 + tx;
  
  // load data 
  if (dataIndex >= len) buffer[tx] = 0;
  else  buffer[tx] = input[dataIndex];
  if (dataIndex + BLOCK_SIZE >= len) buffer[tx + BLOCK_SIZE] = 0;
  else buffer[tx + BLOCK_SIZE] = input[dataIndex + BLOCK_SIZE];

  __syncthreads();
  
  int stride = 1;
  while (stride < BLOCK_SIZE * 2)  // stride less than 
  {
    // forward reduction tree, first round: 1, 3, 5,7 | second round: 3, 7
    int idx = (tx + 1) * stride * 2 - 1;
    if (idx < 2 * BLOCK_SIZE && idx - stride >= 0)
      buffer[idx] += buffer[idx - stride];
    stride *= 2; // stride doubled since every two nodes reduces to one 
    __syncthreads();
  }
  
  stride = BLOCK_SIZE / 2;
  while (stride > 0)
  {
    int idx = (tx + 1) * stride * 2 - 1;
    if (idx + stride < 2 * BLOCK_SIZE)
       buffer[idx + stride] += buffer[idx]; // idx + stride == 2 * BLOCK_SIZE, edge
    stride /= 2;
    __syncthreads();
  }
  
  // write out the elements
  if (dataIndex < len) output[dataIndex] = buffer[tx];
  if (dataIndex + BLOCK_SIZE < len) output[dataIndex + BLOCK_SIZE] = buffer[tx + BLOCK_SIZE];
  
  return;
}

/* add Sum Array element(sum_arr[blockIdx.x]) to all the elements in the block */
__global__ void helper(float * input, float * output, float * arrsum, int len, int len_arrsum)
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int index = tx + bx * BLOCK_SIZE; 
  int startIndex = index * BLOCK_SIZE * 2;
  printf("bx:%d\t tx:%d\t startIndex:%d\t \n", bx, tx, startIndex);
  /*if (tx == 0)
  {
    printf("addon: %f, %f, %f...\n", arrsum[0], arrsum[1], arrsum[2]);
    for (int i = 0; i < 100; ++i)
    {
      printf("%dth: %f \t", input[i]);
     
    }
    printf("\n");
  }*/
  
  if (index < len_arrsum)
  {
    float addon = 0;
    if (index > 0) addon = arrsum[index - 1]; // first thread use arrsum[0] 
    for (int i = startIndex; i < startIndex + BLOCK_SIZE * 2; ++i)
    {
      output[i] = input[i] + addon; 
      printf("%dth, output: %f input: %f \n", i, output[i], input[i]);
    }
  }
  
  __syncthreads();
  return;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements); // host memory allocation
  hostOutput = (float *)malloc(numElements * sizeof(float));
  // mmst is not allowed
  wbTime_stop(Generic, "Importing data and creating memory on host");
  wbLog(TRACE, "The number of input elements in the input is ", numElements); // numbers argument
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));// GPU memory
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float))); // device mmst
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));// deviceInput => hostInput

  
  printf("************ Test Data ************: \n");
  for (int i = 0; i < numElements; ++i)
  {
    printf("%dth: %f \t", i, hostInput[i]);
  }
  
  //fisrt kernel for pre and post scan
  int grid_dim_x = ceil((double)numElements / (2 * BLOCK_SIZE));
  dim3 gridDim(grid_dim_x, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements); // 
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),cudaMemcpyDeviceToHost)); // CC

  // the sum array for each block, 
  float * sumarr = (float *) malloc(grid_dim_x * sizeof(float));
  int startIndex = 0;
  for (int i=0; i < grid_dim_x; ++i)
  {
    startIndex += (BLOCK_SIZE * 2);
    if (startIndex > numElements) sumarr[i] = hostOutput[numElements - 1];
    else sumarr[i] = hostOutput[startIndex-1];
  }

  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float))); // device mmst
  cudaMemcpy(deviceInput, sumarr, grid_dim_x * sizeof(float), cudaMemcpyHostToDevice); // CC
  dim3 gridDim_sum(ceil((double) grid_dim_x / (2 * BLOCK_SIZE)), 1, 1);
  dim3 blockDim_sum(BLOCK_SIZE, 1, 1); 
  scan<<<gridDim_sum, blockDim_sum>>>(deviceInput, deviceOutput, grid_dim_x);
  cudaDeviceSynchronize();
  cudaMemcpy(sumarr, deviceOutput, grid_dim_x * sizeof(float), cudaMemcpyDeviceToHost); // CC
  
  // start last kernel for aggregate, for each thread if there are more than one block in kernel
  float * device_arrsum;
  cudaMalloc((void **) & device_arrsum, grid_dim_x * sizeof(float));
  
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float))); // device mmst
  wbCheck(cudaMemcpy(deviceInput, hostOutput, numElements * sizeof(float), cudaMemcpyHostToDevice)); // CC
  wbCheck(cudaMemcpy(device_arrsum, sumarr, grid_dim_x * sizeof(float), cudaMemcpyHostToDevice));
  int block_last = ceil((double) (grid_dim_x-1) / BLOCK_SIZE);
  printf("LAST KERNEL: block number: %d\n", block_last);
  dim3 gridDim_last(block_last, 1, 1); // total amout of thread needed, 
  dim3 blockDim_last(BLOCK_SIZE, 1, 1); // BLOCK_SIZE
  helper<<<gridDim_last, blockDim_last>>>(deviceInput, deviceOutput, device_arrsum, numElements, grid_dim_x); //helper kernel, <1,1,1> <512,1,1>
  cudaDeviceSynchronize();
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),cudaMemcpyDeviceToHost)); // CC grid_dim_x => numElements
  cudaFree(device_arrsum);
  }
  
  wbTime_start(GPU, "Freeing GPU Memory");
  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(sumarr);

  return 0;
}
