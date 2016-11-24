#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

int main() {
  cudaError_t cuda_status;
  int num;
  cuda_status = cudaGetDeviceCount(&num);
  if (cuda_status != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount error\n");
    return -1;
  }

  cudaDeviceProp prop;
  for (int i = 0; i < num; i++) {
    cuda_status = cudaGetDeviceProperties(&prop, i);
    if (cuda_status != cudaSuccess) {
      fprintf(stderr, "cudaGetDeviceProperties error\n");
      continue;
    }

    printf("%d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Device memory: %ld\n", prop.totalGlobalMem);
    printf("Version: %d.%d\n", prop.major, prop.minor);
    printf("clockRate: %d\n", prop.clockRate);
    printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
  }

  return 0;
}
