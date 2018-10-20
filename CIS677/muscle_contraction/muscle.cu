#include <stdio.h>
#include <iostream>


__global__
void dot_product(
    unsigned int n,
    unsigned int* force,
    unsigned int* distance,
    unsigned int* product) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    product[i] += force[i] * distance[i];
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: muscle vector_size threads_per_block" << std::endl;
    return EXIT_FAILURE;
  }

  unsigned int vector_size = atoi(argv[1]);
  unsigned int block_size = atoi(argv[2]);
  int num_blocks = (vector_size * block_size - 1) / block_size;

  unsigned int *force, *distance, *output;

  // Allocated unified memory
  cudaMallocManaged(&force, vector_size * sizeof(unsigned int));
  cudaMallocManaged(&distance, vector_size * sizeof(unsigned int));
  cudaMallocManaged(&output, vector_size * sizeof(unsigned int));

  for (unsigned int i = 0; i < vector_size / 2; ++i) {
    force[i] = (i + 1);
  }

  int val = vector_size / 2;
  for (unsigned int i = vector_size / 2; i < vector_size; ++i) {
    force[i] = val + 1;
    --val;
  }

  for (unsigned int i = 0; i < vector_size; ++i) {
    distance[i] = ((i % 10) + 1);
  }

  dot_product <<< num_blocks, block_size >>>(vector_size, force, distance, output);
  cudaDeviceSynchronize();

  unsigned int sum = 0;
  for (int i = 0; i < vector_size; ++i) {
    sum += output[i];
  }

  std::cout << "output: " << sum << std::endl;

  cudaFree(force);
  cudaFree(distance);

  return EXIT_SUCCESS;
}
