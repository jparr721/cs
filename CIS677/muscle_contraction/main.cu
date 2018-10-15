#include <cstdlib>
#include <iostream>
#include <tuple>

__global__ void Run(
    unsigned int *force,
    unsigned int *distance,
    unsigned int *product,
    unsigned int n
    ) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < n) {
    product += force[id] * distance[id];
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: muscle vector_size threads_per_block " << std::endl;
    return EXIT_FAILURE;
  }

  unsigned int vector_size = atoi(argv[1]);
  unsigned int threads_per_block = atoi(argv[2]);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Error! " << cudaGetErrorString(error) << std::endl;
    return EXIT_FAILURE;
  }
  // host vectors
  unsigned int* h_force;
  unsigned int* h_distance;
  // host output vector
  unsigned int* h_output;


  // device input vectors
  unsigned int* d_force;
  unsigned int* d_distance;
  // device output vector
  unsigned int* d_output;

  size_t bytes = vector_size*sizeof(unsigned int);

  h_force = (unsigned int*)malloc(bytes);
  h_distance = (unsigned int*)malloc(bytes);
  h_output = (unsigned int*)malloc(bytes);

  // Allocate cuda memory
  cudaMalloc(&d_force, bytes);
  cudaMalloc(&d_distance, bytes);
  cudaMalloc(&d_output, bytes);

  for (unsigned int i = 0; i < vector_size / 2; ++i) {
    h_force[i] = (i + 1);
  }
  for (unsigned int i = vector_size / 2; i > 0; --i) {
    h_force[i] = (i + 1);
  }
  for (unsigned int i = 0; i < vector_size; ++i) {
    h_distance[i] = ((i % 10) + 1);
  }

  cudaMemcpy(d_force, h_force, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_distance, h_distance, bytes, cudaMemcpyHostToDevice);

  int g = (int)ceil((float) vector_size / threads_per_block);

  Run <<< g, threads_per_block>>>(d_force, d_distance, d_output, vector_size);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Error! " << cudaGetErrorString(error) << std::endl;
  }

  cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

  unsigned int sum = 0;
  for (int i = 0; i < vector_size; ++i) {
    sum += h_output[i];
  }

  std::cout << "Final result: " << sum / vector_size << std::endl;

  cudaFree(d_force);
  cudaFree(d_distance);
  cudaFree(d_output);

  free(h_force);
  free(h_distance);
  free(h_output);
  return EXIT_SUCCESS;
}
