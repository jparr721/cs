#include "../include/muscle/muscle.h"

#include <cstdlib>
#include <iostream>

__global__ void Run(
    unsigned int* force,
    unsigned int* distance,
    unsigned int vector_size
    ) {
  unsigned long long product;

  for (int i = 0; i < vector_size; ++i) {
    product += force[i] * distance[i];
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: muscle vector_size threads_per_block " << std::endl;
  }

  unsigned int vector_size = atoi(argv[1]);
  unsigned int threads_per_block = atoi(argv[2]);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Error! " << cudaGetErrorString(error) << std::endl;
    return EXIT_FAILURE;
  }

  muscle::Muscle m(vector_size);
  const auto vecs = m.create_vectors(vector_size);
  const std::vector<unsigned int> force = std::get<0>(vecs);
  const std::vector<unsigned int> distance = std::get<1>(vecs);
  unsigned int *force_data;
  unsigned int *distance_data;
  cudaMalloc((void**) &force_data, sizeof(unsigned int) * force.size());
  cudaMalloc((void**) &distance_data, sizeof(unsigned int) * distance.size());

  std::copy(force.begin(), force.end(), force_data);
  std::copy(distance.begin(), distance.end(), distance_data);

  Run <<<ceil((float) vector_size / threads_per_block), threads_per_block>>>(force_data, distance_data, vector_size);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Error! " << cudaGetErrorString(error) << std::endl;
  }

  return EXIT_SUCCESS;
}
