#include "../include/muscle/muscle.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: muscle vector_size threads_per_block " << std::endl;
  }

  unsigned int vector_size = atoi(argv[1]);
  unsigned int threads_per_block = atoi(argv[2]);
  unsigned long long *product;

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Error! " << cudaGetErrorString(error) << std::endl;
    return EXIT_FAILURE;
  }

  muscle::Muscle m(vector_size);
  const auto vecs = m.create_vectors(vector_size);
  const std::vector<unsigned int> force = std::get<0>(vecs);
  const std::vector<unsigned int> distance = std::get<1>(vecs);

  m.Run <<<ceil((float) vector_size / threads_per_block), threads_per_block>>>(force, distance, product);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "Error! " << cudaGetErrorString(error) << std::endl;
  }

  return EXIT_SUCCESS;
}
