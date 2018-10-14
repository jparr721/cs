#include "../include/muscle/muscle.h"


namespace muscle {
  Muscle::Muscle(unsigned int vector_size) {
    this->vector_size = vector_size;
  }

  __global__
  void Muscle::Run(
      std::vector<unsigned int> force,
      std::vector<unsigned int> distance,
      unsigned long long *product
      ) {

    *product = compute(force, distance);
  }

  __host__ __device__ unsigned long long Muscle::compute(
      const std::vector<unsigned int>& v1,
      const std::vector<unsigned int>& v2
      ) {
    unsigned long long product = 0;

    for (int i = 0; i < this->vector_size; ++i) {
      product += v1[i] * v2[i];
    }

    return product;
  }

  __host__ __device__ std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> create_vectors(
      int vector_size
      ) {
    int force_vector_range = vector_size / 2;
    std::vector<unsigned int> force;
    force.reserve(vector_size);

    std::vector<unsigned int> distance;
    distance.reserve(vector_size);

    for (unsigned int i = 0; i < force_vector_range; ++i) {
      force.push_back((i + 1));
    }

    for (unsigned int i = force_vector_range; i > 0; --i) {
      force.push_back((i + 1));
    }

    for (unsigned int i = 0; i < vector_size; ++i) {
      distance.push_back((i % 10) + 1);
    }

    return std::make_tuple(force, distance);
  }
} // namespace muscle
