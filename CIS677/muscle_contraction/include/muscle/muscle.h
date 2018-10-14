#ifndef MUSCLE_MUSCLE_H
#define MUSCLE_MUSCLE_H

#include <tuple>
#include <vector>

namespace muscle {
class Muscle {
  public:
    Muscle(unsigned int vector_size);
    __global__ void Run(
        std::vector<unsigned int> force,
        std::vector<unsigned int> distance,
        unsigned long long *product);
  private:
    std::vector<unsigned int> force;
    std::vector<unsigned int> distance;
    unsigned int vector_size;

    // Calculate the dot product of two vectors
    __host__ __device__ unsigned long long compute(
        const std::vector<unsigned int>& v1,
        const std::vector<unsigned int>& v2);
    __host__ __device__ std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> create_vectors(
        int vector_size);
};
} // namespace muscle

#endif
