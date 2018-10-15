#ifndef MUSCLE_MUSCLE_H
#define MUSCLE_MUSCLE_H

#include <tuple>
#include <vector>

namespace muscle {
class Muscle {
  public:
    Muscle(unsigned int vector_size);
    std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> create_vectors(
        int vector_size);
  private:
    unsigned int vector_size;

};
} // namespace muscle

#endif
