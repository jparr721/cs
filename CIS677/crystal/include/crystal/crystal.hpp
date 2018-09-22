#ifndef CRYSTAL_CRYSTAL_H
#define CRYSTAL_CRYSTAL_H

#include <vector>

namespace crystal {
  class Crystal {
  public:
    std::vector<std::vector<int>> simulation_space;
    explicit Crystal(std::vector<int> board_size);
    ~Crystal() = default;

    void Run(int iterations);

    virtual void update(int notification);

    int get_particles();

  private:
    int particles;
    void init(int number_of_particles, std::vector<int> dimensions);
  };
} // namespace crystal

#endif
