#ifndef CRYSTAL_CRYSTAL_HPP
#define CRYSTAL_CRYSTAL_HPP

#include <tuple>
#include <vector>

namespace crystal {
  class Crystal {
  public:
    const int ROWS = 100;
    const int COLS = 100;
    const int MAX_MOVES = 200;
    const std::tuple<int, int> ORIGIN = std::make_tuple(ROWS / 2, COLS / 2);

    Crystal() = default;
    ~Crystal() = default;

    void Run(int);
    void end_simulation(const std::vector<std::vector<int>>&);
    std::tuple<int, int> insert_particle(const std::vector<std::vector<int>>&);
    bool valid_coordinates(const std::vector<std::vector<int>>&, int, int);
    void random_walk(std::tuple<int, int>, std::vector<std::vector<int>>&);
    bool collision(int, int, const std::vector<std::vector<int>>&);
  };
} // namespace crystal

#endif
