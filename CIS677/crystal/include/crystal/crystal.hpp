#ifndef CRYSTAL_CRYSTAL_HPP
#define CRYSTAL_CRYSTAL_HPP

#include <tuple>
#include <vector>

namespace crystal {
  class Crystal {
  public:
    int ROWS;
    int COLS;
    int SIMULATION_SIZE;
    int MAX_MOVES;
    int CENTER;

    Crystal(int, int);
    ~Crystal() = default;

    void Run(int);
    void end_simulation(const std::vector<std::vector<int>>&);
    std::tuple<int, int> insert_particle(int);
    bool valid_coordinates(const std::vector<std::vector<int>>&, int, int);
    void random_walk(int&, int&, std::vector<std::vector<int>>&);
    bool collision(int, int, const std::vector<std::vector<int>>&);
    void print(const std::vector<std::vector<int>>&);
  };
} // namespace crystal

#endif
