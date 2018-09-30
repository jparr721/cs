#ifndef CRYSTAL_CRYSTAL_HPP
#define CRYSTAL_CRYSTAL_HPP

#include <tuple>
#include <vector>

namespace crystal {
  class Crystal {
  public:
    long int ROWS;
    long int COLS;
    long int SIMULATION_SIZE;
    long int MAX_MOVES;
    long int CENTER;

    Crystal(long int, long int);
    ~Crystal() = default;

    void Run(long int);
    void end_simulation(const std::vector<std::vector<int>>&);
    std::tuple<long int, long int> insert_particle(const std::vector<std::vector<int>>&, int);
    void random_walk(long int&, long int&, std::vector<std::vector<int>>&);
    bool collision(long int, long int, const std::vector<std::vector<int>>&);
    void print(const std::vector<std::vector<int>>&);
  };
} // namespace crystal

#endif
